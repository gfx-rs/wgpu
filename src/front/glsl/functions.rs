use super::{
    ast::*,
    builtins::{inject_builtin, inject_double_builtin, sampled_to_depth},
    context::{Context, ExprPos, StmtContext},
    error::{Error, ErrorKind},
    types::scalar_components,
    Parser, Result, SourceMetadata,
};
use crate::{
    front::glsl::types::type_power, proc::ensure_block_returns, Arena, Block, Constant,
    ConstantInner, EntryPoint, Expression, FastHashMap, Function, FunctionArgument, FunctionResult,
    Handle, LocalVariable, ScalarKind, ScalarValue, Span, Statement, StructMember, Type, TypeInner,
};
use std::iter;

impl Parser {
    fn add_constant_value(
        &mut self,
        scalar_kind: ScalarKind,
        value: u64,
        meta: SourceMetadata,
    ) -> Handle<Constant> {
        let value = match scalar_kind {
            ScalarKind::Uint => ScalarValue::Uint(value),
            ScalarKind::Sint => ScalarValue::Sint(value as i64),
            ScalarKind::Float => ScalarValue::Float(value as f64),
            _ => unreachable!(),
        };

        self.module.constants.fetch_or_append(
            Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Scalar { width: 4, value },
            },
            meta.as_span(),
        )
    }

    pub(crate) fn function_or_constructor_call(
        &mut self,
        ctx: &mut Context,
        stmt: &StmtContext,
        body: &mut Block,
        fc: FunctionCallKind,
        raw_args: &[Handle<HirExpr>],
        meta: SourceMetadata,
    ) -> Result<Option<Handle<Expression>>> {
        let args: Vec<_> = raw_args
            .iter()
            .map(|e| ctx.lower_expect_inner(stmt, self, *e, ExprPos::Rhs, body))
            .collect::<Result<_>>()?;

        match fc {
            FunctionCallKind::TypeConstructor(ty) => {
                let h = if args.len() == 1 {
                    let expr_type = self.resolve_type(ctx, args[0].0, args[0].1)?;

                    let vector_size = match *expr_type {
                        TypeInner::Vector { size, .. } => Some(size),
                        _ => None,
                    };

                    // Special case: if casting from a bool, we need to use Select and not As.
                    match self.module.types[ty].inner.scalar_kind() {
                        Some(result_scalar_kind)
                            if expr_type.scalar_kind() == Some(ScalarKind::Bool)
                                && result_scalar_kind != ScalarKind::Bool =>
                        {
                            let (condition, expr_meta) = args[0];
                            let c0 = self.add_constant_value(result_scalar_kind, 0u64, meta);
                            let c1 = self.add_constant_value(result_scalar_kind, 1u64, meta);
                            let mut reject =
                                ctx.add_expression(Expression::Constant(c0), expr_meta, body);
                            let mut accept =
                                ctx.add_expression(Expression::Constant(c1), expr_meta, body);

                            ctx.implicit_splat(self, &mut reject, meta, vector_size)?;
                            ctx.implicit_splat(self, &mut accept, meta, vector_size)?;

                            let h = ctx.add_expression(
                                Expression::Select {
                                    accept,
                                    reject,
                                    condition,
                                },
                                expr_meta,
                                body,
                            );

                            return Ok(Some(h));
                        }
                        _ => {}
                    }

                    match self.module.types[ty].inner {
                        TypeInner::Vector { size, kind, width } if vector_size.is_none() => {
                            let (mut value, meta) = args[0];
                            ctx.implicit_conversion(self, &mut value, meta, kind, width)?;

                            ctx.add_expression(Expression::Splat { size, value }, meta, body)
                        }
                        TypeInner::Scalar { kind, width } => ctx.add_expression(
                            Expression::As {
                                kind,
                                expr: args[0].0,
                                convert: Some(width),
                            },
                            args[0].1,
                            body,
                        ),
                        TypeInner::Vector { size, kind, width } => {
                            let mut expr = args[0].0;

                            if vector_size.map_or(true, |s| s != size) {
                                expr = ctx.vector_resize(size, expr, args[0].1, body);
                            }

                            ctx.add_expression(
                                Expression::As {
                                    kind,
                                    expr,
                                    convert: Some(width),
                                },
                                args[0].1,
                                body,
                            )
                        }
                        TypeInner::Matrix {
                            columns,
                            rows,
                            width,
                        } => {
                            // TODO: casts
                            // `Expression::As` doesn't support matrix width
                            // casts so we need to do some extra work for casts

                            let (mut value, meta) = args[0];
                            ctx.implicit_conversion(
                                self,
                                &mut value,
                                meta,
                                ScalarKind::Float,
                                width,
                            )?;
                            match *self.resolve_type(ctx, value, meta)? {
                                TypeInner::Scalar { .. } => {
                                    // If a matrix is constructed with a single scalar value, then that
                                    // value is used to initialize all the values along the diagonal of
                                    // the matrix; the rest are given zeros.
                                    let mut components = Vec::with_capacity(columns as usize);
                                    let vector_ty = self.module.types.fetch_or_append(
                                        Type {
                                            name: None,
                                            inner: TypeInner::Vector {
                                                size: rows,
                                                kind: ScalarKind::Float,
                                                width,
                                            },
                                        },
                                        meta.as_span(),
                                    );
                                    let zero_constant = self.module.constants.fetch_or_append(
                                        Constant {
                                            name: None,
                                            specialization: None,
                                            inner: ConstantInner::Scalar {
                                                width,
                                                value: ScalarValue::Float(0.0),
                                            },
                                        },
                                        meta.as_span(),
                                    );
                                    let zero = ctx.add_expression(
                                        Expression::Constant(zero_constant),
                                        meta,
                                        body,
                                    );

                                    for i in 0..columns as u32 {
                                        components.push(
                                            ctx.add_expression(
                                                Expression::Compose {
                                                    ty: vector_ty,
                                                    components: (0..rows as u32)
                                                        .into_iter()
                                                        .map(|r| match r == i {
                                                            true => value,
                                                            false => zero,
                                                        })
                                                        .collect(),
                                                },
                                                meta,
                                                body,
                                            ),
                                        )
                                    }

                                    ctx.add_expression(
                                        Expression::Compose { ty, components },
                                        meta,
                                        body,
                                    )
                                }
                                TypeInner::Matrix { rows: ori_rows, .. } => {
                                    let mut components = Vec::new();

                                    for n in 0..columns as u32 {
                                        let mut vector = ctx.add_expression(
                                            Expression::AccessIndex {
                                                base: value,
                                                index: n,
                                            },
                                            meta,
                                            body,
                                        );

                                        if ori_rows != rows {
                                            vector = ctx.vector_resize(rows, vector, meta, body);
                                        }

                                        components.push(vector)
                                    }

                                    ctx.add_expression(
                                        Expression::Compose { ty, components },
                                        meta,
                                        body,
                                    )
                                }
                                _ => {
                                    let columns =
                                        iter::repeat(value).take(columns as usize).collect();

                                    ctx.add_expression(
                                        Expression::Compose {
                                            ty,
                                            components: columns,
                                        },
                                        meta,
                                        body,
                                    )
                                }
                            }
                        }
                        TypeInner::Struct { .. } => ctx.add_expression(
                            Expression::Compose {
                                ty,
                                components: args.into_iter().map(|arg| arg.0).collect(),
                            },
                            meta,
                            body,
                        ),
                        _ => {
                            self.errors.push(Error {
                                kind: ErrorKind::SemanticError("Bad cast".into()),
                                meta,
                            });

                            args[0].0
                        }
                    }
                } else {
                    let mut components = Vec::with_capacity(args.len());

                    match self.module.types[ty].inner {
                        TypeInner::Matrix {
                            columns,
                            rows,
                            width,
                        } => {
                            let mut flattened =
                                Vec::with_capacity(columns as usize * rows as usize);

                            for (mut arg, meta) in args.iter().copied() {
                                let scalar_components =
                                    scalar_components(&self.module.types[ty].inner);
                                if let Some((kind, width)) = scalar_components {
                                    ctx.implicit_conversion(self, &mut arg, meta, kind, width)?;
                                }

                                match *self.resolve_type(ctx, arg, meta)? {
                                    TypeInner::Vector { size, .. } => {
                                        for i in 0..(size as u32) {
                                            flattened.push(ctx.add_expression(
                                                Expression::AccessIndex {
                                                    base: arg,
                                                    index: i,
                                                },
                                                meta,
                                                body,
                                            ))
                                        }
                                    }
                                    _ => flattened.push(arg),
                                }
                            }

                            let ty = self.module.types.fetch_or_append(
                                Type {
                                    name: None,
                                    inner: TypeInner::Vector {
                                        size: rows,
                                        kind: ScalarKind::Float,
                                        width,
                                    },
                                },
                                meta.as_span(),
                            );

                            for chunk in flattened.chunks(rows as usize) {
                                components.push(ctx.add_expression(
                                    Expression::Compose {
                                        ty,
                                        components: Vec::from(chunk),
                                    },
                                    meta,
                                    body,
                                ))
                            }
                        }
                        _ => {
                            for (mut arg, meta) in args.iter().copied() {
                                let scalar_components =
                                    scalar_components(&self.module.types[ty].inner);
                                if let Some((kind, width)) = scalar_components {
                                    ctx.implicit_conversion(self, &mut arg, meta, kind, width)?;
                                }

                                components.push(arg)
                            }
                        }
                    }

                    ctx.add_expression(Expression::Compose { ty, components }, meta, body)
                };

                Ok(Some(h))
            }
            FunctionCallKind::Function(name) => {
                self.function_call(ctx, stmt, body, name, args, raw_args, meta)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn function_call(
        &mut self,
        ctx: &mut Context,
        stmt: &StmtContext,
        body: &mut Block,
        name: String,
        args: Vec<(Handle<Expression>, SourceMetadata)>,
        raw_args: &[Handle<HirExpr>],
        meta: SourceMetadata,
    ) -> Result<Option<Handle<Expression>>> {
        // If the name for the function hasn't yet been initialized check if any
        // builtin can be injected.
        if self.lookup_function.get(&name).is_none() {
            let declaration = self.lookup_function.entry(name.clone()).or_default();
            inject_builtin(declaration, &mut self.module, &name);
        }

        // Check if any argument uses a double type
        let has_double = args
            .iter()
            .any(|&(expr, meta)| self.resolve_type(ctx, expr, meta).map_or(false, is_double));

        // At this point a declaration is guaranteed
        let declaration = self.lookup_function.get_mut(&name).unwrap();

        if declaration.builtin && !declaration.double && has_double {
            inject_double_builtin(declaration, &mut self.module, &name);
        }

        // Borrow again but without mutability
        let declaration = self.lookup_function.get(&name).unwrap();

        // Helper enum containing the type of conversion need for a call
        #[derive(PartialEq, Eq, Clone, Copy, Debug)]
        enum Conversion {
            // No conversion needed
            Exact,
            // Float to double conversion needed
            FloatToDouble,
            // Int or uint to float conversion needed
            IntToFloat,
            // Int or uint to double conversion needed
            IntToDouble,
            // Other type of conversion needed
            Other,
            // No conversion was yet registered
            None,
        }

        let mut maybe_decl = None;
        let mut old_conversions = vec![Conversion::None; args.len()];
        let mut ambiguous = false;

        'outer: for decl in declaration.overloads.iter() {
            if args.len() != decl.parameters.len() {
                continue;
            }

            let mut exact = true;
            // State of the selection
            // If None we still don't know what is the best declaration
            // If Some(true) the new declaration is better
            // If Some(false) the old declaration is better
            let mut superior = None;
            let mut new_conversions = vec![Conversion::None; args.len()];

            for ((i, decl_arg), call_arg) in decl.parameters.iter().enumerate().zip(args.iter()) {
                use ScalarKind::*;

                if decl.parameters_info[i].depth {
                    sampled_to_depth(
                        &mut self.module,
                        ctx,
                        call_arg.0,
                        call_arg.1,
                        &mut self.errors,
                    )?;
                    self.invalidate_expression(ctx, call_arg.0, call_arg.1)?
                }

                let decl_inner = &self.module.types[*decl_arg].inner;
                let call_inner = self.resolve_type(ctx, call_arg.0, call_arg.1)?;

                if decl_inner == call_inner {
                    new_conversions[i] = Conversion::Exact;
                    continue;
                }

                exact = false;

                let (decl_kind, decl_width, call_kind, call_width) = match (decl_inner, call_inner)
                {
                    (
                        &TypeInner::Scalar {
                            kind: decl_kind,
                            width: decl_width,
                        },
                        &TypeInner::Scalar {
                            kind: call_kind,
                            width: call_width,
                        },
                    ) => (decl_kind, decl_width, call_kind, call_width),
                    (
                        &TypeInner::Vector {
                            kind: decl_kind,
                            size: decl_size,
                            width: decl_width,
                        },
                        &TypeInner::Vector {
                            kind: call_kind,
                            size: call_size,
                            width: call_width,
                        },
                    ) if decl_size == call_size => (decl_kind, decl_width, call_kind, call_width),
                    (
                        &TypeInner::Matrix {
                            rows: decl_rows,
                            columns: decl_columns,
                            width: decl_width,
                        },
                        &TypeInner::Matrix {
                            rows: call_rows,
                            columns: call_columns,
                            width: call_width,
                        },
                    ) if decl_columns == call_columns && decl_rows == call_rows => {
                        (Float, decl_width, Float, call_width)
                    }
                    _ => continue 'outer,
                };

                if type_power(decl_kind, decl_width) < type_power(call_kind, call_width) {
                    continue 'outer;
                }

                let conversion = match ((decl_kind, decl_width), (call_kind, call_width)) {
                    ((Float, 8), (Float, 4)) => Conversion::FloatToDouble,
                    ((Float, 4), (Sint, _)) | ((Float, 4), (Uint, _)) => Conversion::IntToFloat,
                    ((Float, 8), (Sint, _)) | ((Float, 8), (Uint, _)) => Conversion::IntToDouble,
                    _ => Conversion::Other,
                };

                // true - New declaration argument has a better conversion
                // false - Old declaration argument has a better conversion
                let best_arg = match (conversion, old_conversions[i]) {
                    (_, Conversion::Exact) => false,
                    (Conversion::FloatToDouble, _)
                    | (_, Conversion::None)
                    | (Conversion::IntToFloat, Conversion::IntToDouble) => true,
                    (_, Conversion::FloatToDouble)
                    | (Conversion::IntToDouble, Conversion::IntToFloat) => false,
                    _ => continue,
                };

                match best_arg {
                    true => match superior {
                        Some(false) => ambiguous = true,
                        _ => {
                            superior = Some(true);
                            new_conversions[i] = conversion
                        }
                    },
                    false => match superior {
                        Some(true) => ambiguous = true,
                        _ => superior = Some(false),
                    },
                }
            }

            if exact {
                maybe_decl = Some(decl);
                ambiguous = false;
                break;
            }

            match superior {
                // New declaration is better keep it
                Some(true) => {
                    maybe_decl = Some(decl);
                    // Replace the conversions
                    old_conversions = new_conversions;
                }
                // Old declaration is better do nothing
                Some(false) => {}
                // No declaration was better than the other this can be caused
                // when
                None => {
                    ambiguous = true;
                    // Assign the new declaration to make sure we always have
                    // one to make error reporting happy
                    maybe_decl = Some(decl);
                }
            }
        }

        if ambiguous {
            self.errors.push(Error {
                kind: ErrorKind::SemanticError(
                    format!("Ambiguous best function for '{}'", name).into(),
                ),
                meta,
            })
        }

        let decl = maybe_decl.ok_or_else(|| Error {
            kind: ErrorKind::SemanticError(format!("Unknown function '{}'", name).into()),
            meta,
        })?;

        let parameters_info = decl.parameters_info.clone();
        let parameters = decl.parameters.clone();
        let is_void = decl.void;
        let kind = decl.kind;

        let mut arguments = Vec::with_capacity(args.len());
        let mut proxy_writes = Vec::new();
        for (parameter_info, (expr, parameter)) in parameters_info
            .iter()
            .zip(raw_args.iter().zip(parameters.iter()))
        {
            let (mut handle, meta) =
                ctx.lower_expect_inner(stmt, self, *expr, parameter_info.qualifier.as_pos(), body)?;

            if let TypeInner::Vector { size, kind, width } =
                *self.resolve_type(ctx, handle, meta)?
            {
                if parameter_info.qualifier.is_lhs()
                    && matches!(ctx[handle], Expression::Swizzle { .. })
                {
                    let ty = self.module.types.fetch_or_append(
                        Type {
                            name: None,
                            inner: TypeInner::Vector { size, kind, width },
                        },
                        Span::Unknown,
                    );
                    let temp_var = ctx.locals.append(
                        LocalVariable {
                            name: None,
                            ty,
                            init: None,
                        },
                        Span::Unknown,
                    );
                    let temp_expr = ctx.add_expression(
                        Expression::LocalVariable(temp_var),
                        SourceMetadata::none(),
                        body,
                    );

                    body.push(
                        Statement::Store {
                            pointer: temp_expr,
                            value: handle,
                        },
                        Span::Unknown,
                    );

                    arguments.push(temp_expr);
                    proxy_writes.push((*expr, temp_expr));
                    continue;
                }
            }

            let scalar_components = scalar_components(&self.module.types[*parameter].inner);
            if let Some((kind, width)) = scalar_components {
                ctx.implicit_conversion(self, &mut handle, meta, kind, width)?;
            }

            arguments.push(handle)
        }

        match kind {
            FunctionKind::Call(function) => {
                ctx.emit_flush(body);

                let result = if !is_void {
                    Some(ctx.add_expression(Expression::CallResult(function), meta, body))
                } else {
                    None
                };

                body.push(
                    crate::Statement::Call {
                        function,
                        arguments,
                        result,
                    },
                    meta.as_span(),
                );

                ctx.emit_start();
                for (tgt, pointer) in proxy_writes {
                    let value = ctx.add_expression(Expression::Load { pointer }, meta, body);
                    let target = ctx
                        .lower_expect_inner(stmt, self, tgt, ExprPos::Rhs, body)?
                        .0;

                    ctx.emit_flush(body);
                    ctx.emit_start();

                    body.push(
                        Statement::Store {
                            pointer: target,
                            value,
                        },
                        meta.as_span(),
                    );
                }

                Ok(result)
            }
            FunctionKind::Macro(builtin) => builtin
                .call(self, ctx, body, arguments.as_mut_slice(), meta)
                .map(Some),
        }
    }

    pub(crate) fn add_function(
        &mut self,
        ctx: Context,
        name: String,
        result: Option<FunctionResult>,
        mut body: Block,
        meta: SourceMetadata,
    ) {
        if self.lookup_function.get(&name).is_none() {
            let declaration = self.lookup_function.entry(name.clone()).or_default();
            inject_builtin(declaration, &mut self.module, &name);
        }

        ensure_block_returns(&mut body);

        let void = result.is_none();

        let &mut Parser {
            ref mut lookup_function,
            ref mut module,
            ..
        } = self;

        let declaration = lookup_function.entry(name.clone()).or_default();

        let Context {
            expressions,
            locals,
            arguments,
            parameters,
            parameters_info,
            ..
        } = ctx;

        if declaration.builtin
            && !declaration.double
            && parameters
                .iter()
                .any(|ty| is_double(&module.types[*ty].inner))
        {
            inject_double_builtin(declaration, module, &name);
        }

        let function = Function {
            name: Some(name),
            arguments,
            result,
            local_variables: locals,
            expressions,
            named_expressions: FastHashMap::default(),
            body,
        };

        'outer: for decl in declaration.overloads.iter_mut() {
            if parameters.len() != decl.parameters.len() {
                continue;
            }

            for (new_parameter, old_parameter) in parameters.iter().zip(decl.parameters.iter()) {
                let new_inner = &module.types[*new_parameter].inner;
                let old_inner = &module.types[*old_parameter].inner;

                if new_inner != old_inner {
                    continue 'outer;
                }
            }

            if decl.defined {
                return self.errors.push(Error {
                    kind: ErrorKind::SemanticError("Function already defined".into()),
                    meta,
                });
            }

            decl.defined = true;
            decl.parameters_info = parameters_info;
            match decl.kind {
                FunctionKind::Call(handle) => *self.module.functions.get_mut(handle) = function,
                FunctionKind::Macro(_) => {
                    let handle = module.functions.append(function, meta.as_span());
                    decl.kind = FunctionKind::Call(handle)
                }
            }
            return;
        }

        let handle = module.functions.append(function, meta.as_span());
        declaration.overloads.push(Overload {
            parameters,
            parameters_info,
            kind: FunctionKind::Call(handle),
            defined: true,
            void,
        });
    }

    pub(crate) fn add_prototype(
        &mut self,
        ctx: Context,
        name: String,
        result: Option<FunctionResult>,
        meta: SourceMetadata,
    ) {
        if self.lookup_function.get(&name).is_none() {
            let declaration = self.lookup_function.entry(name.clone()).or_default();
            inject_builtin(declaration, &mut self.module, &name);
        }

        let void = result.is_none();

        let &mut Parser {
            ref mut lookup_function,
            ref mut module,
            ..
        } = self;

        let declaration = lookup_function.entry(name.clone()).or_default();

        let Context {
            arguments,
            parameters,
            parameters_info,
            ..
        } = ctx;

        if declaration.builtin
            && !declaration.double
            && parameters
                .iter()
                .any(|ty| is_double(&module.types[*ty].inner))
        {
            inject_double_builtin(declaration, module, &name);
        }

        let function = Function {
            name: Some(name),
            arguments,
            result,
            ..Default::default()
        };

        'outer: for decl in declaration.overloads.iter() {
            if parameters.len() != decl.parameters.len() {
                continue;
            }

            for (new_parameter, old_parameter) in parameters.iter().zip(decl.parameters.iter()) {
                let new_inner = &module.types[*new_parameter].inner;
                let old_inner = &module.types[*old_parameter].inner;

                if new_inner != old_inner {
                    continue 'outer;
                }
            }

            return self.errors.push(Error {
                kind: ErrorKind::SemanticError("Prototype already defined".into()),
                meta,
            });
        }

        let handle = module.functions.append(function, meta.as_span());
        declaration.overloads.push(Overload {
            parameters,
            parameters_info,
            kind: FunctionKind::Call(handle),
            defined: false,
            void,
        });
    }

    pub(crate) fn add_entry_point(
        &mut self,
        function: Handle<Function>,
        global_init_body: Block,
        mut expressions: Arena<Expression>,
    ) {
        let mut arguments = Vec::new();
        let mut body = Block::with_capacity(
            // global init body
            global_init_body.len() +
                        // prologue and epilogue
                        self.entry_args.len() * 2
                        // Call, Emit for composing struct and return
                        + 3,
        );

        for arg in self.entry_args.iter() {
            if arg.storage != StorageQualifier::Input {
                continue;
            }

            let ty = self.module.global_variables[arg.handle].ty;
            let idx = arguments.len() as u32;

            arguments.push(FunctionArgument {
                name: arg.name.clone(),
                ty,
                binding: Some(arg.binding.clone()),
            });

            let pointer =
                expressions.append(Expression::GlobalVariable(arg.handle), Default::default());
            let value = expressions.append(Expression::FunctionArgument(idx), Default::default());

            body.push(Statement::Store { pointer, value }, Default::default());
        }

        body.extend_block(global_init_body);

        body.push(
            Statement::Call {
                function,
                arguments: Vec::new(),
                result: None,
            },
            Default::default(),
        );

        let mut span = 0;
        let mut members = Vec::new();
        let mut components = Vec::new();

        for arg in self.entry_args.iter() {
            if arg.storage != StorageQualifier::Output {
                continue;
            }

            let ty = self.module.global_variables[arg.handle].ty;

            members.push(StructMember {
                name: arg.name.clone(),
                ty,
                binding: Some(arg.binding.clone()),
                offset: span,
            });

            span += self.module.types[ty].inner.span(&self.module.constants);

            let pointer =
                expressions.append(Expression::GlobalVariable(arg.handle), Default::default());
            let len = expressions.len();
            let load = expressions.append(Expression::Load { pointer }, Default::default());
            body.push(
                Statement::Emit(expressions.range_from(len)),
                Default::default(),
            );
            components.push(load)
        }

        let (ty, value) = if !components.is_empty() {
            let ty = self.module.types.append(
                Type {
                    name: None,
                    inner: TypeInner::Struct {
                        top_level: false,
                        members,
                        span,
                    },
                },
                Default::default(),
            );

            let len = expressions.len();
            let res =
                expressions.append(Expression::Compose { ty, components }, Default::default());
            body.push(
                Statement::Emit(expressions.range_from(len)),
                Default::default(),
            );

            (Some(ty), Some(res))
        } else {
            (None, None)
        };

        body.push(Statement::Return { value }, Default::default());

        self.module.entry_points.push(EntryPoint {
            name: "main".to_string(),
            stage: self.meta.stage,
            early_depth_test: Some(crate::EarlyDepthTest { conservative: None })
                .filter(|_| self.meta.early_fragment_tests),
            workgroup_size: self.meta.workgroup_size,
            function: Function {
                arguments,
                expressions,
                body,
                result: ty.map(|ty| FunctionResult { ty, binding: None }),
                ..Default::default()
            },
        });
    }
}

fn is_double(ty: &TypeInner) -> bool {
    match *ty {
        TypeInner::ValuePointer { kind, width, .. }
        | TypeInner::Scalar { kind, width }
        | TypeInner::Vector { kind, width, .. } => kind == ScalarKind::Float && width == 8,
        TypeInner::Matrix { width, .. } => width == 8,
        _ => false,
    }
}
