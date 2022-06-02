use super::{
    ast::*,
    builtins::{inject_builtin, sampled_to_depth},
    context::{Context, ExprPos, StmtContext},
    error::{Error, ErrorKind},
    types::scalar_components,
    Parser, Result,
};
use crate::{
    front::glsl::types::type_power, proc::ensure_block_returns, AddressSpace, Arena, Block,
    Constant, ConstantInner, EntryPoint, Expression, FastHashMap, Function, FunctionArgument,
    FunctionResult, Handle, LocalVariable, ScalarKind, ScalarValue, Span, Statement, StructMember,
    Type, TypeInner,
};
use std::iter;

impl Parser {
    fn add_constant_value(
        &mut self,
        scalar_kind: ScalarKind,
        value: u64,
        meta: Span,
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
            meta,
        )
    }

    pub(crate) fn function_or_constructor_call(
        &mut self,
        ctx: &mut Context,
        stmt: &StmtContext,
        body: &mut Block,
        fc: FunctionCallKind,
        raw_args: &[Handle<HirExpr>],
        meta: Span,
    ) -> Result<Option<Handle<Expression>>> {
        let args: Vec<_> = raw_args
            .iter()
            .map(|e| ctx.lower_expect_inner(stmt, self, *e, ExprPos::Rhs, body))
            .collect::<Result<_>>()?;

        match fc {
            FunctionCallKind::TypeConstructor(ty) => {
                if args.len() == 1 {
                    self.constructor_single(ctx, body, ty, args[0], meta)
                        .map(Some)
                } else {
                    self.constructor_many(ctx, body, ty, args, meta).map(Some)
                }
            }
            FunctionCallKind::Function(name) => {
                self.function_call(ctx, stmt, body, name, args, raw_args, meta)
            }
        }
    }

    fn constructor_single(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        ty: Handle<Type>,
        (mut value, expr_meta): (Handle<Expression>, Span),
        meta: Span,
    ) -> Result<Handle<Expression>> {
        let expr_type = self.resolve_type(ctx, value, expr_meta)?;

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
                let c0 = self.add_constant_value(result_scalar_kind, 0u64, meta);
                let c1 = self.add_constant_value(result_scalar_kind, 1u64, meta);
                let mut reject = ctx.add_expression(Expression::Constant(c0), expr_meta, body);
                let mut accept = ctx.add_expression(Expression::Constant(c1), expr_meta, body);

                ctx.implicit_splat(self, &mut reject, meta, vector_size)?;
                ctx.implicit_splat(self, &mut accept, meta, vector_size)?;

                let h = ctx.add_expression(
                    Expression::Select {
                        accept,
                        reject,
                        condition: value,
                    },
                    expr_meta,
                    body,
                );

                return Ok(h);
            }
            _ => {}
        }

        Ok(match self.module.types[ty].inner {
            TypeInner::Vector { size, kind, width } if vector_size.is_none() => {
                ctx.forced_conversion(self, &mut value, expr_meta, kind, width)?;

                if let TypeInner::Scalar { .. } = *self.resolve_type(ctx, value, expr_meta)? {
                    ctx.add_expression(Expression::Splat { size, value }, meta, body)
                } else {
                    self.vector_constructor(
                        ctx,
                        body,
                        ty,
                        size,
                        kind,
                        width,
                        &[(value, expr_meta)],
                        meta,
                    )?
                }
            }
            TypeInner::Scalar { kind, width } => {
                let mut expr = value;
                if let TypeInner::Vector { .. } | TypeInner::Matrix { .. } =
                    *self.resolve_type(ctx, value, expr_meta)?
                {
                    expr = ctx.add_expression(
                        Expression::AccessIndex {
                            base: expr,
                            index: 0,
                        },
                        meta,
                        body,
                    );
                }

                if let TypeInner::Matrix { .. } = *self.resolve_type(ctx, value, expr_meta)? {
                    expr = ctx.add_expression(
                        Expression::AccessIndex {
                            base: expr,
                            index: 0,
                        },
                        meta,
                        body,
                    );
                }

                ctx.add_expression(
                    Expression::As {
                        kind,
                        expr,
                        convert: Some(width),
                    },
                    meta,
                    body,
                )
            }
            TypeInner::Vector { size, kind, width } => {
                if vector_size.map_or(true, |s| s != size) {
                    value = ctx.vector_resize(size, value, expr_meta, body);
                }

                ctx.add_expression(
                    Expression::As {
                        kind,
                        expr: value,
                        convert: Some(width),
                    },
                    meta,
                    body,
                )
            }
            TypeInner::Matrix {
                columns,
                rows,
                width,
            } => self.matrix_one_arg(
                ctx,
                body,
                ty,
                columns,
                rows,
                width,
                (value, expr_meta),
                meta,
            )?,
            TypeInner::Struct { ref members, .. } => {
                let scalar_components = members
                    .get(0)
                    .and_then(|member| scalar_components(&self.module.types[member.ty].inner));
                if let Some((kind, width)) = scalar_components {
                    ctx.implicit_conversion(self, &mut value, expr_meta, kind, width)?;
                }

                ctx.add_expression(
                    Expression::Compose {
                        ty,
                        components: vec![value],
                    },
                    meta,
                    body,
                )
            }

            TypeInner::Array { base, .. } => {
                let scalar_components = scalar_components(&self.module.types[base].inner);
                if let Some((kind, width)) = scalar_components {
                    ctx.implicit_conversion(self, &mut value, expr_meta, kind, width)?;
                }

                ctx.add_expression(
                    Expression::Compose {
                        ty,
                        components: vec![value],
                    },
                    meta,
                    body,
                )
            }
            _ => {
                self.errors.push(Error {
                    kind: ErrorKind::SemanticError("Bad type constructor".into()),
                    meta,
                });

                value
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn matrix_one_arg(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        ty: Handle<Type>,
        columns: crate::VectorSize,
        rows: crate::VectorSize,
        width: crate::Bytes,
        (mut value, expr_meta): (Handle<Expression>, Span),
        meta: Span,
    ) -> Result<Handle<Expression>> {
        let mut components = Vec::with_capacity(columns as usize);
        // TODO: casts
        // `Expression::As` doesn't support matrix width
        // casts so we need to do some extra work for casts

        ctx.forced_conversion(self, &mut value, expr_meta, ScalarKind::Float, width)?;
        match *self.resolve_type(ctx, value, expr_meta)? {
            TypeInner::Scalar { .. } => {
                // If a matrix is constructed with a single scalar value, then that
                // value is used to initialize all the values along the diagonal of
                // the matrix; the rest are given zeros.
                let vector_ty = self.module.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Vector {
                            size: rows,
                            kind: ScalarKind::Float,
                            width,
                        },
                    },
                    meta,
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
                    meta,
                );
                let zero = ctx.add_expression(Expression::Constant(zero_constant), meta, body);

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
            }
            TypeInner::Matrix {
                rows: ori_rows,
                columns: ori_cols,
                ..
            } => {
                // If a matrix is constructed from a matrix, then each component
                // (column i, row j) in the result that has a corresponding component
                // (column i, row j) in the argument will be initialized from there. All
                // other components will be initialized to the identity matrix.

                let zero_constant = self.module.constants.fetch_or_append(
                    Constant {
                        name: None,
                        specialization: None,
                        inner: ConstantInner::Scalar {
                            width,
                            value: ScalarValue::Float(0.0),
                        },
                    },
                    meta,
                );
                let zero = ctx.add_expression(Expression::Constant(zero_constant), meta, body);
                let one_constant = self.module.constants.fetch_or_append(
                    Constant {
                        name: None,
                        specialization: None,
                        inner: ConstantInner::Scalar {
                            width,
                            value: ScalarValue::Float(1.0),
                        },
                    },
                    meta,
                );
                let one = ctx.add_expression(Expression::Constant(one_constant), meta, body);
                let vector_ty = self.module.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Vector {
                            size: rows,
                            kind: ScalarKind::Float,
                            width,
                        },
                    },
                    meta,
                );

                for i in 0..columns as u32 {
                    if i < ori_cols as u32 {
                        use std::cmp::Ordering;

                        let vector = ctx.add_expression(
                            Expression::AccessIndex {
                                base: value,
                                index: i,
                            },
                            meta,
                            body,
                        );

                        components.push(match ori_rows.cmp(&rows) {
                            Ordering::Less => {
                                let components = (0..rows as u32)
                                    .into_iter()
                                    .map(|r| {
                                        if r < ori_rows as u32 {
                                            ctx.add_expression(
                                                Expression::AccessIndex {
                                                    base: vector,
                                                    index: r,
                                                },
                                                meta,
                                                body,
                                            )
                                        } else if r == i {
                                            one
                                        } else {
                                            zero
                                        }
                                    })
                                    .collect();

                                ctx.add_expression(
                                    Expression::Compose {
                                        ty: vector_ty,
                                        components,
                                    },
                                    meta,
                                    body,
                                )
                            }
                            Ordering::Equal => vector,
                            Ordering::Greater => ctx.vector_resize(rows, vector, meta, body),
                        })
                    } else {
                        let vec_constant = self.module.constants.fetch_or_append(
                            Constant {
                                name: None,
                                specialization: None,
                                inner: ConstantInner::Composite {
                                    ty: vector_ty,
                                    components: (0..rows as u32)
                                        .into_iter()
                                        .map(|r| match r == i {
                                            true => one_constant,
                                            false => zero_constant,
                                        })
                                        .collect(),
                                },
                            },
                            meta,
                        );
                        let vec =
                            ctx.add_expression(Expression::Constant(vec_constant), meta, body);

                        components.push(vec)
                    }
                }
            }
            _ => {
                components = iter::repeat(value).take(columns as usize).collect();
            }
        }

        Ok(ctx.add_expression(Expression::Compose { ty, components }, meta, body))
    }

    #[allow(clippy::too_many_arguments)]
    fn vector_constructor(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        ty: Handle<Type>,
        size: crate::VectorSize,
        kind: ScalarKind,
        width: crate::Bytes,
        args: &[(Handle<Expression>, Span)],
        meta: Span,
    ) -> Result<Handle<Expression>> {
        let mut components = Vec::with_capacity(size as usize);

        for (mut arg, expr_meta) in args.iter().copied() {
            ctx.forced_conversion(self, &mut arg, expr_meta, kind, width)?;

            if components.len() >= size as usize {
                break;
            }

            match *self.resolve_type(ctx, arg, expr_meta)? {
                TypeInner::Scalar { .. } => components.push(arg),
                TypeInner::Matrix { rows, columns, .. } => {
                    components.reserve(rows as usize * columns as usize);
                    for c in 0..(columns as u32) {
                        let base = ctx.add_expression(
                            Expression::AccessIndex {
                                base: arg,
                                index: c,
                            },
                            expr_meta,
                            body,
                        );
                        for r in 0..(rows as u32) {
                            components.push(ctx.add_expression(
                                Expression::AccessIndex { base, index: r },
                                expr_meta,
                                body,
                            ))
                        }
                    }
                }
                TypeInner::Vector { size: ori_size, .. } => {
                    components.reserve(ori_size as usize);
                    for index in 0..(ori_size as u32) {
                        components.push(ctx.add_expression(
                            Expression::AccessIndex { base: arg, index },
                            expr_meta,
                            body,
                        ))
                    }
                }
                _ => components.push(arg),
            }
        }

        components.truncate(size as usize);

        Ok(ctx.add_expression(Expression::Compose { ty, components }, meta, body))
    }

    fn constructor_many(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        ty: Handle<Type>,
        args: Vec<(Handle<Expression>, Span)>,
        meta: Span,
    ) -> Result<Handle<Expression>> {
        let mut components = Vec::with_capacity(args.len());

        match self.module.types[ty].inner {
            TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                let mut flattened = Vec::with_capacity(columns as usize * rows as usize);

                for (mut arg, meta) in args.iter().copied() {
                    ctx.forced_conversion(self, &mut arg, meta, ScalarKind::Float, width)?;

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

                let ty = self.module.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Vector {
                            size: rows,
                            kind: ScalarKind::Float,
                            width,
                        },
                    },
                    meta,
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
            TypeInner::Vector { size, kind, width } => {
                return self.vector_constructor(ctx, body, ty, size, kind, width, &args, meta)
            }
            TypeInner::Array { base, .. } => {
                for (mut arg, meta) in args.iter().copied() {
                    let scalar_components = scalar_components(&self.module.types[base].inner);
                    if let Some((kind, width)) = scalar_components {
                        ctx.implicit_conversion(self, &mut arg, meta, kind, width)?;
                    }

                    components.push(arg)
                }
            }
            TypeInner::Struct { ref members, .. } => {
                for ((mut arg, meta), member) in args.iter().copied().zip(members.iter()) {
                    let scalar_components = scalar_components(&self.module.types[member.ty].inner);
                    if let Some((kind, width)) = scalar_components {
                        ctx.implicit_conversion(self, &mut arg, meta, kind, width)?;
                    }

                    components.push(arg)
                }
            }
            _ => {
                return Err(Error {
                    kind: ErrorKind::SemanticError("Constructor: Too many arguments".into()),
                    meta,
                })
            }
        }

        Ok(ctx.add_expression(Expression::Compose { ty, components }, meta, body))
    }

    #[allow(clippy::too_many_arguments)]
    fn function_call(
        &mut self,
        ctx: &mut Context,
        stmt: &StmtContext,
        body: &mut Block,
        name: String,
        args: Vec<(Handle<Expression>, Span)>,
        raw_args: &[Handle<HirExpr>],
        meta: Span,
    ) -> Result<Option<Handle<Expression>>> {
        // Grow the typifier to be able to index it later without needing
        // to hold the context mutably
        for &(expr, span) in args.iter() {
            self.typifier_grow(ctx, expr, span)?;
        }

        // Check if the passed arguments require any special variations
        let mut variations = builtin_required_variations(
            args.iter()
                .map(|&(expr, _)| ctx.typifier.get(expr, &self.module.types)),
        );

        // Initiate the declaration if it wasn't previously initialized and inject builtins
        let declaration = self.lookup_function.entry(name.clone()).or_insert_with(|| {
            variations |= BuiltinVariations::STANDARD;
            Default::default()
        });
        inject_builtin(declaration, &mut self.module, &name, variations);

        // Borrow again but without mutability, at this point a declaration is guaranteed
        let declaration = self.lookup_function.get(&name).unwrap();

        // Possibly contains the overload to be used in the call
        let mut maybe_overload = None;
        // The conversions needed for the best analyzed overload, this is initialized all to
        // `NONE` to make sure that conversions always pass the first time without ambiguity
        let mut old_conversions = vec![Conversion::None; args.len()];
        // Tracks whether the comparison between overloads lead to an ambiguity
        let mut ambiguous = false;

        // Iterate over all the available overloads to select either an exact match or a
        // overload which has suitable implicit conversions
        'outer: for overload in declaration.overloads.iter() {
            // If the overload and the function call don't have the same number of arguments
            // continue to the next overload
            if args.len() != overload.parameters.len() {
                continue;
            }

            // Stores whether the current overload matches exactly the function call
            let mut exact = true;
            // State of the selection
            // If None we still don't know what is the best overload
            // If Some(true) the new overload is better
            // If Some(false) the old overload is better
            let mut superior = None;
            // Store the conversions for the current overload so that later they can replace the
            // conversions used for querying the best overload
            let mut new_conversions = vec![Conversion::None; args.len()];

            // Loop trough the overload parameters and check if the current overload is better
            // compared to the previous best overload.
            for (i, overload_parameter) in overload.parameters.iter().enumerate() {
                let call_argument = &args[i];
                let parameter_info = &overload.parameters_info[i];

                // If the image is used in the overload as a depth texture convert it
                // before comparing, otherwise exact matches wouldn't be reported
                if parameter_info.depth {
                    sampled_to_depth(
                        &mut self.module,
                        ctx,
                        call_argument.0,
                        call_argument.1,
                        &mut self.errors,
                    );
                    self.invalidate_expression(ctx, call_argument.0, call_argument.1)?
                }

                let overload_param_ty = &self.module.types[*overload_parameter].inner;
                let call_arg_ty = self.resolve_type(ctx, call_argument.0, call_argument.1)?;

                log::trace!(
                    "Testing parameter {}\n\tOverload = {:?}\n\tCall = {:?}",
                    i,
                    overload_param_ty,
                    call_arg_ty
                );

                // Storage images cannot be directly compared since while the access is part of the
                // type in naga's IR, in glsl they are a qualifier and don't enter in the match as
                // long as the access needed is satisfied.
                if let (
                    &TypeInner::Image {
                        class:
                            crate::ImageClass::Storage {
                                format: overload_format,
                                access: overload_access,
                            },
                        dim: overload_dim,
                        arrayed: overload_arrayed,
                    },
                    &TypeInner::Image {
                        class:
                            crate::ImageClass::Storage {
                                format: call_format,
                                access: call_access,
                            },
                        dim: call_dim,
                        arrayed: call_arrayed,
                    },
                ) = (overload_param_ty, call_arg_ty)
                {
                    // Images size must match otherwise the overload isn't what we want
                    let good_size = call_dim == overload_dim && call_arrayed == overload_arrayed;
                    // Glsl requires the formats to strictly match unless you are builtin
                    // function overload and have not been replaced, in which case we only
                    // check that the format scalar kind matches
                    let good_format = overload_format == call_format
                        || (overload.internal
                            && ScalarKind::from(overload_format) == ScalarKind::from(call_format));
                    if !(good_size && good_format) {
                        continue 'outer;
                    }

                    // While storage access mismatch is an error it isn't one that causes
                    // the overload matching to fail so we defer the error and consider
                    // that the images match exactly
                    if !call_access.contains(overload_access) {
                        self.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                format!(
                                    "'{}': image needs {:?} access but only {:?} was provided",
                                    name, overload_access, call_access
                                )
                                .into(),
                            ),
                            meta,
                        });
                    }

                    // The images satisfy the conditions to be considered as an exact match
                    new_conversions[i] = Conversion::Exact;
                    continue;
                } else if overload_param_ty == call_arg_ty {
                    // If the types match there's no need to check for conversions so continue
                    new_conversions[i] = Conversion::Exact;
                    continue;
                }

                // If the argument is to be passed as a pointer (i.e. either `out` or
                // `inout` where used as qualifiers) no conversion shall be performed
                if parameter_info.qualifier.is_lhs() {
                    continue 'outer;
                }

                // Try to get the type of conversion needed otherwise this overload can't be used
                // since no conversion makes it possible so skip it
                let conversion = match conversion(overload_param_ty, call_arg_ty) {
                    Some(info) => info,
                    None => continue 'outer,
                };

                // At this point a conversion will be needed so the overload no longer
                // exactly matches the call arguments
                exact = false;

                // Compare the conversions needed for this overload parameter to that of the
                // last overload analyzed respective parameter, the value is:
                // - `true` when the new overload argument has a better conversion
                // - `false` when the old overload argument has a better conversion
                let best_arg = match (conversion, old_conversions[i]) {
                    // An exact match is always better, we don't need to check this for the
                    // current overload since it was checked earlier
                    (_, Conversion::Exact) => false,
                    // No overload was yet analyzed so this one is the best yet
                    (_, Conversion::None) => true,
                    // A conversion from a float to a double is the best possible conversion
                    (Conversion::FloatToDouble, _) => true,
                    (_, Conversion::FloatToDouble) => false,
                    // A conversion from a float to an integer is preferred than one
                    // from double to an integer
                    (Conversion::IntToFloat, Conversion::IntToDouble) => true,
                    (Conversion::IntToDouble, Conversion::IntToFloat) => false,
                    // This case handles things like no conversion and exact which were already
                    // treated and other cases which no conversion is better than the other
                    _ => continue,
                };

                // Check if the best parameter corresponds to the current selected overload
                // to pass to the next comparison, if this isn't true mark it as ambiguous
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

            // The overload matches exactly the function call so there's no ambiguity (since
            // repeated overload aren't allowed) and the current overload is selected, no
            // further querying is needed.
            if exact {
                maybe_overload = Some(overload);
                ambiguous = false;
                break;
            }

            match superior {
                // New overload is better keep it
                Some(true) => {
                    maybe_overload = Some(overload);
                    // Replace the conversions
                    old_conversions = new_conversions;
                }
                // Old overload is better do nothing
                Some(false) => {}
                // No overload was better than the other this can be caused
                // when all conversions are ambiguous in which the overloads themselves are
                // ambiguous.
                None => {
                    ambiguous = true;
                    // Assign the new overload, this helps ensures that in this case of
                    // ambiguity the parsing won't end immediately and allow for further
                    // collection of errors.
                    maybe_overload = Some(overload);
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

        let overload = maybe_overload.ok_or_else(|| Error {
            kind: ErrorKind::SemanticError(format!("Unknown function '{}'", name).into()),
            meta,
        })?;

        let parameters_info = overload.parameters_info.clone();
        let parameters = overload.parameters.clone();
        let is_void = overload.void;
        let kind = overload.kind;

        let mut arguments = Vec::with_capacity(args.len());
        let mut proxy_writes = Vec::new();
        // Iterate trough the function call arguments applying transformations as needed
        for (parameter_info, (expr, parameter)) in parameters_info
            .iter()
            .zip(raw_args.iter().zip(parameters.iter()))
        {
            let (mut handle, meta) =
                ctx.lower_expect_inner(stmt, self, *expr, parameter_info.qualifier.as_pos(), body)?;

            if parameter_info.qualifier.is_lhs() {
                let (ty, value) = match *self.resolve_type(ctx, handle, meta)? {
                    // If the argument is to be passed as a pointer but the type of the
                    // expression returns a vector it must mean that it was for example
                    // swizzled and it must be spilled into a local before calling
                    TypeInner::Vector { size, kind, width } => (
                        self.module.types.insert(
                            Type {
                                name: None,
                                inner: TypeInner::Vector { size, kind, width },
                            },
                            Span::default(),
                        ),
                        handle,
                    ),
                    // If the argument is a pointer whose address space isn't `Function`, an
                    // indirection through a local variable is needed to align the address
                    // spaces of the call argument and the overload parameter.
                    TypeInner::Pointer { base, space } if space != AddressSpace::Function => (
                        base,
                        ctx.add_expression(
                            Expression::Load { pointer: handle },
                            Span::default(),
                            body,
                        ),
                    ),
                    TypeInner::ValuePointer {
                        size,
                        kind,
                        width,
                        space,
                    } if space != AddressSpace::Function => {
                        let inner = match size {
                            Some(size) => TypeInner::Vector { size, kind, width },
                            None => TypeInner::Scalar { kind, width },
                        };

                        (
                            self.module
                                .types
                                .insert(Type { name: None, inner }, Span::default()),
                            ctx.add_expression(
                                Expression::Load { pointer: handle },
                                Span::default(),
                                body,
                            ),
                        )
                    }
                    _ => {
                        arguments.push(handle);
                        continue;
                    }
                };

                let temp_var = ctx.locals.append(
                    LocalVariable {
                        name: None,
                        ty,
                        init: None,
                    },
                    Span::default(),
                );
                let temp_expr =
                    ctx.add_expression(Expression::LocalVariable(temp_var), Span::default(), body);

                body.push(
                    Statement::Store {
                        pointer: temp_expr,
                        value,
                    },
                    Span::default(),
                );

                arguments.push(temp_expr);
                // Register the temporary local to be written back to it's original
                // place after the function call
                if let Expression::Swizzle {
                    size,
                    mut vector,
                    pattern,
                } = ctx.expressions[value]
                {
                    if let Expression::Load { pointer } = ctx.expressions[vector] {
                        vector = pointer;
                    }

                    for (i, component) in pattern.iter().take(size as usize).enumerate() {
                        let original = ctx.add_expression(
                            Expression::AccessIndex {
                                base: vector,
                                index: *component as u32,
                            },
                            Span::default(),
                            body,
                        );

                        let temp = ctx.add_expression(
                            Expression::AccessIndex {
                                base: temp_expr,
                                index: i as u32,
                            },
                            Span::default(),
                            body,
                        );

                        proxy_writes.push((original, temp));
                    }
                } else {
                    proxy_writes.push((handle, temp_expr));
                }
                continue;
            }

            // Apply implicit conversions as needed
            let scalar_components = scalar_components(&self.module.types[*parameter].inner);
            if let Some((kind, width)) = scalar_components {
                ctx.implicit_conversion(self, &mut handle, meta, kind, width)?;
            }

            arguments.push(handle)
        }

        match kind {
            FunctionKind::Call(function) => {
                ctx.emit_end(body);

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
                    meta,
                );

                ctx.emit_start();

                // Write back all the variables that were scheduled to their original place
                for (original, pointer) in proxy_writes {
                    let value = ctx.add_expression(Expression::Load { pointer }, meta, body);

                    ctx.emit_restart(body);

                    body.push(
                        Statement::Store {
                            pointer: original,
                            value,
                        },
                        meta,
                    );
                }

                Ok(result)
            }
            FunctionKind::Macro(builtin) => {
                builtin.call(self, ctx, body, arguments.as_mut_slice(), meta)
            }
        }
    }

    pub(crate) fn add_function(
        &mut self,
        ctx: Context,
        name: String,
        result: Option<FunctionResult>,
        mut body: Block,
        meta: Span,
    ) {
        ensure_block_returns(&mut body);

        let void = result.is_none();

        let &mut Parser {
            ref mut lookup_function,
            ref mut module,
            ..
        } = self;

        // Check if the passed arguments require any special variations
        let mut variations =
            builtin_required_variations(ctx.parameters.iter().map(|&arg| &module.types[arg].inner));

        // Initiate the declaration if it wasn't previously initialized and inject builtins
        let declaration = lookup_function.entry(name.clone()).or_insert_with(|| {
            variations |= BuiltinVariations::STANDARD;
            Default::default()
        });
        inject_builtin(declaration, module, &name, variations);

        let Context {
            expressions,
            locals,
            arguments,
            parameters,
            parameters_info,
            ..
        } = ctx;

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
                    let handle = module.functions.append(function, meta);
                    decl.kind = FunctionKind::Call(handle)
                }
            }
            return;
        }

        let handle = module.functions.append(function, meta);
        declaration.overloads.push(Overload {
            parameters,
            parameters_info,
            kind: FunctionKind::Call(handle),
            defined: true,
            internal: false,
            void,
        });
    }

    pub(crate) fn add_prototype(
        &mut self,
        ctx: Context,
        name: String,
        result: Option<FunctionResult>,
        meta: Span,
    ) {
        let void = result.is_none();

        let &mut Parser {
            ref mut lookup_function,
            ref mut module,
            ..
        } = self;

        // Check if the passed arguments require any special variations
        let mut variations =
            builtin_required_variations(ctx.parameters.iter().map(|&arg| &module.types[arg].inner));

        // Initiate the declaration if it wasn't previously initialized and inject builtins
        let declaration = lookup_function.entry(name.clone()).or_insert_with(|| {
            variations |= BuiltinVariations::STANDARD;
            Default::default()
        });
        inject_builtin(declaration, module, &name, variations);

        let Context {
            arguments,
            parameters,
            parameters_info,
            ..
        } = ctx;

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

        let handle = module.functions.append(function, meta);
        declaration.overloads.push(Overload {
            parameters,
            parameters_info,
            kind: FunctionKind::Call(handle),
            defined: false,
            internal: false,
            void,
        });
    }

    /// Helper function for building the input/output interface of the entry point
    ///
    /// Calls `f` with the data of the entry point argument, flattening composite types
    /// recursively
    ///
    /// The passed arguments to the callback are:
    /// - The name
    /// - The pointer expression to the global storage
    /// - The handle to the type of the entry point argument
    /// - The binding of the entry point argument
    /// - The expression arena
    fn arg_type_walker(
        &self,
        name: Option<String>,
        binding: crate::Binding,
        pointer: Handle<Expression>,
        ty: Handle<Type>,
        expressions: &mut Arena<Expression>,
        f: &mut impl FnMut(
            Option<String>,
            Handle<Expression>,
            Handle<Type>,
            crate::Binding,
            &mut Arena<Expression>,
        ),
    ) {
        match self.module.types[ty].inner {
            TypeInner::Array {
                base,
                size: crate::ArraySize::Constant(constant),
                ..
            } => {
                let mut location = match binding {
                    crate::Binding::Location { location, .. } => location,
                    _ => return,
                };

                // TODO: Better error reporting
                // right now we just don't walk the array if the size isn't known at
                // compile time and let validation catch it
                let size = match self.module.constants[constant].to_array_length() {
                    Some(val) => val,
                    None => return f(name, pointer, ty, binding, expressions),
                };

                let interpolation =
                    self.module.types[base]
                        .inner
                        .scalar_kind()
                        .map(|kind| match kind {
                            ScalarKind::Float => crate::Interpolation::Perspective,
                            _ => crate::Interpolation::Flat,
                        });

                for index in 0..size {
                    let member_pointer = expressions.append(
                        Expression::AccessIndex {
                            base: pointer,
                            index,
                        },
                        crate::Span::default(),
                    );

                    let binding = crate::Binding::Location {
                        location,
                        interpolation,
                        sampling: None,
                    };
                    location += 1;

                    self.arg_type_walker(
                        name.clone(),
                        binding,
                        member_pointer,
                        base,
                        expressions,
                        f,
                    )
                }
            }
            TypeInner::Struct { ref members, .. } => {
                let mut location = match binding {
                    crate::Binding::Location { location, .. } => location,
                    _ => return,
                };

                for (i, member) in members.iter().enumerate() {
                    let member_pointer = expressions.append(
                        Expression::AccessIndex {
                            base: pointer,
                            index: i as u32,
                        },
                        crate::Span::default(),
                    );

                    let binding = match member.binding.clone() {
                        Some(binding) => binding,
                        None => {
                            let interpolation = self.module.types[member.ty]
                                .inner
                                .scalar_kind()
                                .map(|kind| match kind {
                                    ScalarKind::Float => crate::Interpolation::Perspective,
                                    _ => crate::Interpolation::Flat,
                                });
                            let binding = crate::Binding::Location {
                                location,
                                interpolation,
                                sampling: None,
                            };
                            location += 1;
                            binding
                        }
                    };

                    self.arg_type_walker(
                        member.name.clone(),
                        binding,
                        member_pointer,
                        member.ty,
                        expressions,
                        f,
                    )
                }
            }
            _ => f(name, pointer, ty, binding, expressions),
        }
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

            let pointer =
                expressions.append(Expression::GlobalVariable(arg.handle), Default::default());

            self.arg_type_walker(
                arg.name.clone(),
                arg.binding.clone(),
                pointer,
                self.module.global_variables[arg.handle].ty,
                &mut expressions,
                &mut |name, pointer, ty, binding, expressions| {
                    let idx = arguments.len() as u32;

                    arguments.push(FunctionArgument {
                        name,
                        ty,
                        binding: Some(binding),
                    });

                    let value =
                        expressions.append(Expression::FunctionArgument(idx), Default::default());
                    body.push(Statement::Store { pointer, value }, Default::default());
                },
            )
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

            let pointer =
                expressions.append(Expression::GlobalVariable(arg.handle), Default::default());

            self.arg_type_walker(
                arg.name.clone(),
                arg.binding.clone(),
                pointer,
                self.module.global_variables[arg.handle].ty,
                &mut expressions,
                &mut |name, pointer, ty, binding, expressions| {
                    members.push(StructMember {
                        name,
                        ty,
                        binding: Some(binding),
                        offset: span,
                    });

                    span += self.module.types[ty].inner.size(&self.module.constants);

                    let len = expressions.len();
                    let load = expressions.append(Expression::Load { pointer }, Default::default());
                    body.push(
                        Statement::Emit(expressions.range_from(len)),
                        Default::default(),
                    );
                    components.push(load)
                },
            )
        }

        let (ty, value) = if !components.is_empty() {
            let ty = self.module.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Struct { members, span },
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

/// Helper enum containing the type of conversion need for a call
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Conversion {
    /// No conversion needed
    Exact,
    /// Float to double conversion needed
    FloatToDouble,
    /// Int or uint to float conversion needed
    IntToFloat,
    /// Int or uint to double conversion needed
    IntToDouble,
    /// Other type of conversion needed
    Other,
    /// No conversion was yet registered
    None,
}

/// Helper function, returns the type of conversion from `source` to `target`, if a
/// conversion is not possible returns None.
fn conversion(target: &TypeInner, source: &TypeInner) -> Option<Conversion> {
    use ScalarKind::*;

    // Gather the `ScalarKind` and scalar width from both the target and the source
    let (target_kind, target_width, source_kind, source_width) = match (target, source) {
        // Conversions between scalars are allowed
        (
            &TypeInner::Scalar {
                kind: tgt_kind,
                width: tgt_width,
            },
            &TypeInner::Scalar {
                kind: src_kind,
                width: src_width,
            },
        ) => (tgt_kind, tgt_width, src_kind, src_width),
        // Conversions between vectors of the same size are allowed
        (
            &TypeInner::Vector {
                kind: tgt_kind,
                size: tgt_size,
                width: tgt_width,
            },
            &TypeInner::Vector {
                kind: src_kind,
                size: src_size,
                width: src_width,
            },
        ) if tgt_size == src_size => (tgt_kind, tgt_width, src_kind, src_width),
        // Conversions between matrices of the same size are allowed
        (
            &TypeInner::Matrix {
                rows: tgt_rows,
                columns: tgt_cols,
                width: tgt_width,
            },
            &TypeInner::Matrix {
                rows: src_rows,
                columns: src_cols,
                width: src_width,
            },
        ) if tgt_cols == src_cols && tgt_rows == src_rows => (Float, tgt_width, Float, src_width),
        _ => return None,
    };

    // Check if source can be converted into target, if this is the case then the type
    // power of target must be higher than that of source
    let target_power = type_power(target_kind, target_width);
    let source_power = type_power(source_kind, source_width);
    if target_power < source_power {
        return None;
    }

    Some(
        match ((target_kind, target_width), (source_kind, source_width)) {
            // A conversion from a float to a double is special
            ((Float, 8), (Float, 4)) => Conversion::FloatToDouble,
            // A conversion from an integer to a float is special
            ((Float, 4), (Sint | Uint, _)) => Conversion::IntToFloat,
            // A conversion from an integer to a double is special
            ((Float, 8), (Sint | Uint, _)) => Conversion::IntToDouble,
            _ => Conversion::Other,
        },
    )
}

/// Helper method returning all the non standard builtin variations needed
/// to process the function call with the passed arguments
fn builtin_required_variations<'a>(args: impl Iterator<Item = &'a TypeInner>) -> BuiltinVariations {
    let mut variations = BuiltinVariations::empty();

    for ty in args {
        match *ty {
            TypeInner::ValuePointer { kind, width, .. }
            | TypeInner::Scalar { kind, width }
            | TypeInner::Vector { kind, width, .. } => {
                if kind == ScalarKind::Float && width == 8 {
                    variations |= BuiltinVariations::DOUBLE
                }
            }
            TypeInner::Matrix { width, .. } => {
                if width == 8 {
                    variations |= BuiltinVariations::DOUBLE
                }
            }
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                if dim == crate::ImageDimension::Cube && arrayed {
                    variations |= BuiltinVariations::CUBE_TEXTURES_ARRAY
                }

                if dim == crate::ImageDimension::D2 && arrayed && class.is_multisampled() {
                    variations |= BuiltinVariations::D2_MULTI_TEXTURES_ARRAY
                }
            }
            _ => {}
        }
    }

    variations
}
