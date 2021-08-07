use crate::{
    front::{
        glsl::{
            ast::{
                GlobalLookup, GlobalLookupKind, HirExpr, HirExprKind, ParameterInfo,
                ParameterQualifier, VariableReference,
            },
            error::{Error, ErrorKind},
            types::{scalar_components, type_power},
            Parser, Result, SourceMetadata,
        },
        Emitter, Typifier,
    },
    Arena, BinaryOperator, Block, Constant, Expression, FastHashMap, FunctionArgument, Handle,
    LocalVariable, RelationalFunction, ScalarKind, ScalarValue, Statement, StorageClass, Type,
    TypeInner, VectorSize,
};
use std::{convert::TryFrom, ops::Index};

#[derive(Debug)]
pub struct Context {
    pub expressions: Arena<Expression>,
    pub locals: Arena<LocalVariable>,
    pub arguments: Vec<FunctionArgument>,

    pub parameters: Vec<Handle<Type>>,
    pub parameters_info: Vec<ParameterInfo>,

    //TODO: Find less allocation heavy representation
    pub scopes: Vec<FastHashMap<String, VariableReference>>,
    pub lookup_global_var_exps: FastHashMap<String, VariableReference>,
    pub samplers: FastHashMap<Handle<Expression>, Handle<Expression>>,

    pub hir_exprs: Arena<HirExpr>,
    pub typifier: Typifier,
    emitter: Emitter,
}

impl Context {
    pub fn new(parser: &Parser, body: &mut Block) -> Self {
        let mut this = Context {
            expressions: Arena::new(),
            locals: Arena::new(),
            arguments: Vec::new(),

            parameters: Vec::new(),
            parameters_info: Vec::new(),

            scopes: vec![FastHashMap::default()],
            lookup_global_var_exps: FastHashMap::with_capacity_and_hasher(
                parser.global_variables.len(),
                Default::default(),
            ),
            samplers: FastHashMap::default(),

            hir_exprs: Arena::default(),
            typifier: Typifier::new(),
            emitter: Emitter::default(),
        };

        this.emit_start();

        for &(ref name, lookup) in parser.global_variables.iter() {
            this.add_global(parser, name, lookup, body)
        }

        this
    }

    pub fn add_global(
        &mut self,
        parser: &Parser,
        name: &str,
        GlobalLookup {
            kind,
            entry_arg,
            mutable,
        }: GlobalLookup,
        body: &mut Block,
    ) {
        self.emit_flush(body);
        let (expr, load) = match kind {
            GlobalLookupKind::Variable(v) => {
                let res = (
                    self.expressions.append(Expression::GlobalVariable(v)),
                    parser.module.global_variables[v].class != StorageClass::Handle,
                );
                self.emit_start();

                res
            }
            GlobalLookupKind::BlockSelect(handle, index) => {
                let base = self.expressions.append(Expression::GlobalVariable(handle));
                self.emit_start();
                let expr = self
                    .expressions
                    .append(Expression::AccessIndex { base, index });

                (expr, {
                    let ty = parser.module.global_variables[handle].ty;

                    match parser.module.types[ty].inner {
                        TypeInner::Struct { ref members, .. } => {
                            if let TypeInner::Array {
                                size: crate::ArraySize::Dynamic,
                                ..
                            } = parser.module.types[members[index as usize].ty].inner
                            {
                                false
                            } else {
                                true
                            }
                        }
                        _ => true,
                    }
                })
            }
            GlobalLookupKind::Constant(v) => {
                let res = (self.expressions.append(Expression::Constant(v)), false);
                self.emit_start();
                res
            }
        };

        let var = VariableReference {
            expr,
            load,
            mutable,
            entry_arg,
        };

        self.lookup_global_var_exps.insert(name.into(), var);
    }

    pub fn emit_start(&mut self) {
        self.emitter.start(&self.expressions)
    }

    pub fn emit_flush(&mut self, body: &mut Block) {
        body.extend(self.emitter.finish(&self.expressions))
    }

    pub fn add_expression(&mut self, expr: Expression, body: &mut Block) -> Handle<Expression> {
        if expr.needs_pre_emit() {
            self.emit_flush(body);
            let handle = self.expressions.append(expr);
            self.emit_start();
            handle
        } else {
            self.expressions.append(expr)
        }
    }

    pub fn lookup_local_var(&self, name: &str) -> Option<VariableReference> {
        for scope in self.scopes.iter().rev() {
            if let Some(var) = scope.get(name) {
                return Some(var.clone());
            }
        }
        None
    }

    pub fn lookup_global_var(&mut self, name: &str) -> Option<VariableReference> {
        self.lookup_global_var_exps.get(name).cloned()
    }

    #[cfg(feature = "glsl-validate")]
    pub fn lookup_local_var_current_scope(&self, name: &str) -> Option<VariableReference> {
        if let Some(current) = self.scopes.last() {
            current.get(name).cloned()
        } else {
            None
        }
    }

    /// Add variable to current scope
    pub fn add_local_var(&mut self, name: String, expr: Handle<Expression>, mutable: bool) {
        if let Some(current) = self.scopes.last_mut() {
            (*current).insert(
                name,
                VariableReference {
                    expr,
                    load: true,
                    mutable,
                    entry_arg: None,
                },
            );
        }
    }

    /// Add function argument to current scope
    pub fn add_function_arg(
        &mut self,
        parser: &mut Parser,
        body: &mut Block,
        name: Option<String>,
        ty: Handle<Type>,
        qualifier: ParameterQualifier,
    ) {
        let index = self.arguments.len();
        let mut arg = FunctionArgument {
            name: name.clone(),
            ty,
            binding: None,
        };
        self.parameters.push(ty);

        let opaque = match parser.module.types[ty].inner {
            TypeInner::Image { .. } | TypeInner::Sampler { .. } => true,
            _ => false,
        };

        if qualifier.is_lhs() {
            arg.ty = parser.module.types.fetch_or_append(Type {
                name: None,
                inner: TypeInner::Pointer {
                    base: arg.ty,
                    class: StorageClass::Function,
                },
            })
        }

        self.arguments.push(arg);

        self.parameters_info.push(ParameterInfo {
            qualifier,
            depth: false,
        });

        if let Some(name) = name {
            let expr = self.add_expression(Expression::FunctionArgument(index as u32), body);
            let mutable = qualifier != ParameterQualifier::Const && !opaque;
            let load = qualifier.is_lhs();

            if mutable && !load {
                let handle = self.locals.append(LocalVariable {
                    name: Some(name.clone()),
                    ty,
                    init: None,
                });
                let local_expr = self.add_expression(Expression::LocalVariable(handle), body);

                self.emit_flush(body);
                self.emit_start();

                body.push(Statement::Store {
                    pointer: local_expr,
                    value: expr,
                });

                if let Some(current) = self.scopes.last_mut() {
                    (*current).insert(
                        name,
                        VariableReference {
                            expr: local_expr,
                            load: true,
                            mutable,
                            entry_arg: None,
                        },
                    );
                }
            } else if let Some(current) = self.scopes.last_mut() {
                (*current).insert(
                    name,
                    VariableReference {
                        expr,
                        load,
                        mutable,
                        entry_arg: None,
                    },
                );
            }
        }
    }

    /// Add new empty scope
    pub fn push_scope(&mut self) {
        self.scopes.push(FastHashMap::default());
    }

    pub fn remove_current_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn lower_expect(
        &mut self,
        parser: &mut Parser,
        expr: Handle<HirExpr>,
        lhs: bool,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, SourceMetadata)> {
        let (maybe_expr, meta) = self.lower(parser, expr, lhs, body)?;

        let expr = match maybe_expr {
            Some(e) => e,
            None => {
                return Err(Error {
                    kind: ErrorKind::SemanticError("Expression returns void".into()),
                    meta,
                })
            }
        };

        Ok((expr, meta))
    }

    pub fn lower(
        &mut self,
        parser: &mut Parser,
        expr: Handle<HirExpr>,
        lhs: bool,
        body: &mut Block,
    ) -> Result<(Option<Handle<Expression>>, SourceMetadata)> {
        let HirExpr { kind, meta } = self.hir_exprs[expr].clone();

        let handle = match kind {
            HirExprKind::Access { base, index } => {
                let base = self.lower_expect(parser, base, true, body)?.0;
                let (index, index_meta) = self.lower_expect(parser, index, false, body)?;

                let pointer = parser
                    .solve_constant(self, index, index_meta)
                    .ok()
                    .and_then(|constant| {
                        Some(self.add_expression(
                            Expression::AccessIndex {
                                base,
                                index: match parser.module.constants[constant].inner {
                                    crate::ConstantInner::Scalar {
                                        value: ScalarValue::Uint(i),
                                        ..
                                    } => u32::try_from(i).ok()?,
                                    crate::ConstantInner::Scalar {
                                        value: ScalarValue::Sint(i),
                                        ..
                                    } => u32::try_from(i).ok()?,
                                    _ => return None,
                                },
                            },
                            body,
                        ))
                    })
                    .unwrap_or_else(|| {
                        self.add_expression(Expression::Access { base, index }, body)
                    });

                if !lhs {
                    let resolved = parser.resolve_type(self, pointer, meta)?;
                    if resolved.pointer_class().is_some() {
                        return Ok((
                            Some(self.add_expression(Expression::Load { pointer }, body)),
                            meta,
                        ));
                    }
                }

                pointer
            }
            HirExprKind::Select { base, field } => {
                let base = self.lower_expect(parser, base, lhs, body)?.0;

                parser.field_selection(self, lhs, body, base, &field, meta)?
            }
            HirExprKind::Constant(constant) if !lhs => {
                self.add_expression(Expression::Constant(constant), body)
            }
            HirExprKind::Binary { left, op, right } if !lhs => {
                let (mut left, left_meta) = self.lower_expect(parser, left, false, body)?;
                let (mut right, right_meta) = self.lower_expect(parser, right, false, body)?;

                match op {
                    BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => self
                        .implicit_conversion(parser, &mut right, right_meta, ScalarKind::Uint, 4)?,
                    _ => self.binary_implicit_conversion(
                        parser, &mut left, left_meta, &mut right, right_meta,
                    )?,
                }

                parser.typifier_grow(self, left, left_meta)?;
                parser.typifier_grow(self, right, right_meta)?;

                let left_inner = self.typifier.get(left, &parser.module.types);
                let right_inner = self.typifier.get(right, &parser.module.types);

                match (left_inner, right_inner) {
                    (&TypeInner::Vector { .. }, &TypeInner::Vector { .. })
                    | (&TypeInner::Matrix { .. }, &TypeInner::Matrix { .. }) => match op {
                        BinaryOperator::Equal | BinaryOperator::NotEqual => {
                            let equals = op == BinaryOperator::Equal;

                            let (op, fun) = match equals {
                                true => (BinaryOperator::Equal, RelationalFunction::All),
                                false => (BinaryOperator::NotEqual, RelationalFunction::Any),
                            };

                            let argument =
                                self.expressions
                                    .append(Expression::Binary { op, left, right });

                            self.add_expression(Expression::Relational { fun, argument }, body)
                        }
                        _ => self.add_expression(Expression::Binary { left, op, right }, body),
                    },
                    (&TypeInner::Vector { size, .. }, &TypeInner::Scalar { .. }) => match op {
                        BinaryOperator::Add
                        | BinaryOperator::Subtract
                        | BinaryOperator::Divide
                        | BinaryOperator::ShiftLeft
                        | BinaryOperator::ShiftRight => {
                            let scalar_vector =
                                self.add_expression(Expression::Splat { size, value: right }, body);

                            self.add_expression(
                                Expression::Binary {
                                    op,
                                    left,
                                    right: scalar_vector,
                                },
                                body,
                            )
                        }
                        _ => self.add_expression(Expression::Binary { left, op, right }, body),
                    },
                    (&TypeInner::Scalar { .. }, &TypeInner::Vector { size, .. }) => match op {
                        BinaryOperator::Add | BinaryOperator::Subtract | BinaryOperator::Divide => {
                            let scalar_vector =
                                self.add_expression(Expression::Splat { size, value: left }, body);

                            self.add_expression(
                                Expression::Binary {
                                    op,
                                    left: scalar_vector,
                                    right,
                                },
                                body,
                            )
                        }
                        _ => self.add_expression(Expression::Binary { left, op, right }, body),
                    },
                    _ => self.add_expression(Expression::Binary { left, op, right }, body),
                }
            }
            HirExprKind::Unary { op, expr } if !lhs => {
                let expr = self.lower_expect(parser, expr, false, body)?.0;

                self.add_expression(Expression::Unary { op, expr }, body)
            }
            HirExprKind::Variable(var) => {
                if lhs {
                    if !var.mutable {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Variable cannot be used in LHS position".into(),
                            ),
                            meta,
                        })
                    }

                    var.expr
                } else if var.load {
                    self.add_expression(Expression::Load { pointer: var.expr }, body)
                } else {
                    var.expr
                }
            }
            HirExprKind::Call(call) if !lhs => {
                let maybe_expr =
                    parser.function_or_constructor_call(self, body, call.kind, &call.args, meta)?;
                return Ok((maybe_expr, meta));
            }
            HirExprKind::Conditional {
                condition,
                accept,
                reject,
            } if !lhs => {
                let condition = self.lower_expect(parser, condition, false, body)?.0;
                let (mut accept, accept_meta) = self.lower_expect(parser, accept, false, body)?;
                let (mut reject, reject_meta) = self.lower_expect(parser, reject, false, body)?;

                self.binary_implicit_conversion(
                    parser,
                    &mut accept,
                    accept_meta,
                    &mut reject,
                    reject_meta,
                )?;

                self.add_expression(
                    Expression::Select {
                        condition,
                        accept,
                        reject,
                    },
                    body,
                )
            }
            HirExprKind::Assign { tgt, value } if !lhs => {
                let (pointer, ptr_meta) = self.lower_expect(parser, tgt, true, body)?;
                let (mut value, value_meta) = self.lower_expect(parser, value, false, body)?;

                let scalar_components = self.expr_scalar_components(parser, pointer, ptr_meta)?;

                if let Some((kind, width)) = scalar_components {
                    self.implicit_conversion(parser, &mut value, value_meta, kind, width)?;
                }

                if let Expression::Swizzle {
                    size,
                    mut vector,
                    pattern,
                } = self.expressions[pointer]
                {
                    // Stores to swizzled values are not directly supported,
                    // lower them as series of per-component stores.
                    let size = match size {
                        VectorSize::Bi => 2,
                        VectorSize::Tri => 3,
                        VectorSize::Quad => 4,
                    };

                    if let Expression::Load { pointer } = self.expressions[vector] {
                        vector = pointer;
                    }

                    #[allow(clippy::needless_range_loop)]
                    for index in 0..size {
                        let dst = self.add_expression(
                            Expression::AccessIndex {
                                base: vector,
                                index: pattern[index].index(),
                            },
                            body,
                        );
                        let src = self.add_expression(
                            Expression::AccessIndex {
                                base: value,
                                index: index as u32,
                            },
                            body,
                        );

                        self.emit_flush(body);
                        self.emit_start();

                        body.push(Statement::Store {
                            pointer: dst,
                            value: src,
                        });
                    }
                } else {
                    self.emit_flush(body);
                    self.emit_start();

                    body.push(Statement::Store { pointer, value });
                }

                value
            }
            HirExprKind::IncDec {
                increment,
                postfix,
                expr,
            } => {
                let op = match increment {
                    true => BinaryOperator::Add,
                    false => BinaryOperator::Subtract,
                };

                let pointer = self.lower_expect(parser, expr, true, body)?.0;
                let left = self.add_expression(Expression::Load { pointer }, body);

                let uint = match parser.resolve_type(self, left, meta)?.scalar_kind() {
                    Some(ScalarKind::Sint) => false,
                    Some(ScalarKind::Uint) => true,
                    _ => {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Increment/decrement operations must operate in integers".into(),
                            ),
                            meta,
                        });
                        true
                    }
                };

                let one = parser.module.constants.append(Constant {
                    name: None,
                    specialization: None,
                    inner: crate::ConstantInner::Scalar {
                        width: 4,
                        value: match uint {
                            true => crate::ScalarValue::Uint(1),
                            false => crate::ScalarValue::Sint(1),
                        },
                    },
                });
                let right = self.add_expression(Expression::Constant(one), body);

                let value = self.add_expression(Expression::Binary { op, left, right }, body);

                if postfix {
                    let local = self.locals.append(LocalVariable {
                        name: None,
                        ty: parser.module.types.fetch_or_append(Type {
                            name: None,
                            inner: TypeInner::Scalar {
                                kind: match uint {
                                    true => ScalarKind::Uint,
                                    false => ScalarKind::Sint,
                                },
                                width: 4,
                            },
                        }),
                        init: None,
                    });

                    let expr = self.add_expression(Expression::LocalVariable(local), body);
                    let load = self.add_expression(Expression::Load { pointer: expr }, body);

                    self.emit_flush(body);
                    self.emit_start();

                    body.push(Statement::Store {
                        pointer: expr,
                        value: left,
                    });

                    self.emit_flush(body);
                    self.emit_start();

                    body.push(Statement::Store { pointer, value });

                    load
                } else {
                    self.emit_flush(body);
                    self.emit_start();

                    body.push(Statement::Store { pointer, value });

                    left
                }
            }
            _ => {
                return Err(Error {
                    kind: ErrorKind::SemanticError(
                        format!("{:?} cannot be in the left hand side", self.hir_exprs[expr])
                            .into(),
                    ),
                    meta,
                })
            }
        };

        Ok((Some(handle), meta))
    }

    pub fn expr_scalar_components(
        &mut self,
        parser: &mut Parser,
        expr: Handle<Expression>,
        meta: SourceMetadata,
    ) -> Result<Option<(ScalarKind, crate::Bytes)>> {
        let ty = parser.resolve_type(self, expr, meta)?;
        Ok(scalar_components(ty))
    }

    pub fn expr_power(
        &mut self,
        parser: &mut Parser,
        expr: Handle<Expression>,
        meta: SourceMetadata,
    ) -> Result<Option<u32>> {
        Ok(self
            .expr_scalar_components(parser, expr, meta)?
            .and_then(|(kind, _)| type_power(kind)))
    }

    pub fn implicit_conversion(
        &mut self,
        parser: &mut Parser,
        expr: &mut Handle<Expression>,
        meta: SourceMetadata,
        kind: ScalarKind,
        width: crate::Bytes,
    ) -> Result<()> {
        if let (Some(tgt_power), Some(expr_power)) =
            (type_power(kind), self.expr_power(parser, *expr, meta)?)
        {
            if tgt_power > expr_power {
                *expr = self.expressions.append(Expression::As {
                    expr: *expr,
                    kind,
                    convert: Some(width),
                })
            }
        }

        Ok(())
    }

    pub fn binary_implicit_conversion(
        &mut self,
        parser: &mut Parser,
        left: &mut Handle<Expression>,
        left_meta: SourceMetadata,
        right: &mut Handle<Expression>,
        right_meta: SourceMetadata,
    ) -> Result<()> {
        let left_components = self.expr_scalar_components(parser, *left, left_meta)?;
        let right_components = self.expr_scalar_components(parser, *right, right_meta)?;

        if let (
            Some((left_power, left_width, left_kind)),
            Some((right_power, right_width, right_kind)),
        ) = (
            left_components.and_then(|(kind, width)| Some((type_power(kind)?, width, kind))),
            right_components.and_then(|(kind, width)| Some((type_power(kind)?, width, kind))),
        ) {
            match left_power.cmp(&right_power) {
                std::cmp::Ordering::Less => {
                    *left = self.expressions.append(Expression::As {
                        expr: *left,
                        kind: right_kind,
                        convert: Some(right_width),
                    })
                }
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => {
                    *right = self.expressions.append(Expression::As {
                        expr: *right,
                        kind: left_kind,
                        convert: Some(left_width),
                    })
                }
            }
        }

        Ok(())
    }

    pub fn implicit_splat(
        &mut self,
        parser: &mut Parser,

        expr: &mut Handle<Expression>,
        meta: SourceMetadata,
        vector_size: Option<VectorSize>,
    ) -> Result<()> {
        let expr_type = parser.resolve_type(self, *expr, meta)?;

        if let (&TypeInner::Scalar { .. }, Some(size)) = (expr_type, vector_size) {
            *expr = self
                .expressions
                .append(Expression::Splat { size, value: *expr })
        }

        Ok(())
    }

    pub fn vector_resize(
        &mut self,
        size: VectorSize,
        vector: Handle<Expression>,
        body: &mut Block,
    ) -> Handle<Expression> {
        self.add_expression(
            Expression::Swizzle {
                size,
                vector,
                pattern: crate::SwizzleComponent::XYZW,
            },
            body,
        )
    }
}

impl Index<Handle<Expression>> for Context {
    type Output = Expression;

    fn index(&self, index: Handle<Expression>) -> &Self::Output {
        &self.expressions[index]
    }
}
