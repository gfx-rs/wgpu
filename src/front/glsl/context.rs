use super::{
    ast::{
        GlobalLookup, GlobalLookupKind, HirExpr, HirExprKind, ParameterInfo, ParameterQualifier,
        VariableReference,
    },
    error::{Error, ErrorKind},
    types::{scalar_components, type_power},
    Parser, Result,
};
use crate::{
    front::{Emitter, Typifier},
    Arena, BinaryOperator, Block, Constant, Expression, FastHashMap, FunctionArgument, Handle,
    LocalVariable, RelationalFunction, ScalarKind, ScalarValue, Span, Statement, StorageClass,
    Type, TypeInner, VectorSize,
};
use std::{convert::TryFrom, ops::Index};

/// The position at which an expression is, used while lowering
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExprPos {
    /// The expression is in the left hand side of an assignment
    Lhs,
    /// The expression is in the right hand side of an assignment
    Rhs,
    /// The expression is an array being indexed, needed to allow constant
    /// arrays to be dinamically indexed
    ArrayBase {
        /// The index is a constant
        constant_index: bool,
    },
}

impl ExprPos {
    /// Returns an lhs position if the current position is lhs otherwise ArrayBase
    fn maybe_array_base(&self, constant_index: bool) -> Self {
        match *self {
            ExprPos::Lhs => *self,
            _ => ExprPos::ArrayBase { constant_index },
        }
    }
}

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

    pub typifier: Typifier,
    emitter: Emitter,
    stmt_ctx: Option<StmtContext>,
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

            typifier: Typifier::new(),
            emitter: Emitter::default(),
            stmt_ctx: Some(StmtContext::new()),
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
        let (expr, load, constant) = match kind {
            GlobalLookupKind::Variable(v) => {
                let span = parser.module.global_variables.get_span(v);
                let res = (
                    self.expressions.append(Expression::GlobalVariable(v), span),
                    parser.module.global_variables[v].class != StorageClass::Handle,
                    None,
                );
                self.emit_start();

                res
            }
            GlobalLookupKind::BlockSelect(handle, index) => {
                let span = parser.module.global_variables.get_span(handle);
                let base = self
                    .expressions
                    .append(Expression::GlobalVariable(handle), span);
                self.emit_start();
                let expr = self
                    .expressions
                    .append(Expression::AccessIndex { base, index }, span);

                (
                    expr,
                    {
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
                    },
                    None,
                )
            }
            GlobalLookupKind::Constant(v, ty) => {
                let span = parser.module.constants.get_span(v);
                let res = (
                    self.expressions.append(Expression::Constant(v), span),
                    false,
                    Some((v, ty)),
                );
                self.emit_start();
                res
            }
        };

        let var = VariableReference {
            expr,
            load,
            mutable,
            constant,
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

    pub fn add_expression(
        &mut self,
        expr: Expression,
        meta: Span,
        body: &mut Block,
    ) -> Handle<Expression> {
        let needs_pre_emit = expr.needs_pre_emit();
        if needs_pre_emit {
            self.emit_flush(body);
        }
        let handle = self.expressions.append(expr, meta);
        if needs_pre_emit {
            self.emit_start();
        }
        handle
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
                    constant: None,
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
        name_meta: Option<(String, Span)>,
        ty: Handle<Type>,
        qualifier: ParameterQualifier,
    ) {
        let index = self.arguments.len();
        let mut arg = FunctionArgument {
            name: name_meta.as_ref().map(|&(ref name, _)| name.clone()),
            ty,
            binding: None,
        };
        self.parameters.push(ty);

        let opaque = match parser.module.types[ty].inner {
            TypeInner::Image { .. } | TypeInner::Sampler { .. } => true,
            _ => false,
        };

        if qualifier.is_lhs() {
            let span = parser.module.types.get_span(arg.ty);
            arg.ty = parser.module.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Pointer {
                        base: arg.ty,
                        class: StorageClass::Function,
                    },
                },
                span,
            )
        }

        self.arguments.push(arg);

        self.parameters_info.push(ParameterInfo {
            qualifier,
            depth: false,
        });

        if let Some((name, meta)) = name_meta {
            let expr = self.add_expression(Expression::FunctionArgument(index as u32), meta, body);
            let mutable = qualifier != ParameterQualifier::Const && !opaque;
            let load = qualifier.is_lhs();

            if mutable && !load {
                let handle = self.locals.append(
                    LocalVariable {
                        name: Some(name.clone()),
                        ty,
                        init: None,
                    },
                    meta,
                );
                let local_expr = self.add_expression(Expression::LocalVariable(handle), meta, body);

                self.emit_flush(body);
                self.emit_start();

                body.push(
                    Statement::Store {
                        pointer: local_expr,
                        value: expr,
                    },
                    meta,
                );

                if let Some(current) = self.scopes.last_mut() {
                    (*current).insert(
                        name,
                        VariableReference {
                            expr: local_expr,
                            load: true,
                            mutable,
                            constant: None,
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
                        constant: None,
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

    /// Returns a [`StmtContext`](StmtContext) to be used in parsing and lowering
    ///
    /// # Panics
    /// - If more than one [`StmtContext`](StmtContext) are active at the same
    /// time or if the previous call didn't use it in lowering.
    #[must_use]
    pub fn stmt_ctx(&mut self) -> StmtContext {
        self.stmt_ctx.take().unwrap()
    }

    /// Lowers a [`HirExpr`](HirExpr) which might produce a [`Expression`](Expression).
    ///
    /// consumes a [`StmtContext`](StmtContext) returning it to the context so
    /// that it can be used again later.
    pub fn lower(
        &mut self,
        mut stmt: StmtContext,
        parser: &mut Parser,
        expr: Handle<HirExpr>,
        pos: ExprPos,
        body: &mut Block,
    ) -> Result<(Option<Handle<Expression>>, Span)> {
        let res = self.lower_inner(&stmt, parser, expr, pos, body);

        stmt.hir_exprs.clear();
        self.stmt_ctx = Some(stmt);

        res
    }

    /// Similar to [`lower`](Self::lower) but returns an error if the expression
    /// returns void (ie. doesn't produce a [`Expression`](Expression)).
    ///
    /// consumes a [`StmtContext`](StmtContext) returning it to the context so
    /// that it can be used again later.
    pub fn lower_expect(
        &mut self,
        mut stmt: StmtContext,
        parser: &mut Parser,
        expr: Handle<HirExpr>,
        pos: ExprPos,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, Span)> {
        let res = self.lower_expect_inner(&stmt, parser, expr, pos, body);

        stmt.hir_exprs.clear();
        self.stmt_ctx = Some(stmt);

        res
    }

    /// internal implementation of [`lower_expect`](Self::lower_expect)
    ///
    /// this method is only public because it's used in
    /// [`function_call`](Parser::function_call), unless you know what
    /// you're doing use [`lower_expect`](Self::lower_expect)
    pub fn lower_expect_inner(
        &mut self,
        stmt: &StmtContext,
        parser: &mut Parser,
        expr: Handle<HirExpr>,
        pos: ExprPos,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, Span)> {
        let (maybe_expr, meta) = self.lower_inner(stmt, parser, expr, pos, body)?;

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

    /// Internal implementation of [`lower`](Self::lower)
    fn lower_inner(
        &mut self,
        stmt: &StmtContext,
        parser: &mut Parser,
        expr: Handle<HirExpr>,
        pos: ExprPos,
        body: &mut Block,
    ) -> Result<(Option<Handle<Expression>>, Span)> {
        let HirExpr { ref kind, meta } = stmt.hir_exprs[expr];

        let handle = match *kind {
            HirExprKind::Access { base, index } => {
                let (index, index_meta) =
                    self.lower_expect_inner(stmt, parser, index, ExprPos::Rhs, body)?;
                let maybe_constant_index = match pos {
                    // Don't try to generate `AccessIndex` if in a LHS position, since it
                    // wouldn't produce a pointer.
                    ExprPos::Lhs => None,
                    _ => parser.solve_constant(self, index, index_meta).ok(),
                };

                let base = self
                    .lower_expect_inner(
                        stmt,
                        parser,
                        base,
                        pos.maybe_array_base(maybe_constant_index.is_some()),
                        body,
                    )?
                    .0;

                let pointer = maybe_constant_index
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
                            meta,
                            body,
                        ))
                    })
                    .unwrap_or_else(|| {
                        self.add_expression(Expression::Access { base, index }, meta, body)
                    });

                if ExprPos::Rhs == pos {
                    let resolved = parser.resolve_type(self, pointer, meta)?;
                    if resolved.pointer_class().is_some() {
                        return Ok((
                            Some(self.add_expression(Expression::Load { pointer }, meta, body)),
                            meta,
                        ));
                    }
                }

                pointer
            }
            HirExprKind::Select { base, ref field } => {
                let base = self.lower_expect_inner(stmt, parser, base, pos, body)?.0;

                parser.field_selection(self, ExprPos::Lhs == pos, body, base, field, meta)?
            }
            HirExprKind::Constant(constant) if pos != ExprPos::Lhs => {
                self.add_expression(Expression::Constant(constant), meta, body)
            }
            HirExprKind::Binary { left, op, right } if pos != ExprPos::Lhs => {
                let (mut left, left_meta) =
                    self.lower_expect_inner(stmt, parser, left, pos, body)?;
                let (mut right, right_meta) =
                    self.lower_expect_inner(stmt, parser, right, pos, body)?;

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

                            let argument = self
                                .expressions
                                .append(Expression::Binary { op, left, right }, meta);

                            self.add_expression(
                                Expression::Relational { fun, argument },
                                meta,
                                body,
                            )
                        }
                        _ => {
                            self.add_expression(Expression::Binary { left, op, right }, meta, body)
                        }
                    },
                    (&TypeInner::Vector { size, .. }, &TypeInner::Scalar { .. }) => match op {
                        BinaryOperator::Add
                        | BinaryOperator::Subtract
                        | BinaryOperator::Divide
                        | BinaryOperator::And
                        | BinaryOperator::ExclusiveOr
                        | BinaryOperator::InclusiveOr
                        | BinaryOperator::ShiftLeft
                        | BinaryOperator::ShiftRight => {
                            let scalar_vector = self.add_expression(
                                Expression::Splat { size, value: right },
                                meta,
                                body,
                            );

                            self.add_expression(
                                Expression::Binary {
                                    op,
                                    left,
                                    right: scalar_vector,
                                },
                                meta,
                                body,
                            )
                        }
                        _ => {
                            self.add_expression(Expression::Binary { left, op, right }, meta, body)
                        }
                    },
                    (&TypeInner::Scalar { .. }, &TypeInner::Vector { size, .. }) => match op {
                        BinaryOperator::Add
                        | BinaryOperator::Subtract
                        | BinaryOperator::Divide
                        | BinaryOperator::And
                        | BinaryOperator::ExclusiveOr
                        | BinaryOperator::InclusiveOr => {
                            let scalar_vector = self.add_expression(
                                Expression::Splat { size, value: left },
                                meta,
                                body,
                            );

                            self.add_expression(
                                Expression::Binary {
                                    op,
                                    left: scalar_vector,
                                    right,
                                },
                                meta,
                                body,
                            )
                        }
                        _ => {
                            self.add_expression(Expression::Binary { left, op, right }, meta, body)
                        }
                    },
                    _ => self.add_expression(Expression::Binary { left, op, right }, meta, body),
                }
            }
            HirExprKind::Unary { op, expr } if pos != ExprPos::Lhs => {
                let expr = self.lower_expect_inner(stmt, parser, expr, pos, body)?.0;

                self.add_expression(Expression::Unary { op, expr }, meta, body)
            }
            HirExprKind::Variable(ref var) => match pos {
                ExprPos::Lhs => {
                    if !var.mutable {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Variable cannot be used in LHS position".into(),
                            ),
                            meta,
                        })
                    }

                    var.expr
                }
                ExprPos::ArrayBase {
                    constant_index: false,
                } => {
                    if let Some((constant, ty)) = var.constant {
                        let local = self.locals.append(
                            LocalVariable {
                                name: None,
                                ty,
                                init: Some(constant),
                            },
                            Span::default(),
                        );

                        self.add_expression(Expression::LocalVariable(local), Span::default(), body)
                    } else {
                        var.expr
                    }
                }
                _ if var.load => {
                    self.add_expression(Expression::Load { pointer: var.expr }, meta, body)
                }
                _ => var.expr,
            },
            HirExprKind::Call(ref call) if pos != ExprPos::Lhs => {
                let maybe_expr = parser.function_or_constructor_call(
                    self,
                    stmt,
                    body,
                    call.kind.clone(),
                    &call.args,
                    meta,
                )?;
                return Ok((maybe_expr, meta));
            }
            HirExprKind::Conditional {
                condition,
                accept,
                reject,
            } if ExprPos::Lhs != pos => {
                let condition = self
                    .lower_expect_inner(stmt, parser, condition, ExprPos::Rhs, body)?
                    .0;
                let (mut accept, accept_meta) =
                    self.lower_expect_inner(stmt, parser, accept, pos, body)?;
                let (mut reject, reject_meta) =
                    self.lower_expect_inner(stmt, parser, reject, pos, body)?;

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
                    meta,
                    body,
                )
            }
            HirExprKind::Assign { tgt, value } if ExprPos::Lhs != pos => {
                let (pointer, ptr_meta) =
                    self.lower_expect_inner(stmt, parser, tgt, ExprPos::Lhs, body)?;
                let (mut value, value_meta) =
                    self.lower_expect_inner(stmt, parser, value, ExprPos::Rhs, body)?;

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
                            meta,
                            body,
                        );
                        let src = self.add_expression(
                            Expression::AccessIndex {
                                base: value,
                                index: index as u32,
                            },
                            meta,
                            body,
                        );

                        self.emit_flush(body);
                        self.emit_start();

                        body.push(
                            Statement::Store {
                                pointer: dst,
                                value: src,
                            },
                            meta,
                        );
                    }
                } else {
                    self.emit_flush(body);
                    self.emit_start();

                    body.push(Statement::Store { pointer, value }, meta);
                }

                value
            }
            HirExprKind::PrePostfix { op, postfix, expr } if ExprPos::Lhs != pos => {
                let pointer = self
                    .lower_expect_inner(stmt, parser, expr, ExprPos::Lhs, body)?
                    .0;
                let left = self.add_expression(Expression::Load { pointer }, meta, body);

                let make_constant_inner = |kind, width| {
                    let value = match kind {
                        ScalarKind::Sint => crate::ScalarValue::Sint(1),
                        ScalarKind::Uint => crate::ScalarValue::Uint(1),
                        ScalarKind::Float => crate::ScalarValue::Float(1.0),
                        ScalarKind::Bool => return None,
                    };

                    Some(crate::ConstantInner::Scalar { width, value })
                };
                let res = match *parser.resolve_type(self, left, meta)? {
                    TypeInner::Scalar { kind, width } => {
                        let ty = TypeInner::Scalar { kind, width };
                        make_constant_inner(kind, width).map(|i| (ty, i, None, None))
                    }
                    TypeInner::Vector { size, kind, width } => {
                        let ty = TypeInner::Vector { size, kind, width };
                        make_constant_inner(kind, width).map(|i| (ty, i, Some(size), None))
                    }
                    TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    } => {
                        let ty = TypeInner::Matrix {
                            columns,
                            rows,
                            width,
                        };
                        make_constant_inner(ScalarKind::Float, width)
                            .map(|i| (ty, i, Some(rows), Some(columns)))
                    }
                    _ => None,
                };
                let (ty_inner, inner, rows, columns) = match res {
                    Some(res) => res,
                    None => {
                        parser.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Increment/decrement only works on scalar/vector/matrix".into(),
                            ),
                            meta,
                        });
                        return Ok((Some(left), meta));
                    }
                };

                let constant_1 = parser.module.constants.append(
                    Constant {
                        name: None,
                        specialization: None,
                        inner,
                    },
                    Default::default(),
                );
                let mut right = self.add_expression(Expression::Constant(constant_1), meta, body);

                // Glsl allows pre/postfixes operations on vectors and matrices, so if the
                // target is either of them change the right side of the addition to be splatted
                // to the same size as the target, furthermore if the target is a matrix
                // use a composed matrix using the splatted value.
                if let Some(size) = rows {
                    right =
                        self.add_expression(Expression::Splat { size, value: right }, meta, body);

                    if let Some(cols) = columns {
                        let ty = parser.module.types.insert(
                            Type {
                                name: None,
                                inner: ty_inner,
                            },
                            meta,
                        );

                        right = self.add_expression(
                            Expression::Compose {
                                ty,
                                components: std::iter::repeat(right).take(cols as usize).collect(),
                            },
                            meta,
                            body,
                        );
                    }
                }

                let value = self.add_expression(Expression::Binary { op, left, right }, meta, body);

                self.emit_flush(body);
                self.emit_start();

                body.push(Statement::Store { pointer, value }, meta);

                if postfix {
                    left
                } else {
                    value
                }
            }
            _ => {
                return Err(Error {
                    kind: ErrorKind::SemanticError(
                        format!("{:?} cannot be in the left hand side", stmt.hir_exprs[expr])
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
        meta: Span,
    ) -> Result<Option<(ScalarKind, crate::Bytes)>> {
        let ty = parser.resolve_type(self, expr, meta)?;
        Ok(scalar_components(ty))
    }

    pub fn expr_power(
        &mut self,
        parser: &mut Parser,
        expr: Handle<Expression>,
        meta: Span,
    ) -> Result<Option<u32>> {
        Ok(self
            .expr_scalar_components(parser, expr, meta)?
            .and_then(|(kind, width)| type_power(kind, width)))
    }

    pub fn conversion(
        &mut self,
        expr: &mut Handle<Expression>,
        meta: Span,
        kind: ScalarKind,
        width: crate::Bytes,
    ) -> Result<()> {
        *expr = self.expressions.append(
            Expression::As {
                expr: *expr,
                kind,
                convert: Some(width),
            },
            meta,
        );

        Ok(())
    }

    pub fn implicit_conversion(
        &mut self,
        parser: &mut Parser,
        expr: &mut Handle<Expression>,
        meta: Span,
        kind: ScalarKind,
        width: crate::Bytes,
    ) -> Result<()> {
        if let (Some(tgt_power), Some(expr_power)) = (
            type_power(kind, width),
            self.expr_power(parser, *expr, meta)?,
        ) {
            if tgt_power > expr_power {
                self.conversion(expr, meta, kind, width)?;
            }
        }

        Ok(())
    }

    pub fn binary_implicit_conversion(
        &mut self,
        parser: &mut Parser,
        left: &mut Handle<Expression>,
        left_meta: Span,
        right: &mut Handle<Expression>,
        right_meta: Span,
    ) -> Result<()> {
        let left_components = self.expr_scalar_components(parser, *left, left_meta)?;
        let right_components = self.expr_scalar_components(parser, *right, right_meta)?;

        if let (
            Some((left_power, left_width, left_kind)),
            Some((right_power, right_width, right_kind)),
        ) = (
            left_components.and_then(|(kind, width)| Some((type_power(kind, width)?, width, kind))),
            right_components
                .and_then(|(kind, width)| Some((type_power(kind, width)?, width, kind))),
        ) {
            match left_power.cmp(&right_power) {
                std::cmp::Ordering::Less => {
                    self.conversion(left, left_meta, right_kind, right_width)?;
                }
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => {
                    self.conversion(right, right_meta, left_kind, left_width)?;
                }
            }
        }

        Ok(())
    }

    pub fn implicit_splat(
        &mut self,
        parser: &mut Parser,
        expr: &mut Handle<Expression>,
        meta: Span,
        vector_size: Option<VectorSize>,
    ) -> Result<()> {
        let expr_type = parser.resolve_type(self, *expr, meta)?;

        if let (&TypeInner::Scalar { .. }, Some(size)) = (expr_type, vector_size) {
            *expr = self
                .expressions
                .append(Expression::Splat { size, value: *expr }, meta)
        }

        Ok(())
    }

    pub fn vector_resize(
        &mut self,
        size: VectorSize,
        vector: Handle<Expression>,
        meta: Span,
        body: &mut Block,
    ) -> Handle<Expression> {
        self.add_expression(
            Expression::Swizzle {
                size,
                vector,
                pattern: crate::SwizzleComponent::XYZW,
            },
            meta,
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

/// Helper struct passed when parsing expressions
///
/// This struct should only be obtained trough [`stmt_ctx`](Context::stmt_ctx)
/// and only one of these may be active at any time per context.
#[derive(Debug)]
pub struct StmtContext {
    /// A arena of high level expressions which can be lowered trough a
    /// [`Context`](Context) to naga's [`Expression`](crate::Expression)s
    pub hir_exprs: Arena<HirExpr>,
}

impl StmtContext {
    fn new() -> Self {
        StmtContext {
            hir_exprs: Arena::new(),
        }
    }
}
