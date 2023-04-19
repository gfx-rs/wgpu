use super::{
    ast::{
        GlobalLookup, GlobalLookupKind, HirExpr, HirExprKind, ParameterInfo, ParameterQualifier,
        VariableReference,
    },
    error::{Error, ErrorKind},
    types::{scalar_components, type_power},
    Frontend, Result,
};
use crate::{
    front::{Emitter, Typifier},
    AddressSpace, Arena, BinaryOperator, Block, Expression, FastHashMap, FunctionArgument, Handle,
    Literal, LocalVariable, RelationalFunction, ScalarKind, Span, Statement, Type, TypeInner,
    VectorSize,
};
use std::ops::Index;

/// The position at which an expression is, used while lowering
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExprPos {
    /// The expression is in the left hand side of an assignment
    Lhs,
    /// The expression is in the right hand side of an assignment
    Rhs,
    /// The expression is an array being indexed, needed to allow constant
    /// arrays to be dynamically indexed
    AccessBase {
        /// The index is a constant
        constant_index: bool,
    },
}

impl ExprPos {
    /// Returns an lhs position if the current position is lhs otherwise AccessBase
    const fn maybe_access_base(&self, constant_index: bool) -> Self {
        match *self {
            ExprPos::Lhs
            | ExprPos::AccessBase {
                constant_index: false,
            } => *self,
            _ => ExprPos::AccessBase { constant_index },
        }
    }
}

#[derive(Debug)]
pub struct Context {
    pub expressions: Arena<Expression>,
    pub locals: Arena<LocalVariable>,

    /// The [`FunctionArgument`]s for the final [`crate::Function`].
    ///
    /// Parameters with the `out` and `inout` qualifiers have [`Pointer`] types
    /// here. For example, an `inout vec2 a` argument would be a [`Pointer`] to
    /// a [`Vector`].
    ///
    /// [`Pointer`]: crate::TypeInner::Pointer
    /// [`Vector`]: crate::TypeInner::Vector
    pub arguments: Vec<FunctionArgument>,

    /// The parameter types given in the source code.
    ///
    /// The `out` and `inout` qualifiers don't affect the types that appear
    /// here. For example, an `inout vec2 a` argument would simply be a
    /// [`Vector`], not a pointer to one.
    ///
    /// [`Vector`]: crate::TypeInner::Vector
    pub parameters: Vec<Handle<Type>>,
    pub parameters_info: Vec<ParameterInfo>,

    pub symbol_table: crate::front::SymbolTable<String, VariableReference>,
    pub samplers: FastHashMap<Handle<Expression>, Handle<Expression>>,

    pub typifier: Typifier,
    emitter: Emitter,
    stmt_ctx: Option<StmtContext>,
}

impl Context {
    pub fn new(frontend: &Frontend, body: &mut Block) -> Self {
        let mut this = Context {
            expressions: Arena::new(),
            locals: Arena::new(),
            arguments: Vec::new(),

            parameters: Vec::new(),
            parameters_info: Vec::new(),

            symbol_table: crate::front::SymbolTable::default(),
            samplers: FastHashMap::default(),

            typifier: Typifier::new(),
            emitter: Emitter::default(),
            stmt_ctx: Some(StmtContext::new()),
        };

        this.emit_start();

        for &(ref name, lookup) in frontend.global_variables.iter() {
            this.add_global(frontend, name, lookup, body)
        }

        this
    }

    pub fn add_global(
        &mut self,
        frontend: &Frontend,
        name: &str,
        GlobalLookup {
            kind,
            entry_arg,
            mutable,
        }: GlobalLookup,
        body: &mut Block,
    ) {
        let (expr, load, constant) = match kind {
            GlobalLookupKind::Variable(v) => {
                let span = frontend.module.global_variables.get_span(v);
                (
                    self.add_expression(Expression::GlobalVariable(v), span, body),
                    frontend.module.global_variables[v].space != AddressSpace::Handle,
                    None,
                )
            }
            GlobalLookupKind::BlockSelect(handle, index) => {
                let span = frontend.module.global_variables.get_span(handle);
                let base = self.add_expression(Expression::GlobalVariable(handle), span, body);
                let expr = self.add_expression(Expression::AccessIndex { base, index }, span, body);

                (
                    expr,
                    {
                        let ty = frontend.module.global_variables[handle].ty;

                        match frontend.module.types[ty].inner {
                            TypeInner::Struct { ref members, .. } => {
                                if let TypeInner::Array {
                                    size: crate::ArraySize::Dynamic,
                                    ..
                                } = frontend.module.types[members[index as usize].ty].inner
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
                let span = frontend.module.constants.get_span(v);
                (
                    self.add_expression(Expression::Constant(v), span, body),
                    false,
                    Some((v, ty)),
                )
            }
        };

        let var = VariableReference {
            expr,
            load,
            mutable,
            constant,
            entry_arg,
        };

        self.symbol_table.add(name.into(), var);
    }

    /// Starts the expression emitter
    ///
    /// # Panics
    ///
    /// - If called twice in a row without calling [`emit_end`][Self::emit_end].
    #[inline]
    pub fn emit_start(&mut self) {
        self.emitter.start(&self.expressions)
    }

    /// Emits all the expressions captured by the emitter to the passed `body`
    ///
    /// # Panics
    ///
    /// - If called before calling [`emit_start`].
    /// - If called twice in a row without calling [`emit_start`].
    ///
    /// [`emit_start`]: Self::emit_start
    pub fn emit_end(&mut self, body: &mut Block) {
        body.extend(self.emitter.finish(&self.expressions))
    }

    /// Emits all the expressions captured by the emitter to the passed `body`
    /// and starts the emitter again
    ///
    /// # Panics
    ///
    /// - If called before calling [`emit_start`][Self::emit_start].
    pub fn emit_restart(&mut self, body: &mut Block) {
        self.emit_end(body);
        self.emit_start()
    }

    pub fn add_expression(
        &mut self,
        expr: Expression,
        meta: Span,
        body: &mut Block,
    ) -> Handle<Expression> {
        let needs_pre_emit = expr.needs_pre_emit();
        if needs_pre_emit {
            self.emit_end(body);
        }
        let handle = self.expressions.append(expr, meta);
        if needs_pre_emit {
            self.emit_start();
        }
        handle
    }

    /// Add variable to current scope
    ///
    /// Returns a variable if a variable with the same name was already defined,
    /// otherwise returns `None`
    pub fn add_local_var(
        &mut self,
        name: String,
        expr: Handle<Expression>,
        mutable: bool,
    ) -> Option<VariableReference> {
        let var = VariableReference {
            expr,
            load: true,
            mutable,
            constant: None,
            entry_arg: None,
        };

        self.symbol_table.add(name, var)
    }

    /// Add function argument to current scope
    pub fn add_function_arg(
        &mut self,
        frontend: &mut Frontend,
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

        let opaque = match frontend.module.types[ty].inner {
            TypeInner::Image { .. } | TypeInner::Sampler { .. } => true,
            _ => false,
        };

        if qualifier.is_lhs() {
            let span = frontend.module.types.get_span(arg.ty);
            arg.ty = frontend.module.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Pointer {
                        base: arg.ty,
                        space: AddressSpace::Function,
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

            let var = if mutable && !load {
                let handle = self.locals.append(
                    LocalVariable {
                        name: Some(name.clone()),
                        ty,
                        init: None,
                    },
                    meta,
                );
                let local_expr = self.add_expression(Expression::LocalVariable(handle), meta, body);

                self.emit_restart(body);

                body.push(
                    Statement::Store {
                        pointer: local_expr,
                        value: expr,
                    },
                    meta,
                );

                VariableReference {
                    expr: local_expr,
                    load: true,
                    mutable,
                    constant: None,
                    entry_arg: None,
                }
            } else {
                VariableReference {
                    expr,
                    load,
                    mutable,
                    constant: None,
                    entry_arg: None,
                }
            };

            self.symbol_table.add(name, var);
        }
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
        frontend: &mut Frontend,
        expr: Handle<HirExpr>,
        pos: ExprPos,
        body: &mut Block,
    ) -> Result<(Option<Handle<Expression>>, Span)> {
        let res = self.lower_inner(&stmt, frontend, expr, pos, body);

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
        frontend: &mut Frontend,
        expr: Handle<HirExpr>,
        pos: ExprPos,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, Span)> {
        let res = self.lower_expect_inner(&stmt, frontend, expr, pos, body);

        stmt.hir_exprs.clear();
        self.stmt_ctx = Some(stmt);

        res
    }

    /// internal implementation of [`lower_expect`](Self::lower_expect)
    ///
    /// this method is only public because it's used in
    /// [`function_call`](Frontend::function_call), unless you know what
    /// you're doing use [`lower_expect`](Self::lower_expect)
    pub fn lower_expect_inner(
        &mut self,
        stmt: &StmtContext,
        frontend: &mut Frontend,
        expr: Handle<HirExpr>,
        pos: ExprPos,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, Span)> {
        let (maybe_expr, meta) = self.lower_inner(stmt, frontend, expr, pos, body)?;

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

    fn lower_store(
        &mut self,
        pointer: Handle<Expression>,
        value: Handle<Expression>,
        meta: Span,
        body: &mut Block,
    ) {
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

                self.emit_restart(body);

                body.push(
                    Statement::Store {
                        pointer: dst,
                        value: src,
                    },
                    meta,
                );
            }
        } else {
            self.emit_restart(body);

            body.push(Statement::Store { pointer, value }, meta);
        }
    }

    /// Internal implementation of [`lower`](Self::lower)
    fn lower_inner(
        &mut self,
        stmt: &StmtContext,
        frontend: &mut Frontend,
        expr: Handle<HirExpr>,
        pos: ExprPos,
        body: &mut Block,
    ) -> Result<(Option<Handle<Expression>>, Span)> {
        let HirExpr { ref kind, meta } = stmt.hir_exprs[expr];

        log::debug!("Lowering {:?} (kind {:?}, pos {:?})", expr, kind, pos);

        let handle = match *kind {
            HirExprKind::Access { base, index } => {
                let (index, index_meta) =
                    self.lower_expect_inner(stmt, frontend, index, ExprPos::Rhs, body)?;
                let maybe_constant_index = match pos {
                    // Don't try to generate `AccessIndex` if in a LHS position, since it
                    // wouldn't produce a pointer.
                    ExprPos::Lhs => None,
                    _ => frontend.solve_constant(self, index, index_meta).ok(),
                };

                let base = self
                    .lower_expect_inner(
                        stmt,
                        frontend,
                        base,
                        pos.maybe_access_base(maybe_constant_index.is_some()),
                        body,
                    )?
                    .0;

                let pointer = maybe_constant_index
                    .and_then(|const_expr| {
                        Some(self.add_expression(
                            Expression::AccessIndex {
                                base,
                                index:
                                    frontend.module.to_ctx().eval_expr_to_u32(const_expr).ok()?,
                            },
                            meta,
                            body,
                        ))
                    })
                    .unwrap_or_else(|| {
                        self.add_expression(Expression::Access { base, index }, meta, body)
                    });

                if ExprPos::Rhs == pos {
                    let resolved = frontend.resolve_type(self, pointer, meta)?;
                    if resolved.pointer_space().is_some() {
                        return Ok((
                            Some(self.add_expression(Expression::Load { pointer }, meta, body)),
                            meta,
                        ));
                    }
                }

                pointer
            }
            HirExprKind::Select { base, ref field } => {
                let base = self.lower_expect_inner(stmt, frontend, base, pos, body)?.0;

                frontend.field_selection(self, pos, body, base, field, meta)?
            }
            HirExprKind::Literal(literal) if pos != ExprPos::Lhs => {
                self.add_expression(Expression::Literal(literal), meta, body)
            }
            HirExprKind::Binary { left, op, right } if pos != ExprPos::Lhs => {
                let (mut left, left_meta) =
                    self.lower_expect_inner(stmt, frontend, left, ExprPos::Rhs, body)?;
                let (mut right, right_meta) =
                    self.lower_expect_inner(stmt, frontend, right, ExprPos::Rhs, body)?;

                match op {
                    BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => self
                        .implicit_conversion(
                            frontend,
                            &mut right,
                            right_meta,
                            ScalarKind::Uint,
                            4,
                            body,
                        )?,
                    _ => self.binary_implicit_conversion(
                        frontend, &mut left, left_meta, &mut right, right_meta, body,
                    )?,
                }

                frontend.typifier_grow(self, left, left_meta)?;
                frontend.typifier_grow(self, right, right_meta)?;

                let left_inner = self.typifier.get(left, &frontend.module.types);
                let right_inner = self.typifier.get(right, &frontend.module.types);

                match (left_inner, right_inner) {
                    (
                        &TypeInner::Matrix {
                            columns: left_columns,
                            rows: left_rows,
                            width: left_width,
                        },
                        &TypeInner::Matrix {
                            columns: right_columns,
                            rows: right_rows,
                            width: right_width,
                        },
                    ) => {
                        let dimensions_ok = if op == BinaryOperator::Multiply {
                            left_columns == right_rows
                        } else {
                            left_columns == right_columns && left_rows == right_rows
                        };

                        // Check that the two arguments have the same dimensions
                        if !dimensions_ok || left_width != right_width {
                            frontend.errors.push(Error {
                                kind: ErrorKind::SemanticError(
                                    format!(
                                        "Cannot apply operation to {left_inner:?} and {right_inner:?}"
                                    )
                                    .into(),
                                ),
                                meta,
                            })
                        }

                        match op {
                            BinaryOperator::Divide => {
                                // Naga IR doesn't support matrix division so we need to
                                // divide the columns individually and reassemble the matrix
                                let mut components = Vec::with_capacity(left_columns as usize);

                                for index in 0..left_columns as u32 {
                                    // Get the column vectors
                                    let left_vector = self.add_expression(
                                        Expression::AccessIndex { base: left, index },
                                        meta,
                                        body,
                                    );
                                    let right_vector = self.add_expression(
                                        Expression::AccessIndex { base: right, index },
                                        meta,
                                        body,
                                    );

                                    // Divide the vectors
                                    let column = self.add_expression(
                                        Expression::Binary {
                                            op,
                                            left: left_vector,
                                            right: right_vector,
                                        },
                                        meta,
                                        body,
                                    );

                                    components.push(column)
                                }

                                // Rebuild the matrix from the divided vectors
                                self.add_expression(
                                    Expression::Compose {
                                        ty: frontend.module.types.insert(
                                            Type {
                                                name: None,
                                                inner: TypeInner::Matrix {
                                                    columns: left_columns,
                                                    rows: left_rows,
                                                    width: left_width,
                                                },
                                            },
                                            Span::default(),
                                        ),
                                        components,
                                    },
                                    meta,
                                    body,
                                )
                            }
                            BinaryOperator::Equal | BinaryOperator::NotEqual => {
                                // Naga IR doesn't support matrix comparisons so we need to
                                // compare the columns individually and then fold them together
                                //
                                // The folding is done using a logical and for equality and
                                // a logical or for inequality
                                let equals = op == BinaryOperator::Equal;

                                let (op, combine, fun) = match equals {
                                    true => (
                                        BinaryOperator::Equal,
                                        BinaryOperator::LogicalAnd,
                                        RelationalFunction::All,
                                    ),
                                    false => (
                                        BinaryOperator::NotEqual,
                                        BinaryOperator::LogicalOr,
                                        RelationalFunction::Any,
                                    ),
                                };

                                let mut root = None;

                                for index in 0..left_columns as u32 {
                                    // Get the column vectors
                                    let left_vector = self.add_expression(
                                        Expression::AccessIndex { base: left, index },
                                        meta,
                                        body,
                                    );
                                    let right_vector = self.add_expression(
                                        Expression::AccessIndex { base: right, index },
                                        meta,
                                        body,
                                    );

                                    let argument = self.add_expression(
                                        Expression::Binary {
                                            op,
                                            left: left_vector,
                                            right: right_vector,
                                        },
                                        meta,
                                        body,
                                    );

                                    // The result of comparing two vectors is a boolean vector
                                    // so use a relational function like all to get a single
                                    // boolean value
                                    let compare = self.add_expression(
                                        Expression::Relational { fun, argument },
                                        meta,
                                        body,
                                    );

                                    // Fold the result
                                    root = Some(match root {
                                        Some(right) => self.add_expression(
                                            Expression::Binary {
                                                op: combine,
                                                left: compare,
                                                right,
                                            },
                                            meta,
                                            body,
                                        ),
                                        None => compare,
                                    });
                                }

                                root.unwrap()
                            }
                            _ => self.add_expression(
                                Expression::Binary { left, op, right },
                                meta,
                                body,
                            ),
                        }
                    }
                    (&TypeInner::Vector { .. }, &TypeInner::Vector { .. }) => match op {
                        BinaryOperator::Equal | BinaryOperator::NotEqual => {
                            let equals = op == BinaryOperator::Equal;

                            let (op, fun) = match equals {
                                true => (BinaryOperator::Equal, RelationalFunction::All),
                                false => (BinaryOperator::NotEqual, RelationalFunction::Any),
                            };

                            let argument = self.add_expression(
                                Expression::Binary { op, left, right },
                                meta,
                                body,
                            );

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
                    (
                        &TypeInner::Scalar {
                            width: left_width, ..
                        },
                        &TypeInner::Matrix {
                            rows,
                            columns,
                            width: right_width,
                        },
                    ) => {
                        // Check that the two arguments have the same width
                        if left_width != right_width {
                            frontend.errors.push(Error {
                                kind: ErrorKind::SemanticError(
                                    format!(
                                        "Cannot apply operation to {left_inner:?} and {right_inner:?}"
                                    )
                                    .into(),
                                ),
                                meta,
                            })
                        }

                        match op {
                            BinaryOperator::Divide
                            | BinaryOperator::Add
                            | BinaryOperator::Subtract => {
                                // Naga IR doesn't support all matrix by scalar operations so
                                // we need for some to turn the scalar into a vector by
                                // splatting it and then for each column vector apply the
                                // operation and finally reconstruct the matrix
                                let scalar_vector = self.add_expression(
                                    Expression::Splat {
                                        size: rows,
                                        value: left,
                                    },
                                    meta,
                                    body,
                                );

                                let mut components = Vec::with_capacity(columns as usize);

                                for index in 0..columns as u32 {
                                    // Get the column vector
                                    let matrix_column = self.add_expression(
                                        Expression::AccessIndex { base: right, index },
                                        meta,
                                        body,
                                    );

                                    // Apply the operation to the splatted vector and
                                    // the column vector
                                    let column = self.add_expression(
                                        Expression::Binary {
                                            op,
                                            left: scalar_vector,
                                            right: matrix_column,
                                        },
                                        meta,
                                        body,
                                    );

                                    components.push(column)
                                }

                                // Rebuild the matrix from the operation result vectors
                                self.add_expression(
                                    Expression::Compose {
                                        ty: frontend.module.types.insert(
                                            Type {
                                                name: None,
                                                inner: TypeInner::Matrix {
                                                    columns,
                                                    rows,
                                                    width: left_width,
                                                },
                                            },
                                            Span::default(),
                                        ),
                                        components,
                                    },
                                    meta,
                                    body,
                                )
                            }
                            _ => self.add_expression(
                                Expression::Binary { left, op, right },
                                meta,
                                body,
                            ),
                        }
                    }
                    (
                        &TypeInner::Matrix {
                            rows,
                            columns,
                            width: left_width,
                        },
                        &TypeInner::Scalar {
                            width: right_width, ..
                        },
                    ) => {
                        // Check that the two arguments have the same width
                        if left_width != right_width {
                            frontend.errors.push(Error {
                                kind: ErrorKind::SemanticError(
                                    format!(
                                        "Cannot apply operation to {left_inner:?} and {right_inner:?}"
                                    )
                                    .into(),
                                ),
                                meta,
                            })
                        }

                        match op {
                            BinaryOperator::Divide
                            | BinaryOperator::Add
                            | BinaryOperator::Subtract => {
                                // Naga IR doesn't support all matrix by scalar operations so
                                // we need for some to turn the scalar into a vector by
                                // splatting it and then for each column vector apply the
                                // operation and finally reconstruct the matrix

                                let scalar_vector = self.add_expression(
                                    Expression::Splat {
                                        size: rows,
                                        value: right,
                                    },
                                    meta,
                                    body,
                                );

                                let mut components = Vec::with_capacity(columns as usize);

                                for index in 0..columns as u32 {
                                    // Get the column vector
                                    let matrix_column = self.add_expression(
                                        Expression::AccessIndex { base: left, index },
                                        meta,
                                        body,
                                    );

                                    // Apply the operation to the splatted vector and
                                    // the column vector
                                    let column = self.add_expression(
                                        Expression::Binary {
                                            op,
                                            left: matrix_column,
                                            right: scalar_vector,
                                        },
                                        meta,
                                        body,
                                    );

                                    components.push(column)
                                }

                                // Rebuild the matrix from the operation result vectors
                                self.add_expression(
                                    Expression::Compose {
                                        ty: frontend.module.types.insert(
                                            Type {
                                                name: None,
                                                inner: TypeInner::Matrix {
                                                    columns,
                                                    rows,
                                                    width: left_width,
                                                },
                                            },
                                            Span::default(),
                                        ),
                                        components,
                                    },
                                    meta,
                                    body,
                                )
                            }
                            _ => self.add_expression(
                                Expression::Binary { left, op, right },
                                meta,
                                body,
                            ),
                        }
                    }
                    _ => self.add_expression(Expression::Binary { left, op, right }, meta, body),
                }
            }
            HirExprKind::Unary { op, expr } if pos != ExprPos::Lhs => {
                let expr = self
                    .lower_expect_inner(stmt, frontend, expr, ExprPos::Rhs, body)?
                    .0;

                self.add_expression(Expression::Unary { op, expr }, meta, body)
            }
            HirExprKind::Variable(ref var) => match pos {
                ExprPos::Lhs => {
                    if !var.mutable {
                        frontend.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Variable cannot be used in LHS position".into(),
                            ),
                            meta,
                        })
                    }

                    var.expr
                }
                ExprPos::AccessBase { constant_index } => {
                    // If the index isn't constant all accesses backed by a constant base need
                    // to be done through a proxy local variable, since constants have a non
                    // pointer type which is required for dynamic indexing
                    if !constant_index {
                        if let Some((constant, ty)) = var.constant {
                            let local =
                                self.locals.append(
                                    LocalVariable {
                                        name: None,
                                        ty,
                                        init: Some(frontend.module.const_expressions.append(
                                            Expression::Constant(constant),
                                            Span::default(),
                                        )),
                                    },
                                    Span::default(),
                                );

                            self.add_expression(
                                Expression::LocalVariable(local),
                                Span::default(),
                                body,
                            )
                        } else {
                            var.expr
                        }
                    } else {
                        var.expr
                    }
                }
                _ if var.load => {
                    self.add_expression(Expression::Load { pointer: var.expr }, meta, body)
                }
                ExprPos::Rhs => var.expr,
            },
            HirExprKind::Call(ref call) if pos != ExprPos::Lhs => {
                let maybe_expr = frontend.function_or_constructor_call(
                    self,
                    stmt,
                    body,
                    call.kind.clone(),
                    &call.args,
                    meta,
                )?;
                return Ok((maybe_expr, meta));
            }
            // `HirExprKind::Conditional` represents the ternary operator in glsl (`:?`)
            //
            // The ternary operator is defined to only evaluate one of the two possible
            // expressions which means that it's behavior is that of an `if` statement,
            // and it's merely syntatic sugar for it.
            HirExprKind::Conditional {
                condition,
                accept,
                reject,
            } if ExprPos::Lhs != pos => {
                // Given an expression `a ? b : c`, we need to produce a Naga
                // statement roughly like:
                //
                //     var temp;
                //     if a {
                //         temp = convert(b);
                //     } else  {
                //         temp = convert(c);
                //     }
                //
                // where `convert` stands for type conversions to bring `b` and `c` to
                // the same type, and then use `temp` to represent the value of the whole
                // conditional expression in subsequent code.

                // Lower the condition first to the current bodyy
                let condition = self
                    .lower_expect_inner(stmt, frontend, condition, ExprPos::Rhs, body)?
                    .0;

                // Emit all expressions since we will be adding statements to
                // other bodies next
                self.emit_restart(body);

                // Create the bodies for the two cases
                let mut accept_body = Block::new();
                let mut reject_body = Block::new();

                // Lower the `true` branch
                let (mut accept, accept_meta) =
                    self.lower_expect_inner(stmt, frontend, accept, pos, &mut accept_body)?;

                // Flush the body of the `true` branch, to start emitting on the
                // `false` branch
                self.emit_restart(&mut accept_body);

                // Lower the `false` branch
                let (mut reject, reject_meta) =
                    self.lower_expect_inner(stmt, frontend, reject, pos, &mut reject_body)?;

                // Flush the body of the `false` branch
                self.emit_restart(&mut reject_body);

                // We need to do some custom implicit conversions since the two target expressions
                // are in different bodies
                if let (
                    Some((accept_power, accept_width, accept_kind)),
                    Some((reject_power, reject_width, reject_kind)),
                ) = (
                    // Get the components of both branches and calculate the type power
                    self.expr_scalar_components(frontend, accept, accept_meta)?
                        .and_then(|(kind, width)| Some((type_power(kind, width)?, width, kind))),
                    self.expr_scalar_components(frontend, reject, reject_meta)?
                        .and_then(|(kind, width)| Some((type_power(kind, width)?, width, kind))),
                ) {
                    match accept_power.cmp(&reject_power) {
                        std::cmp::Ordering::Less => {
                            self.conversion(
                                &mut accept,
                                accept_meta,
                                reject_kind,
                                reject_width,
                                &mut accept_body,
                            )?;
                            // The expression belongs to the `true` branch so we need to flush to
                            // the respective body
                            self.emit_restart(&mut accept_body);
                        }
                        std::cmp::Ordering::Equal => {}
                        std::cmp::Ordering::Greater => {
                            self.conversion(
                                &mut reject,
                                reject_meta,
                                accept_kind,
                                accept_width,
                                &mut reject_body,
                            )?;
                            // The expression belongs to the `false` branch so we need to flush to
                            // the respective body
                            self.emit_restart(&mut reject_body);
                        }
                    }
                }

                // We need to get the type of the resulting expression to create the local,
                // this must be done after implicit conversions to ensure both branches have
                // the same type.
                let ty = frontend.resolve_type_handle(self, accept, accept_meta)?;

                // Add the local that will hold the result of our conditional
                let local = self.locals.append(
                    LocalVariable {
                        name: None,
                        ty,
                        init: None,
                    },
                    meta,
                );

                let local_expr = self.add_expression(Expression::LocalVariable(local), meta, body);

                // Add to each body the store to the result variable
                accept_body.push(
                    Statement::Store {
                        pointer: local_expr,
                        value: accept,
                    },
                    accept_meta,
                );
                reject_body.push(
                    Statement::Store {
                        pointer: local_expr,
                        value: reject,
                    },
                    reject_meta,
                );

                // Finally add the `If` to the main body with the `condition` we lowered
                // earlier and the branches we prepared.
                body.push(
                    Statement::If {
                        condition,
                        accept: accept_body,
                        reject: reject_body,
                    },
                    meta,
                );

                // Note: `Expression::Load` must be emited before it's used so make
                // sure the emitter is active here.
                self.add_expression(
                    Expression::Load {
                        pointer: local_expr,
                    },
                    meta,
                    body,
                )
            }
            HirExprKind::Assign { tgt, value } if ExprPos::Lhs != pos => {
                let (pointer, ptr_meta) =
                    self.lower_expect_inner(stmt, frontend, tgt, ExprPos::Lhs, body)?;
                let (mut value, value_meta) =
                    self.lower_expect_inner(stmt, frontend, value, ExprPos::Rhs, body)?;

                let ty = match *frontend.resolve_type(self, pointer, ptr_meta)? {
                    TypeInner::Pointer { base, .. } => &frontend.module.types[base].inner,
                    ref ty => ty,
                };

                if let Some((kind, width)) = scalar_components(ty) {
                    self.implicit_conversion(frontend, &mut value, value_meta, kind, width, body)?;
                }

                self.lower_store(pointer, value, meta, body);

                value
            }
            HirExprKind::PrePostfix { op, postfix, expr } if ExprPos::Lhs != pos => {
                let (pointer, _) =
                    self.lower_expect_inner(stmt, frontend, expr, ExprPos::Lhs, body)?;
                let left = if let Expression::Swizzle { .. } = self.expressions[pointer] {
                    pointer
                } else {
                    self.add_expression(Expression::Load { pointer }, meta, body)
                };

                let res = match *frontend.resolve_type(self, left, meta)? {
                    TypeInner::Scalar { kind, width } => {
                        let ty = TypeInner::Scalar { kind, width };
                        Literal::one(kind, width).map(|i| (ty, i, None, None))
                    }
                    TypeInner::Vector { size, kind, width } => {
                        let ty = TypeInner::Vector { size, kind, width };
                        Literal::one(kind, width).map(|i| (ty, i, Some(size), None))
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
                        Literal::one(ScalarKind::Float, width)
                            .map(|i| (ty, i, Some(rows), Some(columns)))
                    }
                    _ => None,
                };
                let (ty_inner, literal, rows, columns) = match res {
                    Some(res) => res,
                    None => {
                        frontend.errors.push(Error {
                            kind: ErrorKind::SemanticError(
                                "Increment/decrement only works on scalar/vector/matrix".into(),
                            ),
                            meta,
                        });
                        return Ok((Some(left), meta));
                    }
                };

                let mut right = self.add_expression(Expression::Literal(literal), meta, body);

                // Glsl allows pre/postfixes operations on vectors and matrices, so if the
                // target is either of them change the right side of the addition to be splatted
                // to the same size as the target, furthermore if the target is a matrix
                // use a composed matrix using the splatted value.
                if let Some(size) = rows {
                    right =
                        self.add_expression(Expression::Splat { size, value: right }, meta, body);

                    if let Some(cols) = columns {
                        let ty = frontend.module.types.insert(
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

                self.lower_store(pointer, value, meta, body);

                if postfix {
                    left
                } else {
                    value
                }
            }
            HirExprKind::Method {
                expr: object,
                ref name,
                ref args,
            } if ExprPos::Lhs != pos => {
                let args = args
                    .iter()
                    .map(|e| self.lower_expect_inner(stmt, frontend, *e, ExprPos::Rhs, body))
                    .collect::<Result<Vec<_>>>()?;
                match name.as_ref() {
                    "length" => {
                        if !args.is_empty() {
                            frontend.errors.push(Error {
                                kind: ErrorKind::SemanticError(
                                    ".length() doesn't take any arguments".into(),
                                ),
                                meta,
                            });
                        }
                        let lowered_array = self
                            .lower_expect_inner(stmt, frontend, object, pos, body)?
                            .0;
                        let array_type = frontend.resolve_type(self, lowered_array, meta)?;

                        match *array_type {
                            TypeInner::Array {
                                size: crate::ArraySize::Constant(size),
                                ..
                            } => {
                                let mut array_length = self.add_expression(
                                    Expression::Literal(Literal::U32(size.get())),
                                    meta,
                                    body,
                                );
                                self.forced_conversion(
                                    frontend,
                                    &mut array_length,
                                    meta,
                                    ScalarKind::Sint,
                                    4,
                                    body,
                                )?;
                                array_length
                            }
                            // let the error be handled in type checking if it's not a dynamic array
                            _ => {
                                let mut array_length = self.add_expression(
                                    Expression::ArrayLength(lowered_array),
                                    meta,
                                    body,
                                );
                                self.conversion(
                                    &mut array_length,
                                    meta,
                                    ScalarKind::Sint,
                                    4,
                                    body,
                                )?;
                                array_length
                            }
                        }
                    }
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::SemanticError(
                                format!("unknown method '{name}'").into(),
                            ),
                            meta,
                        });
                    }
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

        log::trace!(
            "Lowered {:?}\n\tKind = {:?}\n\tPos = {:?}\n\tResult = {:?}",
            expr,
            kind,
            pos,
            handle
        );

        Ok((Some(handle), meta))
    }

    pub fn expr_scalar_components(
        &mut self,
        frontend: &Frontend,
        expr: Handle<Expression>,
        meta: Span,
    ) -> Result<Option<(ScalarKind, crate::Bytes)>> {
        let ty = frontend.resolve_type(self, expr, meta)?;
        Ok(scalar_components(ty))
    }

    pub fn expr_power(
        &mut self,
        frontend: &Frontend,
        expr: Handle<Expression>,
        meta: Span,
    ) -> Result<Option<u32>> {
        Ok(self
            .expr_scalar_components(frontend, expr, meta)?
            .and_then(|(kind, width)| type_power(kind, width)))
    }

    pub fn conversion(
        &mut self,
        expr: &mut Handle<Expression>,
        meta: Span,
        kind: ScalarKind,
        width: crate::Bytes,
        body: &mut Block,
    ) -> Result<()> {
        *expr = self.add_expression(
            Expression::As {
                expr: *expr,
                kind,
                convert: Some(width),
            },
            meta,
            body,
        );

        Ok(())
    }

    pub fn implicit_conversion(
        &mut self,
        frontend: &Frontend,
        expr: &mut Handle<Expression>,
        meta: Span,
        kind: ScalarKind,
        width: crate::Bytes,
        body: &mut Block,
    ) -> Result<()> {
        if let (Some(tgt_power), Some(expr_power)) = (
            type_power(kind, width),
            self.expr_power(frontend, *expr, meta)?,
        ) {
            if tgt_power > expr_power {
                self.conversion(expr, meta, kind, width, body)?;
            }
        }

        Ok(())
    }

    pub fn forced_conversion(
        &mut self,
        frontend: &Frontend,
        expr: &mut Handle<Expression>,
        meta: Span,
        kind: ScalarKind,
        width: crate::Bytes,
        body: &mut Block,
    ) -> Result<()> {
        if let Some((expr_scalar_kind, expr_width)) =
            self.expr_scalar_components(frontend, *expr, meta)?
        {
            if expr_scalar_kind != kind || expr_width != width {
                self.conversion(expr, meta, kind, width, body)?;
            }
        }

        Ok(())
    }

    pub fn binary_implicit_conversion(
        &mut self,
        frontend: &Frontend,
        left: &mut Handle<Expression>,
        left_meta: Span,
        right: &mut Handle<Expression>,
        right_meta: Span,
        body: &mut Block,
    ) -> Result<()> {
        let left_components = self.expr_scalar_components(frontend, *left, left_meta)?;
        let right_components = self.expr_scalar_components(frontend, *right, right_meta)?;

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
                    self.conversion(left, left_meta, right_kind, right_width, body)?;
                }
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => {
                    self.conversion(right, right_meta, left_kind, left_width, body)?;
                }
            }
        }

        Ok(())
    }

    pub fn implicit_splat(
        &mut self,
        frontend: &Frontend,
        expr: &mut Handle<Expression>,
        meta: Span,
        vector_size: Option<VectorSize>,
        body: &mut Block,
    ) -> Result<()> {
        let expr_type = frontend.resolve_type(self, *expr, meta)?;

        if let (&TypeInner::Scalar { .. }, Some(size)) = (expr_type, vector_size) {
            *expr = self.add_expression(Expression::Splat { size, value: *expr }, meta, body)
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
/// This struct should only be obtained through [`stmt_ctx`](Context::stmt_ctx)
/// and only one of these may be active at any time per context.
#[derive(Debug)]
pub struct StmtContext {
    /// A arena of high level expressions which can be lowered through a
    /// [`Context`](Context) to naga's [`Expression`](crate::Expression)s
    pub hir_exprs: Arena<HirExpr>,
}

impl StmtContext {
    const fn new() -> Self {
        StmtContext {
            hir_exprs: Arena::new(),
        }
    }
}
