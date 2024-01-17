use crate::{
    arena::{Arena, Handle, UniqueArena},
    ArraySize, BinaryOperator, Constant, Expression, Literal, ScalarKind, Span, Type, TypeInner,
    UnaryOperator,
};

#[derive(Debug)]
enum Behavior {
    Wgsl,
    Glsl,
}

/// A context for evaluating constant expressions.
///
/// A `ConstantEvaluator` points at an expression arena to which it can append
/// newly evaluated expressions: you pass [`try_eval_and_append`] whatever kind
/// of Naga [`Expression`] you like, and if its value can be computed at compile
/// time, `try_eval_and_append` appends an expression representing the computed
/// value - a tree of [`Literal`], [`Compose`], [`ZeroValue`], and [`Swizzle`]
/// expressions - to the arena. See the [`try_eval_and_append`] method for details.
///
/// A `ConstantEvaluator` also holds whatever information we need to carry out
/// that evaluation: types, other constants, and so on.
///
/// [`try_eval_and_append`]: ConstantEvaluator::try_eval_and_append
/// [`Compose`]: Expression::Compose
/// [`ZeroValue`]: Expression::ZeroValue
/// [`Literal`]: Expression::Literal
/// [`Swizzle`]: Expression::Swizzle
#[derive(Debug)]
pub struct ConstantEvaluator<'a> {
    /// Which language's evaluation rules we should follow.
    behavior: Behavior,

    /// The module's type arena.
    ///
    /// Because expressions like [`Splat`] contain type handles, we need to be
    /// able to add new types to produce those expressions.
    ///
    /// [`Splat`]: Expression::Splat
    types: &'a mut UniqueArena<Type>,

    /// The module's constant arena.
    constants: &'a Arena<Constant>,

    /// The arena to which we are contributing expressions.
    expressions: &'a mut Arena<Expression>,

    /// When `self.expressions` refers to a function's local expression
    /// arena, this needs to be populated
    function_local_data: Option<FunctionLocalData<'a>>,
}

#[derive(Debug)]
struct FunctionLocalData<'a> {
    /// Global constant expressions
    const_expressions: &'a Arena<Expression>,
    /// Tracks the constness of expressions residing in `ConstantEvaluator.expressions`
    expression_constness: &'a mut ExpressionConstnessTracker,
    emitter: &'a mut super::Emitter,
    block: &'a mut crate::Block,
}

#[derive(Debug)]
pub struct ExpressionConstnessTracker {
    inner: bit_set::BitSet,
}

impl ExpressionConstnessTracker {
    pub fn new() -> Self {
        Self {
            inner: bit_set::BitSet::new(),
        }
    }

    /// Forces the the expression to not be const
    pub fn force_non_const(&mut self, value: Handle<Expression>) {
        self.inner.remove(value.index());
    }

    fn insert(&mut self, value: Handle<Expression>) {
        self.inner.insert(value.index());
    }

    pub fn is_const(&self, value: Handle<Expression>) -> bool {
        self.inner.contains(value.index())
    }

    pub fn from_arena(arena: &Arena<Expression>) -> Self {
        let mut tracker = Self::new();
        for (handle, expr) in arena.iter() {
            let insert = match *expr {
                crate::Expression::Literal(_)
                | crate::Expression::ZeroValue(_)
                | crate::Expression::Constant(_) => true,
                crate::Expression::Compose { ref components, .. } => {
                    components.iter().all(|h| tracker.is_const(*h))
                }
                crate::Expression::Splat { value, .. } => tracker.is_const(value),
                _ => false,
            };
            if insert {
                tracker.insert(handle);
            }
        }
        tracker
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum ConstantEvaluatorError {
    #[error("Constants cannot access function arguments")]
    FunctionArg,
    #[error("Constants cannot access global variables")]
    GlobalVariable,
    #[error("Constants cannot access local variables")]
    LocalVariable,
    #[error("Cannot get the array length of a non array type")]
    InvalidArrayLengthArg,
    #[error("Constants cannot get the array length of a dynamically sized array")]
    ArrayLengthDynamic,
    #[error("Constants cannot call functions")]
    Call,
    #[error("Constants don't support workGroupUniformLoad")]
    WorkGroupUniformLoadResult,
    #[error("Constants don't support atomic functions")]
    Atomic,
    #[error("Constants don't support derivative functions")]
    Derivative,
    #[error("Constants don't support load expressions")]
    Load,
    #[error("Constants don't support image expressions")]
    ImageExpression,
    #[error("Constants don't support ray query expressions")]
    RayQueryExpression,
    #[error("Cannot access the type")]
    InvalidAccessBase,
    #[error("Cannot access at the index")]
    InvalidAccessIndex,
    #[error("Cannot access with index of type")]
    InvalidAccessIndexTy,
    #[error("Constants don't support array length expressions")]
    ArrayLength,
    #[error("Cannot cast scalar components of expression `{from}` to type `{to}`")]
    InvalidCastArg { from: String, to: String },
    #[error("Cannot apply the unary op to the argument")]
    InvalidUnaryOpArg,
    #[error("Cannot apply the binary op to the arguments")]
    InvalidBinaryOpArgs,
    #[error("Cannot apply math function to type")]
    InvalidMathArg,
    #[error("{0:?} built-in function expects {1:?} arguments but {2:?} were supplied")]
    InvalidMathArgCount(crate::MathFunction, usize, usize),
    #[error("value of `low` is greater than `high` for clamp built-in function")]
    InvalidClamp,
    #[error("Splat is defined only on scalar values")]
    SplatScalarOnly,
    #[error("Can only swizzle vector constants")]
    SwizzleVectorOnly,
    #[error("swizzle component not present in source expression")]
    SwizzleOutOfBounds,
    #[error("Type is not constructible")]
    TypeNotConstructible,
    #[error("Subexpression(s) are not constant")]
    SubexpressionsAreNotConstant,
    #[error("Not implemented as constant expression: {0}")]
    NotImplemented(String),
    #[error("{0} operation overflowed")]
    Overflow(String),
    #[error(
        "the concrete type `{to_type}` cannot represent the abstract value `{value}` accurately"
    )]
    AutomaticConversionLossy {
        value: String,
        to_type: &'static str,
    },
    #[error("abstract floating-point values cannot be automatically converted to integers")]
    AutomaticConversionFloatToInt { to_type: &'static str },
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Remainder by zero")]
    RemainderByZero,
    #[error("RHS of shift operation is greater than or equal to 32")]
    ShiftedMoreThan32Bits,
    #[error(transparent)]
    Literal(#[from] crate::valid::LiteralError),
}

impl<'a> ConstantEvaluator<'a> {
    /// Return a [`ConstantEvaluator`] that will add expressions to `module`'s
    /// constant expression arena.
    ///
    /// Report errors according to WGSL's rules for constant evaluation.
    pub fn for_wgsl_module(module: &'a mut crate::Module) -> Self {
        Self::for_module(Behavior::Wgsl, module)
    }

    /// Return a [`ConstantEvaluator`] that will add expressions to `module`'s
    /// constant expression arena.
    ///
    /// Report errors according to GLSL's rules for constant evaluation.
    pub fn for_glsl_module(module: &'a mut crate::Module) -> Self {
        Self::for_module(Behavior::Glsl, module)
    }

    fn for_module(behavior: Behavior, module: &'a mut crate::Module) -> Self {
        Self {
            behavior,
            types: &mut module.types,
            constants: &module.constants,
            expressions: &mut module.const_expressions,
            function_local_data: None,
        }
    }

    /// Return a [`ConstantEvaluator`] that will add expressions to `function`'s
    /// expression arena.
    ///
    /// Report errors according to WGSL's rules for constant evaluation.
    pub fn for_wgsl_function(
        module: &'a mut crate::Module,
        expressions: &'a mut Arena<Expression>,
        expression_constness: &'a mut ExpressionConstnessTracker,
        emitter: &'a mut super::Emitter,
        block: &'a mut crate::Block,
    ) -> Self {
        Self::for_function(
            Behavior::Wgsl,
            module,
            expressions,
            expression_constness,
            emitter,
            block,
        )
    }

    /// Return a [`ConstantEvaluator`] that will add expressions to `function`'s
    /// expression arena.
    ///
    /// Report errors according to GLSL's rules for constant evaluation.
    pub fn for_glsl_function(
        module: &'a mut crate::Module,
        expressions: &'a mut Arena<Expression>,
        expression_constness: &'a mut ExpressionConstnessTracker,
        emitter: &'a mut super::Emitter,
        block: &'a mut crate::Block,
    ) -> Self {
        Self::for_function(
            Behavior::Glsl,
            module,
            expressions,
            expression_constness,
            emitter,
            block,
        )
    }

    fn for_function(
        behavior: Behavior,
        module: &'a mut crate::Module,
        expressions: &'a mut Arena<Expression>,
        expression_constness: &'a mut ExpressionConstnessTracker,
        emitter: &'a mut super::Emitter,
        block: &'a mut crate::Block,
    ) -> Self {
        Self {
            behavior,
            types: &mut module.types,
            constants: &module.constants,
            expressions,
            function_local_data: Some(FunctionLocalData {
                const_expressions: &module.const_expressions,
                expression_constness,
                emitter,
                block,
            }),
        }
    }

    pub fn to_ctx(&self) -> crate::proc::GlobalCtx {
        crate::proc::GlobalCtx {
            types: self.types,
            constants: self.constants,
            const_expressions: match self.function_local_data {
                Some(ref data) => data.const_expressions,
                None => self.expressions,
            },
        }
    }

    fn check(&self, expr: Handle<Expression>) -> Result<(), ConstantEvaluatorError> {
        if let Some(ref function_local_data) = self.function_local_data {
            if !function_local_data.expression_constness.is_const(expr) {
                log::debug!("check: SubexpressionsAreNotConstant");
                return Err(ConstantEvaluatorError::SubexpressionsAreNotConstant);
            }
        }
        Ok(())
    }

    fn check_and_get(
        &mut self,
        expr: Handle<Expression>,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.expressions[expr] {
            Expression::Constant(c) => {
                // Are we working in a function's expression arena, or the
                // module's constant expression arena?
                if let Some(ref function_local_data) = self.function_local_data {
                    // Deep-copy the constant's value into our arena.
                    self.copy_from(
                        self.constants[c].init,
                        function_local_data.const_expressions,
                    )
                } else {
                    // "See through" the constant and use its initializer.
                    Ok(self.constants[c].init)
                }
            }
            _ => {
                self.check(expr)?;
                Ok(expr)
            }
        }
    }

    /// Try to evaluate `expr` at compile time.
    ///
    /// The `expr` argument can be any sort of Naga [`Expression`] you like. If
    /// we can determine its value at compile time, we append an expression
    /// representing its value - a tree of [`Literal`], [`Compose`],
    /// [`ZeroValue`], and [`Swizzle`] expressions - to the expression arena
    /// `self` contributes to.
    ///
    /// If `expr`'s value cannot be determined at compile time, return a an
    /// error. If it's acceptable to evaluate `expr` at runtime, this error can
    /// be ignored, and the caller can append `expr` to the arena itself.
    ///
    /// We only consider `expr` itself, without recursing into its operands. Its
    /// operands must all have been produced by prior calls to
    /// `try_eval_and_append`, to ensure that they have already been reduced to
    /// an evaluated form if possible.
    ///
    /// [`Literal`]: Expression::Literal
    /// [`Compose`]: Expression::Compose
    /// [`ZeroValue`]: Expression::ZeroValue
    /// [`Swizzle`]: Expression::Swizzle
    pub fn try_eval_and_append(
        &mut self,
        expr: &Expression,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        log::trace!("try_eval_and_append: {:?}", expr);
        match *expr {
            Expression::Constant(c) if self.function_local_data.is_none() => {
                // "See through" the constant and use its initializer.
                // This is mainly done to avoid having constants pointing to other constants.
                Ok(self.constants[c].init)
            }
            Expression::Literal(_) | Expression::ZeroValue(_) | Expression::Constant(_) => {
                self.register_evaluated_expr(expr.clone(), span)
            }
            Expression::Compose { ty, ref components } => {
                let components = components
                    .iter()
                    .map(|component| self.check_and_get(*component))
                    .collect::<Result<Vec<_>, _>>()?;
                self.register_evaluated_expr(Expression::Compose { ty, components }, span)
            }
            Expression::Splat { size, value } => {
                let value = self.check_and_get(value)?;
                self.register_evaluated_expr(Expression::Splat { size, value }, span)
            }
            Expression::AccessIndex { base, index } => {
                let base = self.check_and_get(base)?;

                self.access(base, index as usize, span)
            }
            Expression::Access { base, index } => {
                let base = self.check_and_get(base)?;
                let index = self.check_and_get(index)?;

                self.access(base, self.constant_index(index)?, span)
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let vector = self.check_and_get(vector)?;

                self.swizzle(size, span, vector, pattern)
            }
            Expression::Unary { expr, op } => {
                let expr = self.check_and_get(expr)?;

                self.unary_op(op, expr, span)
            }
            Expression::Binary { left, right, op } => {
                let left = self.check_and_get(left)?;
                let right = self.check_and_get(right)?;

                self.binary_op(op, left, right, span)
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                let arg = self.check_and_get(arg)?;
                let arg1 = arg1.map(|arg| self.check_and_get(arg)).transpose()?;
                let arg2 = arg2.map(|arg| self.check_and_get(arg)).transpose()?;
                let arg3 = arg3.map(|arg| self.check_and_get(arg)).transpose()?;

                self.math(arg, arg1, arg2, arg3, fun, span)
            }
            Expression::As {
                convert,
                expr,
                kind,
            } => {
                let expr = self.check_and_get(expr)?;

                match convert {
                    Some(width) => self.cast(expr, crate::Scalar { kind, width }, span),
                    None => Err(ConstantEvaluatorError::NotImplemented(
                        "bitcast built-in function".into(),
                    )),
                }
            }
            Expression::Select { .. } => Err(ConstantEvaluatorError::NotImplemented(
                "select built-in function".into(),
            )),
            Expression::Relational { fun, .. } => Err(ConstantEvaluatorError::NotImplemented(
                format!("{fun:?} built-in function"),
            )),
            Expression::ArrayLength(expr) => match self.behavior {
                Behavior::Wgsl => Err(ConstantEvaluatorError::ArrayLength),
                Behavior::Glsl => {
                    let expr = self.check_and_get(expr)?;
                    self.array_length(expr, span)
                }
            },
            Expression::Load { .. } => Err(ConstantEvaluatorError::Load),
            Expression::LocalVariable(_) => Err(ConstantEvaluatorError::LocalVariable),
            Expression::Derivative { .. } => Err(ConstantEvaluatorError::Derivative),
            Expression::CallResult { .. } => Err(ConstantEvaluatorError::Call),
            Expression::WorkGroupUniformLoadResult { .. } => {
                Err(ConstantEvaluatorError::WorkGroupUniformLoadResult)
            }
            Expression::AtomicResult { .. } => Err(ConstantEvaluatorError::Atomic),
            Expression::FunctionArgument(_) => Err(ConstantEvaluatorError::FunctionArg),
            Expression::GlobalVariable(_) => Err(ConstantEvaluatorError::GlobalVariable),
            Expression::ImageSample { .. }
            | Expression::ImageLoad { .. }
            | Expression::ImageQuery { .. } => Err(ConstantEvaluatorError::ImageExpression),
            Expression::RayQueryProceedResult | Expression::RayQueryGetIntersection { .. } => {
                Err(ConstantEvaluatorError::RayQueryExpression)
            }
        }
    }

    /// Splat `value` to `size`, without using [`Splat`] expressions.
    ///
    /// This constructs [`Compose`] or [`ZeroValue`] expressions to
    /// build a vector with the given `size` whose components are all
    /// `value`.
    ///
    /// Use `span` as the span of the inserted expressions and
    /// resulting types.
    ///
    /// [`Splat`]: Expression::Splat
    /// [`Compose`]: Expression::Compose
    /// [`ZeroValue`]: Expression::ZeroValue
    fn splat(
        &mut self,
        value: Handle<Expression>,
        size: crate::VectorSize,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.expressions[value] {
            Expression::Literal(literal) => {
                let scalar = literal.scalar();
                let ty = self.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Vector { size, scalar },
                    },
                    span,
                );
                let expr = Expression::Compose {
                    ty,
                    components: vec![value; size as usize],
                };
                self.register_evaluated_expr(expr, span)
            }
            Expression::ZeroValue(ty) => {
                let inner = match self.types[ty].inner {
                    TypeInner::Scalar(scalar) => TypeInner::Vector { size, scalar },
                    _ => return Err(ConstantEvaluatorError::SplatScalarOnly),
                };
                let res_ty = self.types.insert(Type { name: None, inner }, span);
                let expr = Expression::ZeroValue(res_ty);
                self.register_evaluated_expr(expr, span)
            }
            _ => Err(ConstantEvaluatorError::SplatScalarOnly),
        }
    }

    fn swizzle(
        &mut self,
        size: crate::VectorSize,
        span: Span,
        src_constant: Handle<Expression>,
        pattern: [crate::SwizzleComponent; 4],
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let mut get_dst_ty = |ty| match self.types[ty].inner {
            crate::TypeInner::Vector { size: _, scalar } => Ok(self.types.insert(
                Type {
                    name: None,
                    inner: crate::TypeInner::Vector { size, scalar },
                },
                span,
            )),
            _ => Err(ConstantEvaluatorError::SwizzleVectorOnly),
        };

        match self.expressions[src_constant] {
            Expression::ZeroValue(ty) => {
                let dst_ty = get_dst_ty(ty)?;
                let expr = Expression::ZeroValue(dst_ty);
                self.register_evaluated_expr(expr, span)
            }
            Expression::Splat { value, .. } => {
                let expr = Expression::Splat { size, value };
                self.register_evaluated_expr(expr, span)
            }
            Expression::Compose { ty, ref components } => {
                let dst_ty = get_dst_ty(ty)?;

                let mut flattened = [src_constant; 4]; // dummy value
                let len =
                    crate::proc::flatten_compose(ty, components, self.expressions, self.types)
                        .zip(flattened.iter_mut())
                        .map(|(component, elt)| *elt = component)
                        .count();
                let flattened = &flattened[..len];

                let swizzled_components = pattern[..size as usize]
                    .iter()
                    .map(|&sc| {
                        let sc = sc as usize;
                        if let Some(elt) = flattened.get(sc) {
                            Ok(*elt)
                        } else {
                            Err(ConstantEvaluatorError::SwizzleOutOfBounds)
                        }
                    })
                    .collect::<Result<Vec<Handle<Expression>>, _>>()?;
                let expr = Expression::Compose {
                    ty: dst_ty,
                    components: swizzled_components,
                };
                self.register_evaluated_expr(expr, span)
            }
            _ => Err(ConstantEvaluatorError::SwizzleVectorOnly),
        }
    }

    fn math(
        &mut self,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        arg2: Option<Handle<Expression>>,
        arg3: Option<Handle<Expression>>,
        fun: crate::MathFunction,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let expected = fun.argument_count();
        let given = Some(arg)
            .into_iter()
            .chain(arg1)
            .chain(arg2)
            .chain(arg3)
            .count();
        if expected != given {
            return Err(ConstantEvaluatorError::InvalidMathArgCount(
                fun, expected, given,
            ));
        }

        match fun {
            crate::MathFunction::Pow => self.math_pow(arg, arg1.unwrap(), span),
            crate::MathFunction::Clamp => self.math_clamp(arg, arg1.unwrap(), arg2.unwrap(), span),
            fun => Err(ConstantEvaluatorError::NotImplemented(format!(
                "{fun:?} built-in function"
            ))),
        }
    }

    fn math_pow(
        &mut self,
        e1: Handle<Expression>,
        e2: Handle<Expression>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let e1 = self.eval_zero_value_and_splat(e1, span)?;
        let e2 = self.eval_zero_value_and_splat(e2, span)?;

        let expr = match (&self.expressions[e1], &self.expressions[e2]) {
            (&Expression::Literal(Literal::F32(a)), &Expression::Literal(Literal::F32(b))) => {
                Expression::Literal(Literal::F32(a.powf(b)))
            }
            (
                &Expression::Compose {
                    components: ref src_components0,
                    ty: ty0,
                },
                &Expression::Compose {
                    components: ref src_components1,
                    ty: ty1,
                },
            ) if ty0 == ty1
                && matches!(
                    self.types[ty0].inner,
                    crate::TypeInner::Vector {
                        scalar: crate::Scalar {
                            kind: ScalarKind::Float,
                            ..
                        },
                        ..
                    }
                ) =>
            {
                let mut components: Vec<_> = crate::proc::flatten_compose(
                    ty0,
                    src_components0,
                    self.expressions,
                    self.types,
                )
                .chain(crate::proc::flatten_compose(
                    ty1,
                    src_components1,
                    self.expressions,
                    self.types,
                ))
                .collect();

                let mid = components.len() / 2;
                let (first, last) = components.split_at_mut(mid);
                for (a, b) in first.iter_mut().zip(&*last) {
                    *a = self.math_pow(*a, *b, span)?;
                }
                components.truncate(mid);

                Expression::Compose {
                    ty: ty0,
                    components,
                }
            }
            _ => return Err(ConstantEvaluatorError::InvalidMathArg),
        };

        self.register_evaluated_expr(expr, span)
    }

    fn math_clamp(
        &mut self,
        e: Handle<Expression>,
        low: Handle<Expression>,
        high: Handle<Expression>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let e = self.eval_zero_value_and_splat(e, span)?;
        let low = self.eval_zero_value_and_splat(low, span)?;
        let high = self.eval_zero_value_and_splat(high, span)?;

        let expr = match (
            &self.expressions[e],
            &self.expressions[low],
            &self.expressions[high],
        ) {
            (&Expression::Literal(e), &Expression::Literal(low), &Expression::Literal(high)) => {
                let literal = match (e, low, high) {
                    (Literal::I32(e), Literal::I32(low), Literal::I32(high)) => {
                        if low > high {
                            return Err(ConstantEvaluatorError::InvalidClamp);
                        } else {
                            Literal::I32(e.clamp(low, high))
                        }
                    }
                    (Literal::U32(e), Literal::U32(low), Literal::U32(high)) => {
                        if low > high {
                            return Err(ConstantEvaluatorError::InvalidClamp);
                        } else {
                            Literal::U32(e.clamp(low, high))
                        }
                    }
                    (Literal::F32(e), Literal::F32(low), Literal::F32(high)) => {
                        if low > high {
                            return Err(ConstantEvaluatorError::InvalidClamp);
                        } else {
                            Literal::F32(e.clamp(low, high))
                        }
                    }
                    _ => return Err(ConstantEvaluatorError::InvalidMathArg),
                };
                Expression::Literal(literal)
            }
            (
                &Expression::Compose {
                    components: ref src_components0,
                    ty: ty0,
                },
                &Expression::Compose {
                    components: ref src_components1,
                    ty: ty1,
                },
                &Expression::Compose {
                    components: ref src_components2,
                    ty: ty2,
                },
            ) if ty0 == ty1
                && ty0 == ty2
                && matches!(
                    self.types[ty0].inner,
                    crate::TypeInner::Vector {
                        scalar: crate::Scalar {
                            kind: ScalarKind::Float,
                            ..
                        },
                        ..
                    }
                ) =>
            {
                let mut components: Vec<_> = crate::proc::flatten_compose(
                    ty0,
                    src_components0,
                    self.expressions,
                    self.types,
                )
                .chain(crate::proc::flatten_compose(
                    ty1,
                    src_components1,
                    self.expressions,
                    self.types,
                ))
                .chain(crate::proc::flatten_compose(
                    ty2,
                    src_components2,
                    self.expressions,
                    self.types,
                ))
                .collect();

                let chunk_size = components.len() / 3;
                let (es, rem) = components.split_at_mut(chunk_size);
                let (lows, highs) = rem.split_at(chunk_size);
                for ((e, low), high) in es.iter_mut().zip(lows).zip(highs) {
                    *e = self.math_clamp(*e, *low, *high, span)?;
                }
                components.truncate(chunk_size);

                Expression::Compose {
                    ty: ty0,
                    components,
                }
            }
            _ => return Err(ConstantEvaluatorError::InvalidMathArg),
        };

        self.register_evaluated_expr(expr, span)
    }

    fn array_length(
        &mut self,
        array: Handle<Expression>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.expressions[array] {
            Expression::ZeroValue(ty) | Expression::Compose { ty, .. } => {
                match self.types[ty].inner {
                    TypeInner::Array { size, .. } => match size {
                        crate::ArraySize::Constant(len) => {
                            let expr = Expression::Literal(Literal::U32(len.get()));
                            self.register_evaluated_expr(expr, span)
                        }
                        crate::ArraySize::Dynamic => {
                            Err(ConstantEvaluatorError::ArrayLengthDynamic)
                        }
                    },
                    _ => Err(ConstantEvaluatorError::InvalidArrayLengthArg),
                }
            }
            _ => Err(ConstantEvaluatorError::InvalidArrayLengthArg),
        }
    }

    fn access(
        &mut self,
        base: Handle<Expression>,
        index: usize,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.expressions[base] {
            Expression::ZeroValue(ty) => {
                let ty_inner = &self.types[ty].inner;
                let components = ty_inner
                    .components()
                    .ok_or(ConstantEvaluatorError::InvalidAccessBase)?;

                if index >= components as usize {
                    Err(ConstantEvaluatorError::InvalidAccessBase)
                } else {
                    let ty_res = ty_inner
                        .component_type(index)
                        .ok_or(ConstantEvaluatorError::InvalidAccessIndex)?;
                    let ty = match ty_res {
                        crate::proc::TypeResolution::Handle(ty) => ty,
                        crate::proc::TypeResolution::Value(inner) => {
                            self.types.insert(Type { name: None, inner }, span)
                        }
                    };
                    self.register_evaluated_expr(Expression::ZeroValue(ty), span)
                }
            }
            Expression::Splat { size, value } => {
                if index >= size as usize {
                    Err(ConstantEvaluatorError::InvalidAccessBase)
                } else {
                    Ok(value)
                }
            }
            Expression::Compose { ty, ref components } => {
                let _ = self.types[ty]
                    .inner
                    .components()
                    .ok_or(ConstantEvaluatorError::InvalidAccessBase)?;

                crate::proc::flatten_compose(ty, components, self.expressions, self.types)
                    .nth(index)
                    .ok_or(ConstantEvaluatorError::InvalidAccessIndex)
            }
            _ => Err(ConstantEvaluatorError::InvalidAccessBase),
        }
    }

    fn constant_index(&self, expr: Handle<Expression>) -> Result<usize, ConstantEvaluatorError> {
        match self.expressions[expr] {
            Expression::ZeroValue(ty)
                if matches!(
                    self.types[ty].inner,
                    crate::TypeInner::Scalar(crate::Scalar {
                        kind: ScalarKind::Uint,
                        ..
                    })
                ) =>
            {
                Ok(0)
            }
            Expression::Literal(Literal::U32(index)) => Ok(index as usize),
            _ => Err(ConstantEvaluatorError::InvalidAccessIndexTy),
        }
    }

    /// Lower [`ZeroValue`] and [`Splat`] expressions to [`Literal`] and [`Compose`] expressions.
    ///
    /// [`ZeroValue`]: Expression::ZeroValue
    /// [`Splat`]: Expression::Splat
    /// [`Literal`]: Expression::Literal
    /// [`Compose`]: Expression::Compose
    fn eval_zero_value_and_splat(
        &mut self,
        expr: Handle<Expression>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.expressions[expr] {
            Expression::ZeroValue(ty) => self.eval_zero_value_impl(ty, span),
            Expression::Splat { size, value } => self.splat(value, size, span),
            _ => Ok(expr),
        }
    }

    /// Lower [`ZeroValue`] expressions to [`Literal`] and [`Compose`] expressions.
    ///
    /// [`ZeroValue`]: Expression::ZeroValue
    /// [`Literal`]: Expression::Literal
    /// [`Compose`]: Expression::Compose
    fn eval_zero_value(
        &mut self,
        expr: Handle<Expression>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.expressions[expr] {
            Expression::ZeroValue(ty) => self.eval_zero_value_impl(ty, span),
            _ => Ok(expr),
        }
    }

    /// Lower [`ZeroValue`] expressions to [`Literal`] and [`Compose`] expressions.
    ///
    /// [`ZeroValue`]: Expression::ZeroValue
    /// [`Literal`]: Expression::Literal
    /// [`Compose`]: Expression::Compose
    fn eval_zero_value_impl(
        &mut self,
        ty: Handle<Type>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.types[ty].inner {
            TypeInner::Scalar(scalar) => {
                let expr = Expression::Literal(
                    Literal::zero(scalar).ok_or(ConstantEvaluatorError::TypeNotConstructible)?,
                );
                self.register_evaluated_expr(expr, span)
            }
            TypeInner::Vector { size, scalar } => {
                let scalar_ty = self.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Scalar(scalar),
                    },
                    span,
                );
                let el = self.eval_zero_value_impl(scalar_ty, span)?;
                let expr = Expression::Compose {
                    ty,
                    components: vec![el; size as usize],
                };
                self.register_evaluated_expr(expr, span)
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                let vec_ty = self.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Vector { size: rows, scalar },
                    },
                    span,
                );
                let el = self.eval_zero_value_impl(vec_ty, span)?;
                let expr = Expression::Compose {
                    ty,
                    components: vec![el; columns as usize],
                };
                self.register_evaluated_expr(expr, span)
            }
            TypeInner::Array {
                base,
                size: ArraySize::Constant(size),
                ..
            } => {
                let el = self.eval_zero_value_impl(base, span)?;
                let expr = Expression::Compose {
                    ty,
                    components: vec![el; size.get() as usize],
                };
                self.register_evaluated_expr(expr, span)
            }
            TypeInner::Struct { ref members, .. } => {
                let types: Vec<_> = members.iter().map(|m| m.ty).collect();
                let mut components = Vec::with_capacity(members.len());
                for ty in types {
                    components.push(self.eval_zero_value_impl(ty, span)?);
                }
                let expr = Expression::Compose { ty, components };
                self.register_evaluated_expr(expr, span)
            }
            _ => Err(ConstantEvaluatorError::TypeNotConstructible),
        }
    }

    /// Convert the scalar components of `expr` to `target`.
    ///
    /// Treat `span` as the location of the resulting expression.
    pub fn cast(
        &mut self,
        expr: Handle<Expression>,
        target: crate::Scalar,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        use crate::Scalar as Sc;

        let expr = self.eval_zero_value(expr, span)?;

        let make_error = || -> Result<_, ConstantEvaluatorError> {
            let from = format!("{:?} {:?}", expr, self.expressions[expr]);

            #[cfg(feature = "wgsl-in")]
            let to = target.to_wgsl();

            #[cfg(not(feature = "wgsl-in"))]
            let to = format!("{target:?}");

            Err(ConstantEvaluatorError::InvalidCastArg { from, to })
        };

        let expr = match self.expressions[expr] {
            Expression::Literal(literal) => {
                let literal = match target {
                    Sc::I32 => Literal::I32(match literal {
                        Literal::I32(v) => v,
                        Literal::U32(v) => v as i32,
                        Literal::F32(v) => v as i32,
                        Literal::Bool(v) => v as i32,
                        Literal::F64(_) | Literal::I64(_) => {
                            return make_error();
                        }
                        Literal::AbstractInt(v) => i32::try_from_abstract(v)?,
                        Literal::AbstractFloat(v) => i32::try_from_abstract(v)?,
                    }),
                    Sc::U32 => Literal::U32(match literal {
                        Literal::I32(v) => v as u32,
                        Literal::U32(v) => v,
                        Literal::F32(v) => v as u32,
                        Literal::Bool(v) => v as u32,
                        Literal::F64(_) | Literal::I64(_) => {
                            return make_error();
                        }
                        Literal::AbstractInt(v) => u32::try_from_abstract(v)?,
                        Literal::AbstractFloat(v) => u32::try_from_abstract(v)?,
                    }),
                    Sc::F32 => Literal::F32(match literal {
                        Literal::I32(v) => v as f32,
                        Literal::U32(v) => v as f32,
                        Literal::F32(v) => v,
                        Literal::Bool(v) => v as u32 as f32,
                        Literal::F64(_) | Literal::I64(_) => {
                            return make_error();
                        }
                        Literal::AbstractInt(v) => f32::try_from_abstract(v)?,
                        Literal::AbstractFloat(v) => f32::try_from_abstract(v)?,
                    }),
                    Sc::F64 => Literal::F64(match literal {
                        Literal::I32(v) => v as f64,
                        Literal::U32(v) => v as f64,
                        Literal::F32(v) => v as f64,
                        Literal::F64(v) => v,
                        Literal::Bool(v) => v as u32 as f64,
                        Literal::I64(_) => return make_error(),
                        Literal::AbstractInt(v) => f64::try_from_abstract(v)?,
                        Literal::AbstractFloat(v) => f64::try_from_abstract(v)?,
                    }),
                    Sc::BOOL => Literal::Bool(match literal {
                        Literal::I32(v) => v != 0,
                        Literal::U32(v) => v != 0,
                        Literal::F32(v) => v != 0.0,
                        Literal::Bool(v) => v,
                        Literal::F64(_)
                        | Literal::I64(_)
                        | Literal::AbstractInt(_)
                        | Literal::AbstractFloat(_) => {
                            return make_error();
                        }
                    }),
                    Sc::ABSTRACT_FLOAT => Literal::AbstractFloat(match literal {
                        Literal::AbstractInt(v) => {
                            // Overflow is forbidden, but inexact conversions
                            // are fine. The range of f64 is far larger than
                            // that of i64, so we don't have to check anything
                            // here.
                            v as f64
                        }
                        Literal::AbstractFloat(v) => v,
                        _ => return make_error(),
                    }),
                    _ => {
                        log::debug!("Constant evaluator refused to convert value to {target:?}");
                        return make_error();
                    }
                };
                Expression::Literal(literal)
            }
            Expression::Compose {
                ty,
                components: ref src_components,
            } => {
                let ty_inner = match self.types[ty].inner {
                    TypeInner::Vector { size, .. } => TypeInner::Vector {
                        size,
                        scalar: target,
                    },
                    TypeInner::Matrix { columns, rows, .. } => TypeInner::Matrix {
                        columns,
                        rows,
                        scalar: target,
                    },
                    _ => return make_error(),
                };

                let mut components = src_components.clone();
                for component in &mut components {
                    *component = self.cast(*component, target, span)?;
                }

                let ty = self.types.insert(
                    Type {
                        name: None,
                        inner: ty_inner,
                    },
                    span,
                );

                Expression::Compose { ty, components }
            }
            Expression::Splat { size, value } => {
                let value_span = self.expressions.get_span(value);
                let cast_value = self.cast(value, target, value_span)?;
                Expression::Splat {
                    size,
                    value: cast_value,
                }
            }
            _ => return make_error(),
        };

        self.register_evaluated_expr(expr, span)
    }

    /// Convert the scalar leaves of  `expr` to `target`, handling arrays.
    ///
    /// `expr` must be a `Compose` expression whose type is a scalar, vector,
    /// matrix, or nested arrays of such.
    ///
    /// This is basically the same as the [`cast`] method, except that that
    /// should only handle Naga [`As`] expressions, which cannot convert arrays.
    ///
    /// Treat `span` as the location of the resulting expression.
    ///
    /// [`cast`]: ConstantEvaluator::cast
    /// [`As`]: crate::Expression::As
    pub fn cast_array(
        &mut self,
        expr: Handle<Expression>,
        target: crate::Scalar,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let Expression::Compose { ty, ref components } = self.expressions[expr] else {
            return self.cast(expr, target, span);
        };

        let crate::TypeInner::Array { base: _, size, stride: _ } = self.types[ty].inner else {
            return self.cast(expr, target, span);
        };

        let mut components = components.clone();
        for component in &mut components {
            *component = self.cast_array(*component, target, span)?;
        }

        let first = components.first().unwrap();
        let new_base = match self.resolve_type(*first)? {
            crate::proc::TypeResolution::Handle(ty) => ty,
            crate::proc::TypeResolution::Value(inner) => {
                self.types.insert(Type { name: None, inner }, span)
            }
        };
        let new_base_stride = self.types[new_base].inner.size(self.to_ctx());
        let new_array_ty = self.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: new_base,
                    size,
                    stride: new_base_stride,
                },
            },
            span,
        );

        let compose = Expression::Compose {
            ty: new_array_ty,
            components,
        };
        self.register_evaluated_expr(compose, span)
    }

    fn unary_op(
        &mut self,
        op: UnaryOperator,
        expr: Handle<Expression>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let expr = self.eval_zero_value_and_splat(expr, span)?;

        let expr = match self.expressions[expr] {
            Expression::Literal(value) => Expression::Literal(match op {
                UnaryOperator::Negate => match value {
                    Literal::I32(v) => Literal::I32(v.wrapping_neg()),
                    Literal::F32(v) => Literal::F32(-v),
                    Literal::AbstractInt(v) => Literal::AbstractInt(v.wrapping_neg()),
                    Literal::AbstractFloat(v) => Literal::AbstractFloat(-v),
                    _ => return Err(ConstantEvaluatorError::InvalidUnaryOpArg),
                },
                UnaryOperator::LogicalNot => match value {
                    Literal::Bool(v) => Literal::Bool(!v),
                    _ => return Err(ConstantEvaluatorError::InvalidUnaryOpArg),
                },
                UnaryOperator::BitwiseNot => match value {
                    Literal::I32(v) => Literal::I32(!v),
                    Literal::U32(v) => Literal::U32(!v),
                    Literal::AbstractInt(v) => Literal::AbstractInt(!v),
                    _ => return Err(ConstantEvaluatorError::InvalidUnaryOpArg),
                },
            }),
            Expression::Compose {
                ty,
                components: ref src_components,
            } => {
                match self.types[ty].inner {
                    TypeInner::Vector { .. } | TypeInner::Matrix { .. } => (),
                    _ => return Err(ConstantEvaluatorError::InvalidUnaryOpArg),
                }

                let mut components = src_components.clone();
                for component in &mut components {
                    *component = self.unary_op(op, *component, span)?;
                }

                Expression::Compose { ty, components }
            }
            _ => return Err(ConstantEvaluatorError::InvalidUnaryOpArg),
        };

        self.register_evaluated_expr(expr, span)
    }

    fn binary_op(
        &mut self,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let left = self.eval_zero_value_and_splat(left, span)?;
        let right = self.eval_zero_value_and_splat(right, span)?;

        let expr = match (&self.expressions[left], &self.expressions[right]) {
            (&Expression::Literal(left_value), &Expression::Literal(right_value)) => {
                let literal = match op {
                    BinaryOperator::Equal => Literal::Bool(left_value == right_value),
                    BinaryOperator::NotEqual => Literal::Bool(left_value != right_value),
                    BinaryOperator::Less => Literal::Bool(left_value < right_value),
                    BinaryOperator::LessEqual => Literal::Bool(left_value <= right_value),
                    BinaryOperator::Greater => Literal::Bool(left_value > right_value),
                    BinaryOperator::GreaterEqual => Literal::Bool(left_value >= right_value),

                    _ => match (left_value, right_value) {
                        (Literal::I32(a), Literal::I32(b)) => Literal::I32(match op {
                            BinaryOperator::Add => a.checked_add(b).ok_or_else(|| {
                                ConstantEvaluatorError::Overflow("addition".into())
                            })?,
                            BinaryOperator::Subtract => a.checked_sub(b).ok_or_else(|| {
                                ConstantEvaluatorError::Overflow("subtraction".into())
                            })?,
                            BinaryOperator::Multiply => a.checked_mul(b).ok_or_else(|| {
                                ConstantEvaluatorError::Overflow("multiplication".into())
                            })?,
                            BinaryOperator::Divide => a.checked_div(b).ok_or_else(|| {
                                if b == 0 {
                                    ConstantEvaluatorError::DivisionByZero
                                } else {
                                    ConstantEvaluatorError::Overflow("division".into())
                                }
                            })?,
                            BinaryOperator::Modulo => a.checked_rem(b).ok_or_else(|| {
                                if b == 0 {
                                    ConstantEvaluatorError::RemainderByZero
                                } else {
                                    ConstantEvaluatorError::Overflow("remainder".into())
                                }
                            })?,
                            BinaryOperator::And => a & b,
                            BinaryOperator::ExclusiveOr => a ^ b,
                            BinaryOperator::InclusiveOr => a | b,
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
                        (Literal::I32(a), Literal::U32(b)) => Literal::I32(match op {
                            BinaryOperator::ShiftLeft => a
                                .checked_shl(b)
                                .ok_or(ConstantEvaluatorError::ShiftedMoreThan32Bits)?,
                            BinaryOperator::ShiftRight => a
                                .checked_shr(b)
                                .ok_or(ConstantEvaluatorError::ShiftedMoreThan32Bits)?,
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
                        (Literal::U32(a), Literal::U32(b)) => Literal::U32(match op {
                            BinaryOperator::Add => a.checked_add(b).ok_or_else(|| {
                                ConstantEvaluatorError::Overflow("addition".into())
                            })?,
                            BinaryOperator::Subtract => a.checked_sub(b).ok_or_else(|| {
                                ConstantEvaluatorError::Overflow("subtraction".into())
                            })?,
                            BinaryOperator::Multiply => a.checked_mul(b).ok_or_else(|| {
                                ConstantEvaluatorError::Overflow("multiplication".into())
                            })?,
                            BinaryOperator::Divide => a
                                .checked_div(b)
                                .ok_or(ConstantEvaluatorError::DivisionByZero)?,
                            BinaryOperator::Modulo => a
                                .checked_rem(b)
                                .ok_or(ConstantEvaluatorError::RemainderByZero)?,
                            BinaryOperator::And => a & b,
                            BinaryOperator::ExclusiveOr => a ^ b,
                            BinaryOperator::InclusiveOr => a | b,
                            BinaryOperator::ShiftLeft => a
                                .checked_shl(b)
                                .ok_or(ConstantEvaluatorError::ShiftedMoreThan32Bits)?,
                            BinaryOperator::ShiftRight => a
                                .checked_shr(b)
                                .ok_or(ConstantEvaluatorError::ShiftedMoreThan32Bits)?,
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
                        (Literal::F32(a), Literal::F32(b)) => Literal::F32(match op {
                            BinaryOperator::Add => a + b,
                            BinaryOperator::Subtract => a - b,
                            BinaryOperator::Multiply => a * b,
                            BinaryOperator::Divide => a / b,
                            BinaryOperator::Modulo => a % b,
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
                        (Literal::AbstractInt(a), Literal::AbstractInt(b)) => {
                            Literal::AbstractInt(match op {
                                BinaryOperator::Add => a.checked_add(b).ok_or_else(|| {
                                    ConstantEvaluatorError::Overflow("addition".into())
                                })?,
                                BinaryOperator::Subtract => a.checked_sub(b).ok_or_else(|| {
                                    ConstantEvaluatorError::Overflow("subtraction".into())
                                })?,
                                BinaryOperator::Multiply => a.checked_mul(b).ok_or_else(|| {
                                    ConstantEvaluatorError::Overflow("multiplication".into())
                                })?,
                                BinaryOperator::Divide => a.checked_div(b).ok_or_else(|| {
                                    if b == 0 {
                                        ConstantEvaluatorError::DivisionByZero
                                    } else {
                                        ConstantEvaluatorError::Overflow("division".into())
                                    }
                                })?,
                                BinaryOperator::Modulo => a.checked_rem(b).ok_or_else(|| {
                                    if b == 0 {
                                        ConstantEvaluatorError::RemainderByZero
                                    } else {
                                        ConstantEvaluatorError::Overflow("remainder".into())
                                    }
                                })?,
                                BinaryOperator::And => a & b,
                                BinaryOperator::ExclusiveOr => a ^ b,
                                BinaryOperator::InclusiveOr => a | b,
                                _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                            })
                        }
                        (Literal::AbstractFloat(a), Literal::AbstractFloat(b)) => {
                            Literal::AbstractFloat(match op {
                                BinaryOperator::Add => a + b,
                                BinaryOperator::Subtract => a - b,
                                BinaryOperator::Multiply => a * b,
                                BinaryOperator::Divide => a / b,
                                BinaryOperator::Modulo => a % b,
                                _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                            })
                        }
                        (Literal::Bool(a), Literal::Bool(b)) => Literal::Bool(match op {
                            BinaryOperator::LogicalAnd => a && b,
                            BinaryOperator::LogicalOr => a || b,
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
                        _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                    },
                };
                Expression::Literal(literal)
            }
            (
                &Expression::Compose {
                    components: ref src_components,
                    ty,
                },
                &Expression::Literal(_),
            ) => {
                let mut components = src_components.clone();
                for component in &mut components {
                    *component = self.binary_op(op, *component, right, span)?;
                }
                Expression::Compose { ty, components }
            }
            (
                &Expression::Literal(_),
                &Expression::Compose {
                    components: ref src_components,
                    ty,
                },
            ) => {
                let mut components = src_components.clone();
                for component in &mut components {
                    *component = self.binary_op(op, left, *component, span)?;
                }
                Expression::Compose { ty, components }
            }
            (
                &Expression::Compose {
                    components: ref left_components,
                    ty: left_ty,
                },
                &Expression::Compose {
                    components: ref right_components,
                    ty: right_ty,
                },
            ) => {
                // We have to make a copy of the component lists, because the
                // call to `binary_op_vector` needs `&mut self`, but `self` owns
                // the component lists.
                let left_flattened = crate::proc::flatten_compose(
                    left_ty,
                    left_components,
                    self.expressions,
                    self.types,
                );
                let right_flattened = crate::proc::flatten_compose(
                    right_ty,
                    right_components,
                    self.expressions,
                    self.types,
                );

                // `flatten_compose` doesn't return an `ExactSizeIterator`, so
                // make a reasonable guess of the capacity we'll need.
                let mut flattened = Vec::with_capacity(left_components.len());
                flattened.extend(left_flattened.zip(right_flattened));

                match (&self.types[left_ty].inner, &self.types[right_ty].inner) {
                    (
                        &TypeInner::Vector {
                            size: left_size, ..
                        },
                        &TypeInner::Vector {
                            size: right_size, ..
                        },
                    ) if left_size == right_size => {
                        self.binary_op_vector(op, left_size, &flattened, left_ty, span)?
                    }
                    _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                }
            }
            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
        };

        self.register_evaluated_expr(expr, span)
    }

    fn binary_op_vector(
        &mut self,
        op: BinaryOperator,
        size: crate::VectorSize,
        components: &[(Handle<Expression>, Handle<Expression>)],
        left_ty: Handle<Type>,
        span: Span,
    ) -> Result<Expression, ConstantEvaluatorError> {
        let ty = match op {
            // Relational operators produce vectors of booleans.
            BinaryOperator::Equal
            | BinaryOperator::NotEqual
            | BinaryOperator::Less
            | BinaryOperator::LessEqual
            | BinaryOperator::Greater
            | BinaryOperator::GreaterEqual => self.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Vector {
                        size,
                        scalar: crate::Scalar::BOOL,
                    },
                },
                span,
            ),

            // Other operators produce the same type as their left
            // operand.
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo
            | BinaryOperator::And
            | BinaryOperator::ExclusiveOr
            | BinaryOperator::InclusiveOr
            | BinaryOperator::LogicalAnd
            | BinaryOperator::LogicalOr
            | BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight => left_ty,
        };

        let components = components
            .iter()
            .map(|&(left, right)| self.binary_op(op, left, right, span))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Expression::Compose { ty, components })
    }

    /// Deep copy `expr` from `expressions` into `self.expressions`.
    ///
    /// Return the root of the new copy.
    ///
    /// This is used when we're evaluating expressions in a function's
    /// expression arena that refer to a constant: we need to copy the
    /// constant's value into the function's arena so we can operate on it.
    fn copy_from(
        &mut self,
        expr: Handle<Expression>,
        expressions: &Arena<Expression>,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let span = expressions.get_span(expr);
        match expressions[expr] {
            ref expr @ (Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::ZeroValue(_)) => self.register_evaluated_expr(expr.clone(), span),
            Expression::Compose { ty, ref components } => {
                let mut components = components.clone();
                for component in &mut components {
                    *component = self.copy_from(*component, expressions)?;
                }
                self.register_evaluated_expr(Expression::Compose { ty, components }, span)
            }
            Expression::Splat { size, value } => {
                let value = self.copy_from(value, expressions)?;
                self.register_evaluated_expr(Expression::Splat { size, value }, span)
            }
            _ => {
                log::debug!("copy_from: SubexpressionsAreNotConstant");
                Err(ConstantEvaluatorError::SubexpressionsAreNotConstant)
            }
        }
    }

    fn register_evaluated_expr(
        &mut self,
        expr: Expression,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        // It suffices to only check literals, since we only register one
        // expression at a time, `Compose` expressions can only refer to other
        // expressions, and `ZeroValue` expressions are always okay.
        if let Expression::Literal(literal) = expr {
            crate::valid::check_literal_value(literal)?;
        }

        if let Some(FunctionLocalData {
            ref mut emitter,
            ref mut block,
            ref mut expression_constness,
            ..
        }) = self.function_local_data
        {
            let is_running = emitter.is_running();
            let needs_pre_emit = expr.needs_pre_emit();
            if is_running && needs_pre_emit {
                block.extend(emitter.finish(self.expressions));
                let h = self.expressions.append(expr, span);
                emitter.start(self.expressions);
                expression_constness.insert(h);
                Ok(h)
            } else {
                let h = self.expressions.append(expr, span);
                expression_constness.insert(h);
                Ok(h)
            }
        } else {
            Ok(self.expressions.append(expr, span))
        }
    }

    fn resolve_type(
        &self,
        expr: Handle<Expression>,
    ) -> Result<crate::proc::TypeResolution, ConstantEvaluatorError> {
        use crate::proc::TypeResolution as Tr;
        use crate::Expression as Ex;
        let resolution = match self.expressions[expr] {
            Ex::Literal(ref literal) => Tr::Value(literal.ty_inner()),
            Ex::Constant(c) => Tr::Handle(self.constants[c].ty),
            Ex::ZeroValue(ty) | Ex::Compose { ty, .. } => Tr::Handle(ty),
            Ex::Splat { size, value } => {
                let Tr::Value(TypeInner::Scalar(scalar)) = self.resolve_type(value)? else {
                    return Err(ConstantEvaluatorError::SplatScalarOnly);
                };
                Tr::Value(TypeInner::Vector { scalar, size })
            }
            _ => {
                log::debug!("resolve_type: SubexpressionsAreNotConstant");
                return Err(ConstantEvaluatorError::SubexpressionsAreNotConstant);
            }
        };

        Ok(resolution)
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::{
        Arena, Constant, Expression, Literal, ScalarKind, Type, TypeInner, UnaryOperator,
        UniqueArena, VectorSize,
    };

    use super::{Behavior, ConstantEvaluator};

    #[test]
    fn unary_op() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let mut const_expressions = Arena::new();

        let scalar_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::I32),
            },
            Default::default(),
        );

        let vec_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Bi,
                    scalar: crate::Scalar::I32,
                },
            },
            Default::default(),
        );

        let h = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: scalar_ty,
                init: const_expressions
                    .append(Expression::Literal(Literal::I32(4)), Default::default()),
            },
            Default::default(),
        );

        let h1 = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: scalar_ty,
                init: const_expressions
                    .append(Expression::Literal(Literal::I32(8)), Default::default()),
            },
            Default::default(),
        );

        let vec_h = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: vec_ty,
                init: const_expressions.append(
                    Expression::Compose {
                        ty: vec_ty,
                        components: vec![constants[h].init, constants[h1].init],
                    },
                    Default::default(),
                ),
            },
            Default::default(),
        );

        let expr = const_expressions.append(Expression::Constant(h), Default::default());
        let expr1 = const_expressions.append(Expression::Constant(vec_h), Default::default());

        let expr2 = Expression::Unary {
            op: UnaryOperator::Negate,
            expr,
        };

        let expr3 = Expression::Unary {
            op: UnaryOperator::BitwiseNot,
            expr,
        };

        let expr4 = Expression::Unary {
            op: UnaryOperator::BitwiseNot,
            expr: expr1,
        };

        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl,
            types: &mut types,
            constants: &constants,
            expressions: &mut const_expressions,
            function_local_data: None,
        };

        let res1 = solver
            .try_eval_and_append(&expr2, Default::default())
            .unwrap();
        let res2 = solver
            .try_eval_and_append(&expr3, Default::default())
            .unwrap();
        let res3 = solver
            .try_eval_and_append(&expr4, Default::default())
            .unwrap();

        assert_eq!(
            const_expressions[res1],
            Expression::Literal(Literal::I32(-4))
        );

        assert_eq!(
            const_expressions[res2],
            Expression::Literal(Literal::I32(!4))
        );

        let res3_inner = &const_expressions[res3];

        match *res3_inner {
            Expression::Compose {
                ref ty,
                ref components,
            } => {
                assert_eq!(*ty, vec_ty);
                let mut components_iter = components.iter().copied();
                assert_eq!(
                    const_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::I32(!4))
                );
                assert_eq!(
                    const_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::I32(!8))
                );
                assert!(components_iter.next().is_none());
            }
            _ => panic!("Expected vector"),
        }
    }

    #[test]
    fn cast() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let mut const_expressions = Arena::new();

        let scalar_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::I32),
            },
            Default::default(),
        );

        let h = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: scalar_ty,
                init: const_expressions
                    .append(Expression::Literal(Literal::I32(4)), Default::default()),
            },
            Default::default(),
        );

        let expr = const_expressions.append(Expression::Constant(h), Default::default());

        let root = Expression::As {
            expr,
            kind: ScalarKind::Bool,
            convert: Some(crate::BOOL_WIDTH),
        };

        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl,
            types: &mut types,
            constants: &constants,
            expressions: &mut const_expressions,
            function_local_data: None,
        };

        let res = solver
            .try_eval_and_append(&root, Default::default())
            .unwrap();

        assert_eq!(
            const_expressions[res],
            Expression::Literal(Literal::Bool(true))
        );
    }

    #[test]
    fn access() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let mut const_expressions = Arena::new();

        let matrix_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Matrix {
                    columns: VectorSize::Bi,
                    rows: VectorSize::Tri,
                    scalar: crate::Scalar::F32,
                },
            },
            Default::default(),
        );

        let vec_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: crate::Scalar::F32,
                },
            },
            Default::default(),
        );

        let mut vec1_components = Vec::with_capacity(3);
        let mut vec2_components = Vec::with_capacity(3);

        for i in 0..3 {
            let h = const_expressions.append(
                Expression::Literal(Literal::F32(i as f32)),
                Default::default(),
            );

            vec1_components.push(h)
        }

        for i in 3..6 {
            let h = const_expressions.append(
                Expression::Literal(Literal::F32(i as f32)),
                Default::default(),
            );

            vec2_components.push(h)
        }

        let vec1 = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: vec_ty,
                init: const_expressions.append(
                    Expression::Compose {
                        ty: vec_ty,
                        components: vec1_components,
                    },
                    Default::default(),
                ),
            },
            Default::default(),
        );

        let vec2 = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: vec_ty,
                init: const_expressions.append(
                    Expression::Compose {
                        ty: vec_ty,
                        components: vec2_components,
                    },
                    Default::default(),
                ),
            },
            Default::default(),
        );

        let h = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: matrix_ty,
                init: const_expressions.append(
                    Expression::Compose {
                        ty: matrix_ty,
                        components: vec![constants[vec1].init, constants[vec2].init],
                    },
                    Default::default(),
                ),
            },
            Default::default(),
        );

        let base = const_expressions.append(Expression::Constant(h), Default::default());

        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl,
            types: &mut types,
            constants: &constants,
            expressions: &mut const_expressions,
            function_local_data: None,
        };

        let root1 = Expression::AccessIndex { base, index: 1 };

        let res1 = solver
            .try_eval_and_append(&root1, Default::default())
            .unwrap();

        let root2 = Expression::AccessIndex {
            base: res1,
            index: 2,
        };

        let res2 = solver
            .try_eval_and_append(&root2, Default::default())
            .unwrap();

        match const_expressions[res1] {
            Expression::Compose {
                ref ty,
                ref components,
            } => {
                assert_eq!(*ty, vec_ty);
                let mut components_iter = components.iter().copied();
                assert_eq!(
                    const_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::F32(3.))
                );
                assert_eq!(
                    const_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::F32(4.))
                );
                assert_eq!(
                    const_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::F32(5.))
                );
                assert!(components_iter.next().is_none());
            }
            _ => panic!("Expected vector"),
        }

        assert_eq!(
            const_expressions[res2],
            Expression::Literal(Literal::F32(5.))
        );
    }

    #[test]
    fn compose_of_constants() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let mut const_expressions = Arena::new();

        let i32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::I32),
            },
            Default::default(),
        );

        let vec2_i32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Bi,
                    scalar: crate::Scalar::I32,
                },
            },
            Default::default(),
        );

        let h = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: i32_ty,
                init: const_expressions
                    .append(Expression::Literal(Literal::I32(4)), Default::default()),
            },
            Default::default(),
        );

        let h_expr = const_expressions.append(Expression::Constant(h), Default::default());

        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl,
            types: &mut types,
            constants: &constants,
            expressions: &mut const_expressions,
            function_local_data: None,
        };

        let solved_compose = solver
            .try_eval_and_append(
                &Expression::Compose {
                    ty: vec2_i32_ty,
                    components: vec![h_expr, h_expr],
                },
                Default::default(),
            )
            .unwrap();
        let solved_negate = solver
            .try_eval_and_append(
                &Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: solved_compose,
                },
                Default::default(),
            )
            .unwrap();

        let pass = match const_expressions[solved_negate] {
            Expression::Compose { ty, ref components } => {
                ty == vec2_i32_ty
                    && components.iter().all(|&component| {
                        let component = &const_expressions[component];
                        matches!(*component, Expression::Literal(Literal::I32(-4)))
                    })
            }
            _ => false,
        };
        if !pass {
            panic!("unexpected evaluation result")
        }
    }

    #[test]
    fn splat_of_constant() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let mut const_expressions = Arena::new();

        let i32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::I32),
            },
            Default::default(),
        );

        let vec2_i32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Bi,
                    scalar: crate::Scalar::I32,
                },
            },
            Default::default(),
        );

        let h = constants.append(
            Constant {
                name: None,
                r#override: crate::Override::None,
                ty: i32_ty,
                init: const_expressions
                    .append(Expression::Literal(Literal::I32(4)), Default::default()),
            },
            Default::default(),
        );

        let h_expr = const_expressions.append(Expression::Constant(h), Default::default());

        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl,
            types: &mut types,
            constants: &constants,
            expressions: &mut const_expressions,
            function_local_data: None,
        };

        let solved_compose = solver
            .try_eval_and_append(
                &Expression::Splat {
                    size: VectorSize::Bi,
                    value: h_expr,
                },
                Default::default(),
            )
            .unwrap();
        let solved_negate = solver
            .try_eval_and_append(
                &Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: solved_compose,
                },
                Default::default(),
            )
            .unwrap();

        let pass = match const_expressions[solved_negate] {
            Expression::Compose { ty, ref components } => {
                ty == vec2_i32_ty
                    && components.iter().all(|&component| {
                        let component = &const_expressions[component];
                        matches!(*component, Expression::Literal(Literal::I32(-4)))
                    })
            }
            _ => false,
        };
        if !pass {
            panic!("unexpected evaluation result")
        }
    }
}

/// Trait for conversions of abstract values to concrete types.
trait TryFromAbstract<T>: Sized {
    /// Convert an abstract literal `value` to `Self`.
    ///
    /// Since Naga's `AbstractInt` and `AbstractFloat` exist to support
    /// WGSL, we follow WGSL's conversion rules here:
    ///
    /// - WGSL §6.1.2. Conversion Rank says that automatic conversions
    ///   to integers are either lossless or an error.
    ///
    /// - WGSL §14.6.4 Floating Point Conversion says that conversions
    ///   to floating point in constant expressions and override
    ///   expressions are errors if the value is out of range for the
    ///   destination type, but rounding is okay.
    ///
    /// [`AbstractInt`]: crate::Literal::AbstractInt
    /// [`Float`]: crate::Literal::Float
    fn try_from_abstract(value: T) -> Result<Self, ConstantEvaluatorError>;
}

impl TryFromAbstract<i64> for i32 {
    fn try_from_abstract(value: i64) -> Result<i32, ConstantEvaluatorError> {
        i32::try_from(value).map_err(|_| ConstantEvaluatorError::AutomaticConversionLossy {
            value: format!("{value:?}"),
            to_type: "i32",
        })
    }
}

impl TryFromAbstract<i64> for u32 {
    fn try_from_abstract(value: i64) -> Result<u32, ConstantEvaluatorError> {
        u32::try_from(value).map_err(|_| ConstantEvaluatorError::AutomaticConversionLossy {
            value: format!("{value:?}"),
            to_type: "u32",
        })
    }
}

impl TryFromAbstract<i64> for f32 {
    fn try_from_abstract(value: i64) -> Result<Self, ConstantEvaluatorError> {
        let f = value as f32;
        // The range of `i64` is roughly ±18 × 10¹⁸, whereas the range of
        // `f32` is roughly ±3.4 × 10³⁸, so there's no opportunity for
        // overflow here.
        Ok(f)
    }
}

impl TryFromAbstract<f64> for f32 {
    fn try_from_abstract(value: f64) -> Result<f32, ConstantEvaluatorError> {
        let f = value as f32;
        if f.is_infinite() {
            return Err(ConstantEvaluatorError::AutomaticConversionLossy {
                value: format!("{value:?}"),
                to_type: "f32",
            });
        }
        Ok(f)
    }
}

impl TryFromAbstract<i64> for f64 {
    fn try_from_abstract(value: i64) -> Result<Self, ConstantEvaluatorError> {
        let f = value as f64;
        // The range of `i64` is roughly ±18 × 10¹⁸, whereas the range of
        // `f64` is roughly ±1.8 × 10³⁰⁸, so there's no opportunity for
        // overflow here.
        Ok(f)
    }
}

impl TryFromAbstract<f64> for f64 {
    fn try_from_abstract(value: f64) -> Result<f64, ConstantEvaluatorError> {
        Ok(value)
    }
}

impl TryFromAbstract<f64> for i32 {
    fn try_from_abstract(_: f64) -> Result<Self, ConstantEvaluatorError> {
        Err(ConstantEvaluatorError::AutomaticConversionFloatToInt { to_type: "i32" })
    }
}

impl TryFromAbstract<f64> for u32 {
    fn try_from_abstract(_: f64) -> Result<Self, ConstantEvaluatorError> {
        Err(ConstantEvaluatorError::AutomaticConversionFloatToInt { to_type: "u32" })
    }
}
