use std::iter;

use arrayvec::ArrayVec;

use crate::{
    arena::{Arena, Handle, HandleVec, UniqueArena},
    ArraySize, BinaryOperator, Constant, Expression, Literal, Override, ScalarKind, Span, Type,
    TypeInner, UnaryOperator,
};

/// A macro that allows dollar signs (`$`) to be emitted by other macros. Useful for generating
/// `macro_rules!` items that, in turn, emit their own `macro_rules!` items.
///
/// Technique stolen directly from
/// <https://github.com/rust-lang/rust/issues/35853#issuecomment-415993963>.
macro_rules! with_dollar_sign {
    ($($body:tt)*) => {
        macro_rules! __with_dollar_sign { $($body)* }
        __with_dollar_sign!($);
    }
}

macro_rules! gen_component_wise_extractor {
    (
        $ident:ident -> $target:ident,
        literals: [$( $literal:ident => $mapping:ident: $ty:ident ),+ $(,)?],
        scalar_kinds: [$( $scalar_kind:ident ),* $(,)?],
    ) => {
        /// A subset of [`Literal`]s intended to be used for implementing numeric built-ins.
        #[derive(Debug)]
        #[cfg_attr(test, derive(PartialEq))]
        enum $target<const N: usize> {
            $(
                #[doc = concat!(
                    "Maps to [`Literal::",
                    stringify!($literal),
                    "`]",
                )]
                $mapping([$ty; N]),
            )+
        }

        impl From<$target<1>> for Expression {
            fn from(value: $target<1>) -> Self {
                match value {
                    $(
                        $target::$mapping([value]) => {
                            Expression::Literal(Literal::$literal(value))
                        }
                    )+
                }
            }
        }

        #[doc = concat!(
            "Attempts to evaluate multiple `exprs` as a combined [`",
            stringify!($target),
            "`] to pass to `handler`. ",
        )]
        /// If `exprs` are vectors of the same length, `handler` is called for each corresponding
        /// component of each vector.
        ///
        /// `handler`'s output is registered as a new expression. If `exprs` are vectors of the
        /// same length, a new vector expression is registered, composed of each component emitted
        /// by `handler`.
        fn $ident<const N: usize, const M: usize, F>(
            eval: &mut ConstantEvaluator<'_>,
            span: Span,
            exprs: [Handle<Expression>; N],
            mut handler: F,
        ) -> Result<Handle<Expression>, ConstantEvaluatorError>
        where
            $target<M>: Into<Expression>,
            F: FnMut($target<N>) -> Result<$target<M>, ConstantEvaluatorError> + Clone,
        {
            assert!(N > 0);
            let err = ConstantEvaluatorError::InvalidMathArg;
            let mut exprs = exprs.into_iter();

            macro_rules! sanitize {
                ($expr:expr) => {
                    eval.eval_zero_value_and_splat($expr, span)
                        .map(|expr| &eval.expressions[expr])
                };
            }

            let new_expr = match sanitize!(exprs.next().unwrap())? {
                $(
                    &Expression::Literal(Literal::$literal(x)) => iter::once(Ok(x))
                        .chain(exprs.map(|expr| {
                            sanitize!(expr).and_then(|expr| match expr {
                                &Expression::Literal(Literal::$literal(x)) => Ok(x),
                                _ => Err(err.clone()),
                            })
                        }))
                        .collect::<Result<ArrayVec<_, N>, _>>()
                        .map(|a| a.into_inner().unwrap())
                        .map($target::$mapping)
                        .and_then(|comps| Ok(handler(comps)?.into())),
                )+
                &Expression::Compose { ty, ref components } => match &eval.types[ty].inner {
                    &TypeInner::Vector { size, scalar } => match scalar.kind {
                        $(ScalarKind::$scalar_kind)|* => {
                            let first_ty = ty;
                            let mut component_groups =
                                ArrayVec::<ArrayVec<_, { crate::VectorSize::MAX }>, N>::new();
                            component_groups.push(crate::proc::flatten_compose(
                                first_ty,
                                components,
                                eval.expressions,
                                eval.types,
                            ).collect());
                            component_groups.extend(
                                exprs
                                    .map(|expr| {
                                        sanitize!(expr).and_then(|expr| match expr {
                                            &Expression::Compose { ty, ref components }
                                                if &eval.types[ty].inner
                                                    == &eval.types[first_ty].inner =>
                                            {
                                                Ok(crate::proc::flatten_compose(
                                                    ty,
                                                    components,
                                                    eval.expressions,
                                                    eval.types,
                                                ).collect())
                                            }
                                            _ => Err(err.clone()),
                                        })
                                    })
                                    .collect::<Result<ArrayVec<_, { crate::VectorSize::MAX }>, _>>(
                                    )?,
                            );
                            let component_groups = component_groups.into_inner().unwrap();
                            let mut new_components =
                                ArrayVec::<_, { crate::VectorSize::MAX }>::new();
                            for idx in 0..(size as u8).into() {
                                let group = component_groups
                                    .iter()
                                    .map(|cs| cs.get(idx).cloned().ok_or(err.clone()))
                                    .collect::<Result<ArrayVec<_, N>, _>>()?
                                    .into_inner()
                                    .unwrap();
                                new_components.push($ident(
                                    eval,
                                    span,
                                    group,
                                    handler.clone(),
                                )?);
                            }
                            Ok(Expression::Compose {
                                ty: first_ty,
                                components: new_components.into_iter().collect(),
                            })
                        }
                        _ => return Err(err),
                    },
                    _ => return Err(err),
                },
                _ => return Err(err),
            }?;
            eval.register_evaluated_expr(new_expr, span)
        }

        with_dollar_sign! {
            ($d:tt) => {
                #[allow(unused)]
                #[doc = concat!(
                    "A convenience macro for using the same RHS for each [`",
                    stringify!($target),
                    "`] variant in a call to [`",
                    stringify!($ident),
                    "`].",
                )]
                macro_rules! $ident {
                    (
                        $eval:expr,
                        $span:expr,
                        [$d ($d expr:expr),+ $d (,)?],
                        |$d ($d arg:ident),+| $d tt:tt
                    ) => {
                        $ident($eval, $span, [$d ($d expr),+], |args| match args {
                            $(
                                $target::$mapping([$d ($d arg),+]) => {
                                    let res = $d tt;
                                    Result::map(res, $target::$mapping)
                                },
                            )+
                        })
                    };
                }
            };
        }
    };
}

gen_component_wise_extractor! {
    component_wise_scalar -> Scalar,
    literals: [
        AbstractFloat => AbstractFloat: f64,
        F32 => F32: f32,
        AbstractInt => AbstractInt: i64,
        U32 => U32: u32,
        I32 => I32: i32,
        U64 => U64: u64,
        I64 => I64: i64,
    ],
    scalar_kinds: [
        Float,
        AbstractFloat,
        Sint,
        Uint,
        AbstractInt,
    ],
}

gen_component_wise_extractor! {
    component_wise_float -> Float,
    literals: [
        AbstractFloat => Abstract: f64,
        F32 => F32: f32,
    ],
    scalar_kinds: [
        Float,
        AbstractFloat,
    ],
}

gen_component_wise_extractor! {
    component_wise_concrete_int -> ConcreteInt,
    literals: [
        U32 => U32: u32,
        I32 => I32: i32,
    ],
    scalar_kinds: [
        Sint,
        Uint,
    ],
}

gen_component_wise_extractor! {
    component_wise_signed -> Signed,
    literals: [
        AbstractFloat => AbstractFloat: f64,
        AbstractInt => AbstractInt: i64,
        F32 => F32: f32,
        I32 => I32: i32,
    ],
    scalar_kinds: [
        Sint,
        AbstractInt,
        Float,
        AbstractFloat,
    ],
}

#[derive(Debug)]
enum Behavior<'a> {
    Wgsl(WgslRestrictions<'a>),
    Glsl(GlslRestrictions<'a>),
}

impl Behavior<'_> {
    /// Returns `true` if the inner WGSL/GLSL restrictions are runtime restrictions.
    const fn has_runtime_restrictions(&self) -> bool {
        matches!(
            self,
            &Behavior::Wgsl(WgslRestrictions::Runtime(_))
                | &Behavior::Glsl(GlslRestrictions::Runtime(_))
        )
    }
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
    behavior: Behavior<'a>,

    /// The module's type arena.
    ///
    /// Because expressions like [`Splat`] contain type handles, we need to be
    /// able to add new types to produce those expressions.
    ///
    /// [`Splat`]: Expression::Splat
    types: &'a mut UniqueArena<Type>,

    /// The module's constant arena.
    constants: &'a Arena<Constant>,

    /// The module's override arena.
    overrides: &'a Arena<Override>,

    /// The arena to which we are contributing expressions.
    expressions: &'a mut Arena<Expression>,

    /// Tracks the constness of expressions residing in [`Self::expressions`]
    expression_kind_tracker: &'a mut ExpressionKindTracker,
}

#[derive(Debug)]
enum WgslRestrictions<'a> {
    /// - const-expressions will be evaluated and inserted in the arena
    Const(Option<FunctionLocalData<'a>>),
    /// - const-expressions will be evaluated and inserted in the arena
    /// - override-expressions will be inserted in the arena
    Override,
    /// - const-expressions will be evaluated and inserted in the arena
    /// - override-expressions will be inserted in the arena
    /// - runtime-expressions will be inserted in the arena
    Runtime(FunctionLocalData<'a>),
}

#[derive(Debug)]
enum GlslRestrictions<'a> {
    /// - const-expressions will be evaluated and inserted in the arena
    Const,
    /// - const-expressions will be evaluated and inserted in the arena
    /// - override-expressions will be inserted in the arena
    /// - runtime-expressions will be inserted in the arena
    Runtime(FunctionLocalData<'a>),
}

#[derive(Debug)]
struct FunctionLocalData<'a> {
    /// Global constant expressions
    global_expressions: &'a Arena<Expression>,
    emitter: &'a mut super::Emitter,
    block: &'a mut crate::Block,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum ExpressionKind {
    /// If const is also implemented as const
    ImplConst,
    Const,
    Override,
    Runtime,
}

#[derive(Debug)]
pub struct ExpressionKindTracker {
    inner: HandleVec<Expression, ExpressionKind>,
}

impl ExpressionKindTracker {
    pub const fn new() -> Self {
        Self {
            inner: HandleVec::new(),
        }
    }

    /// Forces the the expression to not be const
    pub fn force_non_const(&mut self, value: Handle<Expression>) {
        self.inner[value] = ExpressionKind::Runtime;
    }

    pub fn insert(&mut self, value: Handle<Expression>, expr_type: ExpressionKind) {
        self.inner.insert(value, expr_type);
    }

    pub fn is_const(&self, h: Handle<Expression>) -> bool {
        matches!(
            self.type_of(h),
            ExpressionKind::Const | ExpressionKind::ImplConst
        )
    }

    /// Returns `true` if naga can also evaluate expression as const
    pub fn is_impl_const(&self, h: Handle<Expression>) -> bool {
        matches!(self.type_of(h), ExpressionKind::ImplConst)
    }

    pub fn is_const_or_override(&self, h: Handle<Expression>) -> bool {
        matches!(
            self.type_of(h),
            ExpressionKind::Const | ExpressionKind::Override | ExpressionKind::ImplConst
        )
    }

    fn type_of(&self, value: Handle<Expression>) -> ExpressionKind {
        self.inner[value]
    }

    pub fn from_arena(arena: &Arena<Expression>) -> Self {
        let mut tracker = Self {
            inner: HandleVec::with_capacity(arena.len()),
        };
        for (handle, expr) in arena.iter() {
            tracker
                .inner
                .insert(handle, tracker.type_of_with_expr(expr));
        }
        tracker
    }

    fn type_of_with_expr(&self, expr: &Expression) -> ExpressionKind {
        use crate::MathFunction as Mf;
        match *expr {
            Expression::Literal(_) | Expression::ZeroValue(_) | Expression::Constant(_) => {
                ExpressionKind::ImplConst
            }
            Expression::Override(_) => ExpressionKind::Override,
            Expression::Compose { ref components, .. } => {
                let mut expr_type = ExpressionKind::ImplConst;
                for component in components {
                    expr_type = expr_type.max(self.type_of(*component))
                }
                expr_type
            }
            Expression::Splat { value, .. } => self.type_of(value),
            Expression::AccessIndex { base, .. } => self.type_of(base),
            Expression::Access { base, index } => self.type_of(base).max(self.type_of(index)),
            Expression::Swizzle { vector, .. } => self.type_of(vector),
            Expression::Unary { expr, .. } => self.type_of(expr),
            Expression::Binary { left, right, .. } => self
                .type_of(left)
                .max(self.type_of(right))
                .max(ExpressionKind::Const),
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self
                .type_of(arg)
                .max(
                    arg1.map(|arg| self.type_of(arg))
                        .unwrap_or(ExpressionKind::Const),
                )
                .max(
                    arg2.map(|arg| self.type_of(arg))
                        .unwrap_or(ExpressionKind::Const),
                )
                .max(
                    arg3.map(|arg| self.type_of(arg))
                        .unwrap_or(ExpressionKind::Const),
                )
                .max(
                    if matches!(
                        fun,
                        Mf::Dot
                            | Mf::Outer
                            | Mf::Cross
                            | Mf::Distance
                            | Mf::Length
                            | Mf::Normalize
                            | Mf::FaceForward
                            | Mf::Reflect
                            | Mf::Refract
                            | Mf::Ldexp
                            | Mf::Modf
                            | Mf::Mix
                            | Mf::Frexp
                    ) {
                        ExpressionKind::Const
                    } else {
                        ExpressionKind::ImplConst
                    },
                ),
            Expression::As { convert, expr, .. } => self.type_of(expr).max(if convert.is_some() {
                ExpressionKind::ImplConst
            } else {
                ExpressionKind::Const
            }),
            Expression::Select {
                condition,
                accept,
                reject,
            } => self
                .type_of(condition)
                .max(self.type_of(accept))
                .max(self.type_of(reject))
                .max(ExpressionKind::Const),
            Expression::Relational { argument, .. } => self.type_of(argument),
            Expression::ArrayLength(expr) => self.type_of(expr),
            _ => ExpressionKind::Runtime,
        }
    }
}

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
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
    #[error("Cannot call arrayLength on array sized by override-expression")]
    ArrayLengthOverridden,
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
    #[error("Constants don't support subgroup expressions")]
    SubgroupExpression,
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
    #[error("Can't use pipeline-overridable constants in const-expressions")]
    Override,
    #[error("Unexpected runtime-expression")]
    RuntimeExpr,
    #[error("Unexpected override-expression")]
    OverrideExpr,
}

impl<'a> ConstantEvaluator<'a> {
    /// Return a [`ConstantEvaluator`] that will add expressions to `module`'s
    /// constant expression arena.
    ///
    /// Report errors according to WGSL's rules for constant evaluation.
    pub fn for_wgsl_module(
        module: &'a mut crate::Module,
        global_expression_kind_tracker: &'a mut ExpressionKindTracker,
        in_override_ctx: bool,
    ) -> Self {
        Self::for_module(
            Behavior::Wgsl(if in_override_ctx {
                WgslRestrictions::Override
            } else {
                WgslRestrictions::Const(None)
            }),
            module,
            global_expression_kind_tracker,
        )
    }

    /// Return a [`ConstantEvaluator`] that will add expressions to `module`'s
    /// constant expression arena.
    ///
    /// Report errors according to GLSL's rules for constant evaluation.
    pub fn for_glsl_module(
        module: &'a mut crate::Module,
        global_expression_kind_tracker: &'a mut ExpressionKindTracker,
    ) -> Self {
        Self::for_module(
            Behavior::Glsl(GlslRestrictions::Const),
            module,
            global_expression_kind_tracker,
        )
    }

    fn for_module(
        behavior: Behavior<'a>,
        module: &'a mut crate::Module,
        global_expression_kind_tracker: &'a mut ExpressionKindTracker,
    ) -> Self {
        Self {
            behavior,
            types: &mut module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            expressions: &mut module.global_expressions,
            expression_kind_tracker: global_expression_kind_tracker,
        }
    }

    /// Return a [`ConstantEvaluator`] that will add expressions to `function`'s
    /// expression arena.
    ///
    /// Report errors according to WGSL's rules for constant evaluation.
    pub fn for_wgsl_function(
        module: &'a mut crate::Module,
        expressions: &'a mut Arena<Expression>,
        local_expression_kind_tracker: &'a mut ExpressionKindTracker,
        emitter: &'a mut super::Emitter,
        block: &'a mut crate::Block,
        is_const: bool,
    ) -> Self {
        let local_data = FunctionLocalData {
            global_expressions: &module.global_expressions,
            emitter,
            block,
        };
        Self {
            behavior: Behavior::Wgsl(if is_const {
                WgslRestrictions::Const(Some(local_data))
            } else {
                WgslRestrictions::Runtime(local_data)
            }),
            types: &mut module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            expressions,
            expression_kind_tracker: local_expression_kind_tracker,
        }
    }

    /// Return a [`ConstantEvaluator`] that will add expressions to `function`'s
    /// expression arena.
    ///
    /// Report errors according to GLSL's rules for constant evaluation.
    pub fn for_glsl_function(
        module: &'a mut crate::Module,
        expressions: &'a mut Arena<Expression>,
        local_expression_kind_tracker: &'a mut ExpressionKindTracker,
        emitter: &'a mut super::Emitter,
        block: &'a mut crate::Block,
    ) -> Self {
        Self {
            behavior: Behavior::Glsl(GlslRestrictions::Runtime(FunctionLocalData {
                global_expressions: &module.global_expressions,
                emitter,
                block,
            })),
            types: &mut module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            expressions,
            expression_kind_tracker: local_expression_kind_tracker,
        }
    }

    pub fn to_ctx(&self) -> crate::proc::GlobalCtx {
        crate::proc::GlobalCtx {
            types: self.types,
            constants: self.constants,
            overrides: self.overrides,
            global_expressions: match self.function_local_data() {
                Some(data) => data.global_expressions,
                None => self.expressions,
            },
        }
    }

    fn check(&self, expr: Handle<Expression>) -> Result<(), ConstantEvaluatorError> {
        if !self.expression_kind_tracker.is_const(expr) {
            log::debug!("check: SubexpressionsAreNotConstant");
            return Err(ConstantEvaluatorError::SubexpressionsAreNotConstant);
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
                if let Some(function_local_data) = self.function_local_data() {
                    // Deep-copy the constant's value into our arena.
                    self.copy_from(
                        self.constants[c].init,
                        function_local_data.global_expressions,
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
    /// If `expr`'s value cannot be determined at compile time, and `self` is
    /// contributing to some function's expression arena, then append `expr` to
    /// that arena unchanged (and thus unevaluated). Otherwise, `self` must be
    /// contributing to the module's constant expression arena; since `expr`'s
    /// value is not a constant, return an error.
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
        expr: Expression,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.expression_kind_tracker.type_of_with_expr(&expr) {
            ExpressionKind::ImplConst => self.try_eval_and_append_impl(&expr, span),
            ExpressionKind::Const => {
                let eval_result = self.try_eval_and_append_impl(&expr, span);
                // We should be able to evaluate `Const` expressions at this
                // point. If we failed to, then that probably means we just
                // haven't implemented that part of constant evaluation. Work
                // around this by simply emitting it as a run-time expression.
                if self.behavior.has_runtime_restrictions()
                    && matches!(
                        eval_result,
                        Err(ConstantEvaluatorError::NotImplemented(_)
                            | ConstantEvaluatorError::InvalidBinaryOpArgs,)
                    )
                {
                    Ok(self.append_expr(expr, span, ExpressionKind::Runtime))
                } else {
                    eval_result
                }
            }
            ExpressionKind::Override => match self.behavior {
                Behavior::Wgsl(WgslRestrictions::Override | WgslRestrictions::Runtime(_)) => {
                    Ok(self.append_expr(expr, span, ExpressionKind::Override))
                }
                Behavior::Wgsl(WgslRestrictions::Const(_)) => {
                    Err(ConstantEvaluatorError::OverrideExpr)
                }
                Behavior::Glsl(_) => {
                    unreachable!()
                }
            },
            ExpressionKind::Runtime => {
                if self.behavior.has_runtime_restrictions() {
                    Ok(self.append_expr(expr, span, ExpressionKind::Runtime))
                } else {
                    Err(ConstantEvaluatorError::RuntimeExpr)
                }
            }
        }
    }

    /// Is the [`Self::expressions`] arena the global module expression arena?
    const fn is_global_arena(&self) -> bool {
        matches!(
            self.behavior,
            Behavior::Wgsl(WgslRestrictions::Const(None) | WgslRestrictions::Override)
                | Behavior::Glsl(GlslRestrictions::Const)
        )
    }

    const fn function_local_data(&self) -> Option<&FunctionLocalData<'a>> {
        match self.behavior {
            Behavior::Wgsl(
                WgslRestrictions::Runtime(ref function_local_data)
                | WgslRestrictions::Const(Some(ref function_local_data)),
            )
            | Behavior::Glsl(GlslRestrictions::Runtime(ref function_local_data)) => {
                Some(function_local_data)
            }
            _ => None,
        }
    }

    fn try_eval_and_append_impl(
        &mut self,
        expr: &Expression,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        log::trace!("try_eval_and_append: {:?}", expr);
        match *expr {
            Expression::Constant(c) if self.is_global_arena() => {
                // "See through" the constant and use its initializer.
                // This is mainly done to avoid having constants pointing to other constants.
                Ok(self.constants[c].init)
            }
            Expression::Override(_) => Err(ConstantEvaluatorError::Override),
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
                Behavior::Wgsl(_) => Err(ConstantEvaluatorError::ArrayLength),
                Behavior::Glsl(_) => {
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
            Expression::SubgroupBallotResult { .. } => {
                Err(ConstantEvaluatorError::SubgroupExpression)
            }
            Expression::SubgroupOperationResult { .. } => {
                Err(ConstantEvaluatorError::SubgroupExpression)
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
            TypeInner::Vector { size: _, scalar } => Ok(self.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Vector { size, scalar },
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

        // NOTE: We try to match the declaration order of `MathFunction` here.
        match fun {
            // comparison
            crate::MathFunction::Abs => {
                component_wise_scalar(self, span, [arg], |args| match args {
                    Scalar::AbstractFloat([e]) => Ok(Scalar::AbstractFloat([e.abs()])),
                    Scalar::F32([e]) => Ok(Scalar::F32([e.abs()])),
                    Scalar::AbstractInt([e]) => Ok(Scalar::AbstractInt([e.abs()])),
                    Scalar::I32([e]) => Ok(Scalar::I32([e.wrapping_abs()])),
                    Scalar::U32([e]) => Ok(Scalar::U32([e])), // TODO: just re-use the expression, ezpz
                    Scalar::I64([e]) => Ok(Scalar::I64([e.wrapping_abs()])),
                    Scalar::U64([e]) => Ok(Scalar::U64([e])),
                })
            }
            crate::MathFunction::Min => {
                component_wise_scalar!(self, span, [arg, arg1.unwrap()], |e1, e2| {
                    Ok([e1.min(e2)])
                })
            }
            crate::MathFunction::Max => {
                component_wise_scalar!(self, span, [arg, arg1.unwrap()], |e1, e2| {
                    Ok([e1.max(e2)])
                })
            }
            crate::MathFunction::Clamp => {
                component_wise_scalar!(
                    self,
                    span,
                    [arg, arg1.unwrap(), arg2.unwrap()],
                    |e, low, high| {
                        if low > high {
                            Err(ConstantEvaluatorError::InvalidClamp)
                        } else {
                            Ok([e.clamp(low, high)])
                        }
                    }
                )
            }
            crate::MathFunction::Saturate => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.clamp(0., 1.)]) })
            }

            // trigonometry
            crate::MathFunction::Cos => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.cos()]) })
            }
            crate::MathFunction::Cosh => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.cosh()]) })
            }
            crate::MathFunction::Sin => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.sin()]) })
            }
            crate::MathFunction::Sinh => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.sinh()]) })
            }
            crate::MathFunction::Tan => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.tan()]) })
            }
            crate::MathFunction::Tanh => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.tanh()]) })
            }
            crate::MathFunction::Acos => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.acos()]) })
            }
            crate::MathFunction::Asin => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.asin()]) })
            }
            crate::MathFunction::Atan => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.atan()]) })
            }
            crate::MathFunction::Asinh => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.asinh()]) })
            }
            crate::MathFunction::Acosh => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.acosh()]) })
            }
            crate::MathFunction::Atanh => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.atanh()]) })
            }
            crate::MathFunction::Radians => {
                component_wise_float!(self, span, [arg], |e1| { Ok([e1.to_radians()]) })
            }
            crate::MathFunction::Degrees => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.to_degrees()]) })
            }

            // decomposition
            crate::MathFunction::Ceil => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.ceil()]) })
            }
            crate::MathFunction::Floor => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.floor()]) })
            }
            crate::MathFunction::Round => {
                // TODO: this hit stable on 1.77, but MSRV hasn't caught up yet
                // This polyfill is shamelessly [~~stolen from~~ inspired by `ndarray-image`][polyfill source],
                // which has licensing compatible with ours. See also
                // <https://github.com/rust-lang/rust/issues/96710>.
                //
                // [polyfill source]: https://github.com/imeka/ndarray-ndimage/blob/8b14b4d6ecfbc96a8a052f802e342a7049c68d8f/src/lib.rs#L98
                fn round_ties_even(x: f64) -> f64 {
                    let i = x as i64;
                    let f = (x - i as f64).abs();
                    if f == 0.5 {
                        if i & 1 == 1 {
                            // -1.5, 1.5, 3.5, ...
                            (x.abs() + 0.5).copysign(x)
                        } else {
                            (x.abs() - 0.5).copysign(x)
                        }
                    } else {
                        x.round()
                    }
                }
                component_wise_float(self, span, [arg], |e| match e {
                    Float::Abstract([e]) => Ok(Float::Abstract([round_ties_even(e)])),
                    Float::F32([e]) => Ok(Float::F32([(round_ties_even(e as f64) as f32)])),
                })
            }
            crate::MathFunction::Fract => {
                component_wise_float!(self, span, [arg], |e| {
                    // N.B., Rust's definition of `fract` is `e - e.trunc()`, so we can't use that
                    // here.
                    Ok([e - e.floor()])
                })
            }
            crate::MathFunction::Trunc => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.trunc()]) })
            }

            // exponent
            crate::MathFunction::Exp => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.exp()]) })
            }
            crate::MathFunction::Exp2 => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.exp2()]) })
            }
            crate::MathFunction::Log => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.ln()]) })
            }
            crate::MathFunction::Log2 => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.log2()]) })
            }
            crate::MathFunction::Pow => {
                component_wise_float!(self, span, [arg, arg1.unwrap()], |e1, e2| {
                    Ok([e1.powf(e2)])
                })
            }

            // computational
            crate::MathFunction::Sign => {
                component_wise_signed!(self, span, [arg], |e| { Ok([e.signum()]) })
            }
            crate::MathFunction::Fma => {
                component_wise_float!(
                    self,
                    span,
                    [arg, arg1.unwrap(), arg2.unwrap()],
                    |e1, e2, e3| { Ok([e1.mul_add(e2, e3)]) }
                )
            }
            crate::MathFunction::Step => {
                component_wise_float!(self, span, [arg, arg1.unwrap()], |edge, x| {
                    Ok([if edge <= x { 1.0 } else { 0.0 }])
                })
            }
            crate::MathFunction::Sqrt => {
                component_wise_float!(self, span, [arg], |e| { Ok([e.sqrt()]) })
            }
            crate::MathFunction::InverseSqrt => {
                component_wise_float!(self, span, [arg], |e| { Ok([1. / e.sqrt()]) })
            }

            // bits
            crate::MathFunction::CountTrailingZeros => {
                component_wise_concrete_int!(self, span, [arg], |e| {
                    #[allow(clippy::useless_conversion)]
                    Ok([e
                        .trailing_zeros()
                        .try_into()
                        .expect("bit count overflowed 32 bits, somehow!?")])
                })
            }
            crate::MathFunction::CountLeadingZeros => {
                component_wise_concrete_int!(self, span, [arg], |e| {
                    #[allow(clippy::useless_conversion)]
                    Ok([e
                        .leading_zeros()
                        .try_into()
                        .expect("bit count overflowed 32 bits, somehow!?")])
                })
            }
            crate::MathFunction::CountOneBits => {
                component_wise_concrete_int!(self, span, [arg], |e| {
                    #[allow(clippy::useless_conversion)]
                    Ok([e
                        .count_ones()
                        .try_into()
                        .expect("bit count overflowed 32 bits, somehow!?")])
                })
            }
            crate::MathFunction::ReverseBits => {
                component_wise_concrete_int!(self, span, [arg], |e| { Ok([e.reverse_bits()]) })
            }
            crate::MathFunction::FirstTrailingBit => {
                component_wise_concrete_int(self, span, [arg], |ci| Ok(first_trailing_bit(ci)))
            }
            crate::MathFunction::FirstLeadingBit => {
                component_wise_concrete_int(self, span, [arg], |ci| Ok(first_leading_bit(ci)))
            }

            fun => Err(ConstantEvaluatorError::NotImplemented(format!(
                "{fun:?} built-in function"
            ))),
        }
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
                        ArraySize::Constant(len) => {
                            let expr = Expression::Literal(Literal::U32(len.get()));
                            self.register_evaluated_expr(expr, span)
                        }
                        ArraySize::Pending(_) => Err(ConstantEvaluatorError::ArrayLengthOverridden),
                        ArraySize::Dynamic => Err(ConstantEvaluatorError::ArrayLengthDynamic),
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
                    TypeInner::Scalar(crate::Scalar {
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
                        Literal::F64(_) | Literal::I64(_) | Literal::U64(_) => {
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
                        Literal::F64(_) | Literal::I64(_) | Literal::U64(_) => {
                            return make_error();
                        }
                        Literal::AbstractInt(v) => u32::try_from_abstract(v)?,
                        Literal::AbstractFloat(v) => u32::try_from_abstract(v)?,
                    }),
                    Sc::I64 => Literal::I64(match literal {
                        Literal::I32(v) => v as i64,
                        Literal::U32(v) => v as i64,
                        Literal::F32(v) => v as i64,
                        Literal::Bool(v) => v as i64,
                        Literal::F64(v) => v as i64,
                        Literal::I64(v) => v,
                        Literal::U64(v) => v as i64,
                        Literal::AbstractInt(v) => i64::try_from_abstract(v)?,
                        Literal::AbstractFloat(v) => i64::try_from_abstract(v)?,
                    }),
                    Sc::U64 => Literal::U64(match literal {
                        Literal::I32(v) => v as u64,
                        Literal::U32(v) => v as u64,
                        Literal::F32(v) => v as u64,
                        Literal::Bool(v) => v as u64,
                        Literal::F64(v) => v as u64,
                        Literal::I64(v) => v as u64,
                        Literal::U64(v) => v,
                        Literal::AbstractInt(v) => u64::try_from_abstract(v)?,
                        Literal::AbstractFloat(v) => u64::try_from_abstract(v)?,
                    }),
                    Sc::F32 => Literal::F32(match literal {
                        Literal::I32(v) => v as f32,
                        Literal::U32(v) => v as f32,
                        Literal::F32(v) => v,
                        Literal::Bool(v) => v as u32 as f32,
                        Literal::F64(_) | Literal::I64(_) | Literal::U64(_) => {
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
                        Literal::I64(_) | Literal::U64(_) => return make_error(),
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
                        | Literal::U64(_)
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

        let TypeInner::Array {
            base: _,
            size,
            stride: _,
        } = self.types[ty].inner
        else {
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
                    Literal::I64(v) => Literal::I64(v.wrapping_neg()),
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
                    Literal::I64(v) => Literal::I64(!v),
                    Literal::U32(v) => Literal::U32(!v),
                    Literal::U64(v) => Literal::U64(!v),
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
                            BinaryOperator::ShiftLeft => {
                                if (if a.is_negative() { !a } else { a }).leading_zeros() <= b {
                                    return Err(ConstantEvaluatorError::Overflow("<<".to_string()));
                                }
                                a.checked_shl(b)
                                    .ok_or(ConstantEvaluatorError::ShiftedMoreThan32Bits)?
                            }
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
                                .checked_mul(
                                    1u32.checked_shl(b)
                                        .ok_or(ConstantEvaluatorError::ShiftedMoreThan32Bits)?,
                                )
                                .ok_or(ConstantEvaluatorError::Overflow("<<".to_string()))?,
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

        Ok(self.append_expr(expr, span, ExpressionKind::Const))
    }

    fn append_expr(
        &mut self,
        expr: Expression,
        span: Span,
        expr_type: ExpressionKind,
    ) -> Handle<Expression> {
        let h = match self.behavior {
            Behavior::Wgsl(
                WgslRestrictions::Runtime(ref mut function_local_data)
                | WgslRestrictions::Const(Some(ref mut function_local_data)),
            )
            | Behavior::Glsl(GlslRestrictions::Runtime(ref mut function_local_data)) => {
                let is_running = function_local_data.emitter.is_running();
                let needs_pre_emit = expr.needs_pre_emit();
                if is_running && needs_pre_emit {
                    function_local_data
                        .block
                        .extend(function_local_data.emitter.finish(self.expressions));
                    let h = self.expressions.append(expr, span);
                    function_local_data.emitter.start(self.expressions);
                    h
                } else {
                    self.expressions.append(expr, span)
                }
            }
            _ => self.expressions.append(expr, span),
        };
        self.expression_kind_tracker.insert(h, expr_type);
        h
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

fn first_trailing_bit(concrete_int: ConcreteInt<1>) -> ConcreteInt<1> {
    // NOTE: Bit indices for this built-in start at 0 at the "right" (or LSB). For example, a value
    // of 1 means the least significant bit is set. Therefore, an input of `0x[80 00…]` would
    // return a right-to-left bit index of 0.
    let trailing_zeros_to_bit_idx = |e: u32| -> u32 {
        match e {
            idx @ 0..=31 => idx,
            32 => u32::MAX,
            _ => unreachable!(),
        }
    };
    match concrete_int {
        ConcreteInt::U32([e]) => ConcreteInt::U32([trailing_zeros_to_bit_idx(e.trailing_zeros())]),
        ConcreteInt::I32([e]) => {
            ConcreteInt::I32([trailing_zeros_to_bit_idx(e.trailing_zeros()) as i32])
        }
    }
}

#[test]
fn first_trailing_bit_smoke() {
    assert_eq!(
        first_trailing_bit(ConcreteInt::I32([0])),
        ConcreteInt::I32([-1])
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::I32([1])),
        ConcreteInt::I32([0])
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::I32([2])),
        ConcreteInt::I32([1])
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::I32([-1])),
        ConcreteInt::I32([0]),
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::I32([i32::MIN])),
        ConcreteInt::I32([31]),
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::I32([i32::MAX])),
        ConcreteInt::I32([0]),
    );
    for idx in 0..32 {
        assert_eq!(
            first_trailing_bit(ConcreteInt::I32([1 << idx])),
            ConcreteInt::I32([idx])
        )
    }

    assert_eq!(
        first_trailing_bit(ConcreteInt::U32([0])),
        ConcreteInt::U32([u32::MAX])
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::U32([1])),
        ConcreteInt::U32([0])
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::U32([2])),
        ConcreteInt::U32([1])
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::U32([1 << 31])),
        ConcreteInt::U32([31]),
    );
    assert_eq!(
        first_trailing_bit(ConcreteInt::U32([u32::MAX])),
        ConcreteInt::U32([0]),
    );
    for idx in 0..32 {
        assert_eq!(
            first_trailing_bit(ConcreteInt::U32([1 << idx])),
            ConcreteInt::U32([idx])
        )
    }
}

fn first_leading_bit(concrete_int: ConcreteInt<1>) -> ConcreteInt<1> {
    // NOTE: Bit indices for this built-in start at 0 at the "right" (or LSB). For example, 1 means
    // the least significant bit is set. Therefore, an input of 1 would return a right-to-left bit
    // index of 0.
    let rtl_to_ltr_bit_idx = |e: u32| -> u32 {
        match e {
            idx @ 0..=31 => 31 - idx,
            32 => u32::MAX,
            _ => unreachable!(),
        }
    };
    match concrete_int {
        ConcreteInt::I32([e]) => ConcreteInt::I32([{
            let rtl_bit_index = if e.is_negative() {
                e.leading_ones()
            } else {
                e.leading_zeros()
            };
            rtl_to_ltr_bit_idx(rtl_bit_index) as i32
        }]),
        ConcreteInt::U32([e]) => ConcreteInt::U32([rtl_to_ltr_bit_idx(e.leading_zeros())]),
    }
}

#[test]
fn first_leading_bit_smoke() {
    assert_eq!(
        first_leading_bit(ConcreteInt::I32([-1])),
        ConcreteInt::I32([-1])
    );
    assert_eq!(
        first_leading_bit(ConcreteInt::I32([0])),
        ConcreteInt::I32([-1])
    );
    assert_eq!(
        first_leading_bit(ConcreteInt::I32([1])),
        ConcreteInt::I32([0])
    );
    assert_eq!(
        first_leading_bit(ConcreteInt::I32([-2])),
        ConcreteInt::I32([0])
    );
    assert_eq!(
        first_leading_bit(ConcreteInt::I32([1234 + 4567])),
        ConcreteInt::I32([12])
    );
    assert_eq!(
        first_leading_bit(ConcreteInt::I32([i32::MAX])),
        ConcreteInt::I32([30])
    );
    assert_eq!(
        first_leading_bit(ConcreteInt::I32([i32::MIN])),
        ConcreteInt::I32([30])
    );
    // NOTE: Ignore the sign bit, which is a separate (above) case.
    for idx in 0..(32 - 1) {
        assert_eq!(
            first_leading_bit(ConcreteInt::I32([1 << idx])),
            ConcreteInt::I32([idx])
        );
    }
    for idx in 1..(32 - 1) {
        assert_eq!(
            first_leading_bit(ConcreteInt::I32([-(1 << idx)])),
            ConcreteInt::I32([idx - 1])
        );
    }

    assert_eq!(
        first_leading_bit(ConcreteInt::U32([0])),
        ConcreteInt::U32([u32::MAX])
    );
    assert_eq!(
        first_leading_bit(ConcreteInt::U32([1])),
        ConcreteInt::U32([0])
    );
    assert_eq!(
        first_leading_bit(ConcreteInt::U32([u32::MAX])),
        ConcreteInt::U32([31])
    );
    for idx in 0..32 {
        assert_eq!(
            first_leading_bit(ConcreteInt::U32([1 << idx])),
            ConcreteInt::U32([idx])
        )
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

impl TryFromAbstract<i64> for u64 {
    fn try_from_abstract(value: i64) -> Result<u64, ConstantEvaluatorError> {
        u64::try_from(value).map_err(|_| ConstantEvaluatorError::AutomaticConversionLossy {
            value: format!("{value:?}"),
            to_type: "u64",
        })
    }
}

impl TryFromAbstract<i64> for i64 {
    fn try_from_abstract(value: i64) -> Result<i64, ConstantEvaluatorError> {
        Ok(value)
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

impl TryFromAbstract<f64> for i64 {
    fn try_from_abstract(_: f64) -> Result<Self, ConstantEvaluatorError> {
        Err(ConstantEvaluatorError::AutomaticConversionFloatToInt { to_type: "i64" })
    }
}

impl TryFromAbstract<f64> for u64 {
    fn try_from_abstract(_: f64) -> Result<Self, ConstantEvaluatorError> {
        Err(ConstantEvaluatorError::AutomaticConversionFloatToInt { to_type: "u64" })
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::{
        Arena, Constant, Expression, Literal, ScalarKind, Type, TypeInner, UnaryOperator,
        UniqueArena, VectorSize,
    };

    use super::{Behavior, ConstantEvaluator, ExpressionKindTracker, WgslRestrictions};

    #[test]
    fn unary_op() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let overrides = Arena::new();
        let mut global_expressions = Arena::new();

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
                ty: scalar_ty,
                init: global_expressions
                    .append(Expression::Literal(Literal::I32(4)), Default::default()),
            },
            Default::default(),
        );

        let h1 = constants.append(
            Constant {
                name: None,
                ty: scalar_ty,
                init: global_expressions
                    .append(Expression::Literal(Literal::I32(8)), Default::default()),
            },
            Default::default(),
        );

        let vec_h = constants.append(
            Constant {
                name: None,
                ty: vec_ty,
                init: global_expressions.append(
                    Expression::Compose {
                        ty: vec_ty,
                        components: vec![constants[h].init, constants[h1].init],
                    },
                    Default::default(),
                ),
            },
            Default::default(),
        );

        let expr = global_expressions.append(Expression::Constant(h), Default::default());
        let expr1 = global_expressions.append(Expression::Constant(vec_h), Default::default());

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

        let expression_kind_tracker = &mut ExpressionKindTracker::from_arena(&global_expressions);
        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl(WgslRestrictions::Const(None)),
            types: &mut types,
            constants: &constants,
            overrides: &overrides,
            expressions: &mut global_expressions,
            expression_kind_tracker,
        };

        let res1 = solver
            .try_eval_and_append(expr2, Default::default())
            .unwrap();
        let res2 = solver
            .try_eval_and_append(expr3, Default::default())
            .unwrap();
        let res3 = solver
            .try_eval_and_append(expr4, Default::default())
            .unwrap();

        assert_eq!(
            global_expressions[res1],
            Expression::Literal(Literal::I32(-4))
        );

        assert_eq!(
            global_expressions[res2],
            Expression::Literal(Literal::I32(!4))
        );

        let res3_inner = &global_expressions[res3];

        match *res3_inner {
            Expression::Compose {
                ref ty,
                ref components,
            } => {
                assert_eq!(*ty, vec_ty);
                let mut components_iter = components.iter().copied();
                assert_eq!(
                    global_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::I32(!4))
                );
                assert_eq!(
                    global_expressions[components_iter.next().unwrap()],
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
        let overrides = Arena::new();
        let mut global_expressions = Arena::new();

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
                ty: scalar_ty,
                init: global_expressions
                    .append(Expression::Literal(Literal::I32(4)), Default::default()),
            },
            Default::default(),
        );

        let expr = global_expressions.append(Expression::Constant(h), Default::default());

        let root = Expression::As {
            expr,
            kind: ScalarKind::Bool,
            convert: Some(crate::BOOL_WIDTH),
        };

        let expression_kind_tracker = &mut ExpressionKindTracker::from_arena(&global_expressions);
        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl(WgslRestrictions::Const(None)),
            types: &mut types,
            constants: &constants,
            overrides: &overrides,
            expressions: &mut global_expressions,
            expression_kind_tracker,
        };

        let res = solver
            .try_eval_and_append(root, Default::default())
            .unwrap();

        assert_eq!(
            global_expressions[res],
            Expression::Literal(Literal::Bool(true))
        );
    }

    #[test]
    fn access() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let overrides = Arena::new();
        let mut global_expressions = Arena::new();

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
            let h = global_expressions.append(
                Expression::Literal(Literal::F32(i as f32)),
                Default::default(),
            );

            vec1_components.push(h)
        }

        for i in 3..6 {
            let h = global_expressions.append(
                Expression::Literal(Literal::F32(i as f32)),
                Default::default(),
            );

            vec2_components.push(h)
        }

        let vec1 = constants.append(
            Constant {
                name: None,
                ty: vec_ty,
                init: global_expressions.append(
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
                ty: vec_ty,
                init: global_expressions.append(
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
                ty: matrix_ty,
                init: global_expressions.append(
                    Expression::Compose {
                        ty: matrix_ty,
                        components: vec![constants[vec1].init, constants[vec2].init],
                    },
                    Default::default(),
                ),
            },
            Default::default(),
        );

        let base = global_expressions.append(Expression::Constant(h), Default::default());

        let expression_kind_tracker = &mut ExpressionKindTracker::from_arena(&global_expressions);
        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl(WgslRestrictions::Const(None)),
            types: &mut types,
            constants: &constants,
            overrides: &overrides,
            expressions: &mut global_expressions,
            expression_kind_tracker,
        };

        let root1 = Expression::AccessIndex { base, index: 1 };

        let res1 = solver
            .try_eval_and_append(root1, Default::default())
            .unwrap();

        let root2 = Expression::AccessIndex {
            base: res1,
            index: 2,
        };

        let res2 = solver
            .try_eval_and_append(root2, Default::default())
            .unwrap();

        match global_expressions[res1] {
            Expression::Compose {
                ref ty,
                ref components,
            } => {
                assert_eq!(*ty, vec_ty);
                let mut components_iter = components.iter().copied();
                assert_eq!(
                    global_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::F32(3.))
                );
                assert_eq!(
                    global_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::F32(4.))
                );
                assert_eq!(
                    global_expressions[components_iter.next().unwrap()],
                    Expression::Literal(Literal::F32(5.))
                );
                assert!(components_iter.next().is_none());
            }
            _ => panic!("Expected vector"),
        }

        assert_eq!(
            global_expressions[res2],
            Expression::Literal(Literal::F32(5.))
        );
    }

    #[test]
    fn compose_of_constants() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let overrides = Arena::new();
        let mut global_expressions = Arena::new();

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
                ty: i32_ty,
                init: global_expressions
                    .append(Expression::Literal(Literal::I32(4)), Default::default()),
            },
            Default::default(),
        );

        let h_expr = global_expressions.append(Expression::Constant(h), Default::default());

        let expression_kind_tracker = &mut ExpressionKindTracker::from_arena(&global_expressions);
        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl(WgslRestrictions::Const(None)),
            types: &mut types,
            constants: &constants,
            overrides: &overrides,
            expressions: &mut global_expressions,
            expression_kind_tracker,
        };

        let solved_compose = solver
            .try_eval_and_append(
                Expression::Compose {
                    ty: vec2_i32_ty,
                    components: vec![h_expr, h_expr],
                },
                Default::default(),
            )
            .unwrap();
        let solved_negate = solver
            .try_eval_and_append(
                Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: solved_compose,
                },
                Default::default(),
            )
            .unwrap();

        let pass = match global_expressions[solved_negate] {
            Expression::Compose { ty, ref components } => {
                ty == vec2_i32_ty
                    && components.iter().all(|&component| {
                        let component = &global_expressions[component];
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
        let overrides = Arena::new();
        let mut global_expressions = Arena::new();

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
                ty: i32_ty,
                init: global_expressions
                    .append(Expression::Literal(Literal::I32(4)), Default::default()),
            },
            Default::default(),
        );

        let h_expr = global_expressions.append(Expression::Constant(h), Default::default());

        let expression_kind_tracker = &mut ExpressionKindTracker::from_arena(&global_expressions);
        let mut solver = ConstantEvaluator {
            behavior: Behavior::Wgsl(WgslRestrictions::Const(None)),
            types: &mut types,
            constants: &constants,
            overrides: &overrides,
            expressions: &mut global_expressions,
            expression_kind_tracker,
        };

        let solved_compose = solver
            .try_eval_and_append(
                Expression::Splat {
                    size: VectorSize::Bi,
                    value: h_expr,
                },
                Default::default(),
            )
            .unwrap();
        let solved_negate = solver
            .try_eval_and_append(
                Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: solved_compose,
                },
                Default::default(),
            )
            .unwrap();

        let pass = match global_expressions[solved_negate] {
            Expression::Compose { ty, ref components } => {
                ty == vec2_i32_ty
                    && components.iter().all(|&component| {
                        let component = &global_expressions[component];
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
