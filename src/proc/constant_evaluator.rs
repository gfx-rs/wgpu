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

#[derive(Debug)]
pub struct ConstantEvaluator<'a> {
    behavior: Behavior,
    types: &'a mut UniqueArena<Type>,
    constants: &'a Arena<Constant>,
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

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
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
    #[error("Constants don't support relational functions")]
    Relational,
    #[error("Constants don't support derivative functions")]
    Derivative,
    #[error("Constants don't support select expressions")]
    Select,
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
    #[error("Constants don't support bitcasts")]
    Bitcast,
    #[error("Constants don't support array length expressions")]
    ArrayLength,
    #[error("Cannot cast type")]
    InvalidCastArg,
    #[error("Cannot apply the unary op to the argument")]
    InvalidUnaryOpArg,
    #[error("Cannot apply the binary op to the arguments")]
    InvalidBinaryOpArgs,
    #[error("Cannot apply math function to type")]
    InvalidMathArg,
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
}

impl<'a> ConstantEvaluator<'a> {
    pub fn for_wgsl_module(module: &'a mut crate::Module) -> Self {
        Self::for_module(Behavior::Wgsl, module)
    }

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
                Ok(self.register_evaluated_expr(expr.clone(), span))
            }
            Expression::Compose { ty, ref components } => {
                let components = components
                    .iter()
                    .map(|component| self.check_and_get(*component))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(self.register_evaluated_expr(Expression::Compose { ty, components }, span))
            }
            Expression::Splat { size, value } => {
                let value = self.check_and_get(value)?;
                Ok(self.register_evaluated_expr(Expression::Splat { size, value }, span))
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
                    Some(width) => self.cast(expr, kind, width, span),
                    None => match self.behavior {
                        Behavior::Wgsl => Err(ConstantEvaluatorError::NotImplemented(
                            "bitcast built-in function".into(),
                        )),
                        Behavior::Glsl => Err(ConstantEvaluatorError::Bitcast),
                    },
                }
            }
            Expression::Select { .. } => match self.behavior {
                Behavior::Wgsl => Err(ConstantEvaluatorError::NotImplemented(
                    "select built-in function".into(),
                )),
                Behavior::Glsl => Err(ConstantEvaluatorError::Select),
            },
            Expression::Relational { fun, .. } => match self.behavior {
                Behavior::Wgsl => Err(ConstantEvaluatorError::NotImplemented(format!(
                    "{fun:?} built-in function"
                ))),
                Behavior::Glsl => Err(ConstantEvaluatorError::Relational),
            },
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

    fn splat(
        &mut self,
        value: Handle<Expression>,
        size: crate::VectorSize,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.expressions[value] {
            Expression::Literal(literal) => {
                let kind = literal.scalar_kind();
                let width = literal.width();
                let ty = self.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Vector { size, kind, width },
                    },
                    span,
                );
                let expr = Expression::Compose {
                    ty,
                    components: vec![value; size as usize],
                };
                Ok(self.register_evaluated_expr(expr, span))
            }
            Expression::ZeroValue(ty) => {
                let inner = match self.types[ty].inner {
                    TypeInner::Scalar { kind, width } => TypeInner::Vector { size, kind, width },
                    _ => return Err(ConstantEvaluatorError::SplatScalarOnly),
                };
                let res_ty = self.types.insert(Type { name: None, inner }, span);
                let expr = Expression::ZeroValue(res_ty);
                Ok(self.register_evaluated_expr(expr, span))
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
            crate::TypeInner::Vector {
                size: _,
                kind,
                width,
            } => Ok(self.types.insert(
                Type {
                    name: None,
                    inner: crate::TypeInner::Vector { size, kind, width },
                },
                span,
            )),
            _ => Err(ConstantEvaluatorError::SwizzleVectorOnly),
        };

        match self.expressions[src_constant] {
            Expression::ZeroValue(ty) => {
                let dst_ty = get_dst_ty(ty)?;
                let expr = Expression::ZeroValue(dst_ty);
                Ok(self.register_evaluated_expr(expr, span))
            }
            Expression::Splat { value, .. } => {
                let expr = Expression::Splat { size, value };
                Ok(self.register_evaluated_expr(expr, span))
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
                Ok(self.register_evaluated_expr(expr, span))
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
        let const0 = &self.expressions[arg];
        let const1 = arg1.map(|arg| &self.expressions[arg]);
        let const2 = arg2.map(|arg| &self.expressions[arg]);
        let _const3 = arg3.map(|arg| &self.expressions[arg]);

        match fun {
            crate::MathFunction::Pow => {
                let literal = match (const0, const1.unwrap()) {
                    (&Expression::Literal(value0), &Expression::Literal(value1)) => {
                        match (value0, value1) {
                            (Literal::I32(a), Literal::I32(b)) => Literal::I32(a.pow(b as u32)),
                            (Literal::U32(a), Literal::U32(b)) => Literal::U32(a.pow(b)),
                            (Literal::F32(a), Literal::F32(b)) => Literal::F32(a.powf(b)),
                            _ => return Err(ConstantEvaluatorError::InvalidMathArg),
                        }
                    }
                    _ => return Err(ConstantEvaluatorError::InvalidMathArg),
                };

                let expr = Expression::Literal(literal);
                Ok(self.register_evaluated_expr(expr, span))
            }
            crate::MathFunction::Clamp => {
                let literal = match (const0, const1.unwrap(), const2.unwrap()) {
                    (
                        &Expression::Literal(value0),
                        &Expression::Literal(value1),
                        &Expression::Literal(value2),
                    ) => match (value0, value1, value2) {
                        (Literal::I32(a), Literal::I32(b), Literal::I32(c)) => {
                            Literal::I32(a.clamp(b, c))
                        }
                        (Literal::U32(a), Literal::U32(b), Literal::U32(c)) => {
                            Literal::U32(a.clamp(b, c))
                        }
                        (Literal::F32(a), Literal::F32(b), Literal::F32(c)) => {
                            Literal::F32(glsl_float_clamp(a, b, c))
                        }
                        _ => return Err(ConstantEvaluatorError::InvalidMathArg),
                    },
                    _ => {
                        return Err(ConstantEvaluatorError::NotImplemented(
                            "clamp built-in function with vector values".into(),
                        ))
                    }
                };

                let expr = Expression::Literal(literal);
                Ok(self.register_evaluated_expr(expr, span))
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
                        crate::ArraySize::Constant(len) => {
                            let expr = Expression::Literal(Literal::U32(len.get()));
                            Ok(self.register_evaluated_expr(expr, span))
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
                    Ok(self.register_evaluated_expr(Expression::ZeroValue(ty), span))
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
            Expression::Literal(Literal::U32(index)) => Ok(index as usize),
            _ => Err(ConstantEvaluatorError::InvalidAccessIndexTy),
        }
    }

    /// Transforms `Expression::ZeroValue` and `Expression::Splat` into either `Expression::Literal` or `Expression::Compose`
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

    fn eval_zero_value_impl(
        &mut self,
        ty: Handle<Type>,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        match self.types[ty].inner {
            TypeInner::Scalar { kind, width } => {
                let expr = Expression::Literal(
                    Literal::zero(kind, width)
                        .ok_or(ConstantEvaluatorError::TypeNotConstructible)?,
                );
                Ok(self.register_evaluated_expr(expr, span))
            }
            TypeInner::Vector { size, kind, width } => {
                let scalar_ty = self.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Scalar { kind, width },
                    },
                    span,
                );
                let el = self.eval_zero_value_impl(scalar_ty, span)?;
                let expr = Expression::Compose {
                    ty,
                    components: vec![el; size as usize],
                };
                Ok(self.register_evaluated_expr(expr, span))
            }
            TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                let vec_ty = self.types.insert(
                    Type {
                        name: None,
                        inner: TypeInner::Vector {
                            size: rows,
                            kind: ScalarKind::Float,
                            width,
                        },
                    },
                    span,
                );
                let el = self.eval_zero_value_impl(vec_ty, span)?;
                let expr = Expression::Compose {
                    ty,
                    components: vec![el; columns as usize],
                };
                Ok(self.register_evaluated_expr(expr, span))
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
                Ok(self.register_evaluated_expr(expr, span))
            }
            TypeInner::Struct { ref members, .. } => {
                let types: Vec<_> = members.iter().map(|m| m.ty).collect();
                let mut components = Vec::with_capacity(members.len());
                for ty in types {
                    components.push(self.eval_zero_value_impl(ty, span)?);
                }
                let expr = Expression::Compose { ty, components };
                Ok(self.register_evaluated_expr(expr, span))
            }
            _ => Err(ConstantEvaluatorError::TypeNotConstructible),
        }
    }

    fn cast(
        &mut self,
        expr: Handle<Expression>,
        kind: ScalarKind,
        target_width: crate::Bytes,
        span: Span,
    ) -> Result<Handle<Expression>, ConstantEvaluatorError> {
        let expr = self.eval_zero_value_and_splat(expr, span)?;

        let expr = match self.expressions[expr] {
            Expression::Literal(literal) => {
                let literal = match (kind, target_width) {
                    (ScalarKind::Sint, 4) => Literal::I32(match literal {
                        Literal::I32(v) => v,
                        Literal::U32(v) => v as i32,
                        Literal::F32(v) => v as i32,
                        Literal::Bool(v) => v as i32,
                        Literal::F64(_) => return Err(ConstantEvaluatorError::InvalidCastArg),
                    }),
                    (ScalarKind::Uint, 4) => Literal::U32(match literal {
                        Literal::I32(v) => v as u32,
                        Literal::U32(v) => v,
                        Literal::F32(v) => v as u32,
                        Literal::Bool(v) => v as u32,
                        Literal::F64(_) => return Err(ConstantEvaluatorError::InvalidCastArg),
                    }),
                    (ScalarKind::Float, 4) => Literal::F32(match literal {
                        Literal::I32(v) => v as f32,
                        Literal::U32(v) => v as f32,
                        Literal::F32(v) => v,
                        Literal::Bool(v) => v as u32 as f32,
                        Literal::F64(_) => return Err(ConstantEvaluatorError::InvalidCastArg),
                    }),
                    (ScalarKind::Bool, crate::BOOL_WIDTH) => Literal::Bool(match literal {
                        Literal::I32(v) => v != 0,
                        Literal::U32(v) => v != 0,
                        Literal::F32(v) => v != 0.0,
                        Literal::Bool(v) => v,
                        Literal::F64(_) => return Err(ConstantEvaluatorError::InvalidCastArg),
                    }),
                    _ => return Err(ConstantEvaluatorError::InvalidCastArg),
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
                        kind,
                        width: target_width,
                    },
                    TypeInner::Matrix { columns, rows, .. } => TypeInner::Matrix {
                        columns,
                        rows,
                        width: target_width,
                    },
                    _ => return Err(ConstantEvaluatorError::InvalidCastArg),
                };

                let mut components = src_components.clone();
                for component in &mut components {
                    *component = self.cast(*component, kind, target_width, span)?;
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
            _ => return Err(ConstantEvaluatorError::InvalidCastArg),
        };

        Ok(self.register_evaluated_expr(expr, span))
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
                    Literal::I32(v) => Literal::I32(-v),
                    Literal::F32(v) => Literal::F32(-v),
                    _ => return Err(ConstantEvaluatorError::InvalidUnaryOpArg),
                },
                UnaryOperator::Not => match value {
                    Literal::I32(v) => Literal::I32(!v),
                    Literal::U32(v) => Literal::U32(!v),
                    Literal::Bool(v) => Literal::Bool(!v),
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

        Ok(self.register_evaluated_expr(expr, span))
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
                            BinaryOperator::Add => a.wrapping_add(b),
                            BinaryOperator::Subtract => a.wrapping_sub(b),
                            BinaryOperator::Multiply => a.wrapping_mul(b),
                            BinaryOperator::Divide => a.checked_div(b).unwrap_or(0),
                            BinaryOperator::Modulo => a.checked_rem(b).unwrap_or(0),
                            BinaryOperator::And => a & b,
                            BinaryOperator::ExclusiveOr => a ^ b,
                            BinaryOperator::InclusiveOr => a | b,
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
                        (Literal::I32(a), Literal::U32(b)) => Literal::I32(match op {
                            BinaryOperator::ShiftLeft => a.wrapping_shl(b),
                            BinaryOperator::ShiftRight => a.wrapping_shr(b),
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
                        (Literal::U32(a), Literal::U32(b)) => Literal::U32(match op {
                            BinaryOperator::Add => a.wrapping_add(b),
                            BinaryOperator::Subtract => a.wrapping_sub(b),
                            BinaryOperator::Multiply => a.wrapping_mul(b),
                            BinaryOperator::Divide => a.checked_div(b).unwrap_or(0),
                            BinaryOperator::Modulo => a.checked_rem(b).unwrap_or(0),
                            BinaryOperator::And => a & b,
                            BinaryOperator::ExclusiveOr => a ^ b,
                            BinaryOperator::InclusiveOr => a | b,
                            BinaryOperator::ShiftLeft => a.wrapping_shl(b),
                            BinaryOperator::ShiftRight => a.wrapping_shr(b),
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
                        (Literal::F32(a), Literal::F32(b)) => Literal::F32(match op {
                            BinaryOperator::Add => a + b,
                            BinaryOperator::Subtract => a - b,
                            BinaryOperator::Multiply => a * b,
                            BinaryOperator::Divide => a / b,
                            BinaryOperator::Modulo => a - b * (a / b).floor(),
                            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
                        }),
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
            _ => return Err(ConstantEvaluatorError::InvalidBinaryOpArgs),
        };

        Ok(self.register_evaluated_expr(expr, span))
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
            | Expression::ZeroValue(_)) => Ok(self.register_evaluated_expr(expr.clone(), span)),
            Expression::Compose { ty, ref components } => {
                let mut components = components.clone();
                for component in &mut components {
                    *component = self.copy_from(*component, expressions)?;
                }
                Ok(self.register_evaluated_expr(Expression::Compose { ty, components }, span))
            }
            Expression::Splat { size, value } => {
                let value = self.copy_from(value, expressions)?;
                Ok(self.register_evaluated_expr(Expression::Splat { size, value }, span))
            }
            _ => {
                log::debug!("copy_from: SubexpressionsAreNotConstant");
                Err(ConstantEvaluatorError::SubexpressionsAreNotConstant)
            }
        }
    }

    fn register_evaluated_expr(&mut self, expr: Expression, span: Span) -> Handle<Expression> {
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
                h
            } else {
                let h = self.expressions.append(expr, span);
                expression_constness.insert(h);
                h
            }
        } else {
            self.expressions.append(expr, span)
        }
    }
}

/// Helper function to implement the GLSL `max` function for floats.
///
/// While Rust does provide a `f64::max` method, it has a different behavior than the
/// GLSL `max` for NaNs. In Rust, if any of the arguments is a NaN, then the other
/// is returned.
///
/// This leads to different results in the following example
/// ```
/// use std::cmp::max;
/// std::f64::NAN.max(1.0);
/// ```
///
/// Rust will return `1.0` while GLSL should return NaN.
fn glsl_float_max(x: f32, y: f32) -> f32 {
    if x < y {
        y
    } else {
        x
    }
}

/// Helper function to implement the GLSL `min` function for floats.
///
/// While Rust does provide a `f64::min` method, it has a different behavior than the
/// GLSL `min` for NaNs. In Rust, if any of the arguments is a NaN, then the other
/// is returned.
///
/// This leads to different results in the following example
/// ```
/// use std::cmp::min;
/// std::f64::NAN.min(1.0);
/// ```
///
/// Rust will return `1.0` while GLSL should return NaN.
fn glsl_float_min(x: f32, y: f32) -> f32 {
    if y < x {
        y
    } else {
        x
    }
}

/// Helper function to implement the GLSL `clamp` function for floats.
///
/// While Rust does provide a `f64::clamp` method, it panics if either
/// `min` or `max` are `NaN`s which is not the behavior specified by
/// the glsl specification.
fn glsl_float_clamp(value: f32, min: f32, max: f32) -> f32 {
    glsl_float_min(glsl_float_max(value, min), max)
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
    fn nan_handling() {
        assert!(super::glsl_float_max(f32::NAN, 2.0).is_nan());
        assert!(!super::glsl_float_max(2.0, f32::NAN).is_nan());

        assert!(super::glsl_float_min(f32::NAN, 2.0).is_nan());
        assert!(!super::glsl_float_min(2.0, f32::NAN).is_nan());

        assert!(super::glsl_float_clamp(f32::NAN, 1.0, 2.0).is_nan());
        assert!(!super::glsl_float_clamp(1.0, f32::NAN, 2.0).is_nan());
        assert!(!super::glsl_float_clamp(1.0, 2.0, f32::NAN).is_nan());
    }

    #[test]
    fn unary_op() {
        let mut types = UniqueArena::new();
        let mut constants = Arena::new();
        let mut const_expressions = Arena::new();

        let scalar_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                },
            },
            Default::default(),
        );

        let vec_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Bi,
                    kind: ScalarKind::Sint,
                    width: 4,
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
            op: UnaryOperator::Not,
            expr,
        };

        let expr4 = Expression::Unary {
            op: UnaryOperator::Not,
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
                inner: TypeInner::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
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
                    width: 4,
                },
            },
            Default::default(),
        );

        let vec_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    kind: ScalarKind::Float,
                    width: 4,
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
                inner: TypeInner::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                },
            },
            Default::default(),
        );

        let vec2_i32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Bi,
                    kind: ScalarKind::Sint,
                    width: 4,
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
                inner: TypeInner::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                },
            },
            Default::default(),
        );

        let vec2_i32_ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Bi,
                    kind: ScalarKind::Sint,
                    width: 4,
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
