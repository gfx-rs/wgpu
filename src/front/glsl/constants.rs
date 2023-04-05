use crate::{
    arena::{Arena, Handle, UniqueArena},
    ArraySize, BinaryOperator, Constant, Expression, Literal, ScalarKind, Type, TypeInner,
    UnaryOperator,
};

#[derive(Debug)]
pub struct ConstantSolver<'a> {
    pub types: &'a mut UniqueArena<Type>,
    pub expressions: &'a Arena<Expression>,
    pub constants: &'a mut Arena<Constant>,
    pub const_expressions: &'a mut Arena<Expression>,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ConstantSolvingError {
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
    #[error("Type is not constructible")]
    TypeNotConstructible,
    #[error("Not implemented as constant expression: {0}")]
    NotImplemented(String),
}

#[derive(Clone, Copy)]
pub enum ExprType {
    Regular,
    Constant,
}

impl<'a> ConstantSolver<'a> {
    pub fn solve(
        &mut self,
        expr: Handle<Expression>,
    ) -> Result<Handle<Expression>, ConstantSolvingError> {
        self.solve_impl(expr, ExprType::Regular, true)
    }

    pub fn solve_impl(
        &mut self,
        expr: Handle<Expression>,
        expr_type: ExprType,
        top_level: bool,
    ) -> Result<Handle<Expression>, ConstantSolvingError> {
        let expressions = match expr_type {
            ExprType::Regular => self.expressions,
            ExprType::Constant => self.const_expressions,
        };
        let span = expressions.get_span(expr);
        match expressions[expr] {
            ref expression @ (Expression::Literal(_) | Expression::ZeroValue(_)) => match expr_type
            {
                ExprType::Regular => Ok(self.register_constant(expression.clone(), span)),
                ExprType::Constant => Ok(expr),
            },
            Expression::Compose { ty, ref components } => match expr_type {
                ExprType::Regular => {
                    let mut components = components.clone();
                    for component in &mut components {
                        *component = self.solve_impl(*component, expr_type, false)?;
                    }
                    Ok(self.register_constant(Expression::Compose { ty, components }, span))
                }
                ExprType::Constant => Ok(expr),
            },
            Expression::Constant(constant) => {
                if top_level {
                    match expr_type {
                        ExprType::Regular => {
                            Ok(self.register_constant(Expression::Constant(constant), span))
                        }
                        ExprType::Constant => Ok(expr),
                    }
                } else {
                    self.solve_impl(self.constants[constant].init, ExprType::Constant, false)
                }
            }
            Expression::AccessIndex { base, index } => {
                let base = self.solve_impl(base, expr_type, false)?;
                self.access(base, index as usize, span)
            }
            Expression::Access { base, index } => {
                let base = self.solve_impl(base, expr_type, false)?;
                let index = self.solve_impl(index, expr_type, false)?;

                self.access(base, self.constant_index(index)?, span)
            }
            Expression::Splat {
                size,
                value: splat_value,
            } => {
                let value_constant = self.solve_impl(splat_value, expr_type, false)?;
                let ty = match self.const_expressions[value_constant] {
                    Expression::Literal(literal) => {
                        let kind = literal.scalar_kind();
                        let width = literal.width();
                        self.types.insert(
                            Type {
                                name: None,
                                inner: TypeInner::Vector { size, kind, width },
                            },
                            span,
                        )
                    }
                    Expression::ZeroValue(ty) => {
                        let inner = match self.types[ty].inner {
                            TypeInner::Scalar { kind, width } => {
                                TypeInner::Vector { size, kind, width }
                            }
                            _ => return Err(ConstantSolvingError::SplatScalarOnly),
                        };
                        let res_ty = self.types.insert(Type { name: None, inner }, span);
                        let expr = Expression::ZeroValue(res_ty);
                        return Ok(self.register_constant(expr, span));
                    }
                    _ => {
                        return Err(ConstantSolvingError::SplatScalarOnly);
                    }
                };

                let expr = Expression::Compose {
                    ty,
                    components: vec![value_constant; size as usize],
                };
                Ok(self.register_constant(expr, span))
            }
            Expression::Swizzle {
                size,
                vector: src_vector,
                pattern,
            } => {
                let src_constant = self.solve_impl(src_vector, expr_type, false)?;

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
                    _ => Err(ConstantSolvingError::SwizzleVectorOnly),
                };

                match self.const_expressions[src_constant] {
                    Expression::ZeroValue(ty) => {
                        let dst_ty = get_dst_ty(ty)?;
                        let expr = Expression::ZeroValue(dst_ty);
                        Ok(self.register_constant(expr, span))
                    }
                    Expression::Compose {
                        ty,
                        components: ref src_components,
                    } => {
                        let dst_ty = get_dst_ty(ty)?;

                        let components = pattern
                            .iter()
                            .map(|&sc| src_components[sc as usize])
                            .collect();
                        let expr = Expression::Compose {
                            ty: dst_ty,
                            components,
                        };
                        Ok(self.register_constant(expr, span))
                    }
                    _ => Err(ConstantSolvingError::SwizzleVectorOnly),
                }
            }
            Expression::Unary { expr, op } => {
                let expr_constant = self.solve_impl(expr, expr_type, false)?;

                self.unary_op(op, expr_constant, span)
            }
            Expression::Binary { left, right, op } => {
                let left_constant = self.solve_impl(left, expr_type, false)?;
                let right_constant = self.solve_impl(right, expr_type, false)?;

                self.binary_op(op, left_constant, right_constant, span)
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                let arg = self.solve_impl(arg, expr_type, false)?;
                let arg1 = arg1
                    .map(|arg| self.solve_impl(arg, expr_type, false))
                    .transpose()?;
                let arg2 = arg2
                    .map(|arg| self.solve_impl(arg, expr_type, false))
                    .transpose()?;
                let arg3 = arg3
                    .map(|arg| self.solve_impl(arg, expr_type, false))
                    .transpose()?;

                let const0 = &self.const_expressions[arg];
                let const1 = arg1.map(|arg| &self.const_expressions[arg]);
                let const2 = arg2.map(|arg| &self.const_expressions[arg]);
                let _const3 = arg3.map(|arg| &self.const_expressions[arg]);

                match fun {
                    crate::MathFunction::Pow => {
                        let literal = match (const0, const1.unwrap()) {
                            (&Expression::Literal(value0), &Expression::Literal(value1)) => {
                                match (value0, value1) {
                                    (Literal::I32(a), Literal::I32(b)) => {
                                        Literal::I32(a.pow(b as u32))
                                    }
                                    (Literal::U32(a), Literal::U32(b)) => Literal::U32(a.pow(b)),
                                    (Literal::F32(a), Literal::F32(b)) => Literal::F32(a.powf(b)),
                                    _ => return Err(ConstantSolvingError::InvalidMathArg),
                                }
                            }
                            _ => return Err(ConstantSolvingError::InvalidMathArg),
                        };

                        let expr = Expression::Literal(literal);
                        Ok(self.register_constant(expr, span))
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
                                _ => return Err(ConstantSolvingError::InvalidMathArg),
                            },
                            _ => {
                                return Err(ConstantSolvingError::NotImplemented(format!(
                                    "{fun:?} applied to vector values"
                                )))
                            }
                        };

                        let expr = Expression::Literal(literal);
                        Ok(self.register_constant(expr, span))
                    }
                    _ => Err(ConstantSolvingError::NotImplemented(format!("{fun:?}"))),
                }
            }
            Expression::As {
                convert,
                expr,
                kind,
            } => {
                let expr_constant = self.solve_impl(expr, expr_type, false)?;

                match convert {
                    Some(width) => self.cast(expr_constant, kind, width, span),
                    None => Err(ConstantSolvingError::Bitcast),
                }
            }
            Expression::ArrayLength(expr) => {
                let array = self.solve_impl(expr, expr_type, false)?;

                match self.const_expressions[array] {
                    Expression::ZeroValue(ty) | Expression::Compose { ty, .. } => {
                        match self.types[ty].inner {
                            TypeInner::Array { size, .. } => match size {
                                crate::ArraySize::Constant(len) => {
                                    let expr = Expression::Literal(Literal::U32(len.get()));
                                    Ok(self.register_constant(expr, span))
                                }
                                crate::ArraySize::Dynamic => {
                                    Err(ConstantSolvingError::ArrayLengthDynamic)
                                }
                            },
                            _ => Err(ConstantSolvingError::InvalidArrayLengthArg),
                        }
                    }
                    _ => Err(ConstantSolvingError::InvalidArrayLengthArg),
                }
            }

            Expression::Load { .. } => Err(ConstantSolvingError::Load),
            Expression::Select { .. } => Err(ConstantSolvingError::Select),
            Expression::LocalVariable(_) => Err(ConstantSolvingError::LocalVariable),
            Expression::Derivative { .. } => Err(ConstantSolvingError::Derivative),
            Expression::Relational { .. } => Err(ConstantSolvingError::Relational),
            Expression::CallResult { .. } => Err(ConstantSolvingError::Call),
            Expression::WorkGroupUniformLoadResult { .. } => unreachable!(),
            Expression::AtomicResult { .. } => Err(ConstantSolvingError::Atomic),
            Expression::FunctionArgument(_) => Err(ConstantSolvingError::FunctionArg),
            Expression::GlobalVariable(_) => Err(ConstantSolvingError::GlobalVariable),
            Expression::ImageSample { .. }
            | Expression::ImageLoad { .. }
            | Expression::ImageQuery { .. } => Err(ConstantSolvingError::ImageExpression),
            Expression::RayQueryProceedResult | Expression::RayQueryGetIntersection { .. } => {
                Err(ConstantSolvingError::RayQueryExpression)
            }
        }
    }

    fn access(
        &mut self,
        base: Handle<Expression>,
        index: usize,
        span: crate::Span,
    ) -> Result<Handle<Expression>, ConstantSolvingError> {
        match self.const_expressions[base] {
            Expression::ZeroValue(ty) => {
                let ty_inner = &self.types[ty].inner;
                let components = ty_inner
                    .components()
                    .ok_or(ConstantSolvingError::InvalidAccessBase)?;

                if index >= components as usize {
                    Err(ConstantSolvingError::InvalidAccessBase)
                } else {
                    let ty_res = ty_inner
                        .component_type(index)
                        .ok_or(ConstantSolvingError::InvalidAccessIndex)?;
                    let ty = match ty_res {
                        crate::proc::TypeResolution::Handle(ty) => ty,
                        crate::proc::TypeResolution::Value(inner) => {
                            self.types.insert(Type { name: None, inner }, span)
                        }
                    };
                    Ok(self.register_constant(Expression::ZeroValue(ty), span))
                }
            }
            Expression::Compose { ty, ref components } => {
                let _ = self.types[ty]
                    .inner
                    .components()
                    .ok_or(ConstantSolvingError::InvalidAccessBase)?;

                components
                    .get(index)
                    .copied()
                    .ok_or(ConstantSolvingError::InvalidAccessIndex)
            }
            _ => Err(ConstantSolvingError::InvalidAccessBase),
        }
    }

    fn constant_index(&self, expr: Handle<Expression>) -> Result<usize, ConstantSolvingError> {
        match self.const_expressions[expr] {
            Expression::Literal(Literal::U32(index)) => Ok(index as usize),
            _ => Err(ConstantSolvingError::InvalidAccessIndexTy),
        }
    }

    /// Transforms a `Expression::ZeroValue` into either `Expression::Literal` or `Expression::Compose`
    fn eval_zero_value(
        &mut self,
        expr: Handle<Expression>,
        span: crate::Span,
    ) -> Result<Handle<Expression>, ConstantSolvingError> {
        match self.const_expressions[expr] {
            Expression::ZeroValue(ty) => self.eval_zero_value_impl(ty, span),
            _ => Ok(expr),
        }
    }

    fn eval_zero_value_impl(
        &mut self,
        ty: Handle<Type>,
        span: crate::Span,
    ) -> Result<Handle<Expression>, ConstantSolvingError> {
        match self.types[ty].inner {
            TypeInner::Scalar { kind, width } => {
                let expr = Expression::Literal(
                    Literal::zero(kind, width).ok_or(ConstantSolvingError::TypeNotConstructible)?,
                );
                Ok(self.register_constant(expr, span))
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
                Ok(self.register_constant(expr, span))
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
                Ok(self.register_constant(expr, span))
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
                Ok(self.register_constant(expr, span))
            }
            TypeInner::Struct { ref members, .. } => {
                let types: Vec<_> = members.iter().map(|m| m.ty).collect();
                let mut components = Vec::with_capacity(members.len());
                for ty in types {
                    components.push(self.eval_zero_value_impl(ty, span)?);
                }
                let expr = Expression::Compose { ty, components };
                Ok(self.register_constant(expr, span))
            }
            _ => Err(ConstantSolvingError::TypeNotConstructible),
        }
    }

    fn cast(
        &mut self,
        expr: Handle<Expression>,
        kind: ScalarKind,
        target_width: crate::Bytes,
        span: crate::Span,
    ) -> Result<Handle<Expression>, ConstantSolvingError> {
        let expr = self.eval_zero_value(expr, span)?;

        let expr = match self.const_expressions[expr] {
            Expression::Literal(literal) => {
                let literal = match (kind, target_width) {
                    (ScalarKind::Sint, 4) => Literal::I32(match literal {
                        Literal::I32(v) => v,
                        Literal::U32(v) => v as i32,
                        Literal::F32(v) => v as i32,
                        Literal::Bool(v) => v as i32,
                        Literal::F64(_) => return Err(ConstantSolvingError::InvalidCastArg),
                    }),
                    (ScalarKind::Uint, 4) => Literal::U32(match literal {
                        Literal::I32(v) => v as u32,
                        Literal::U32(v) => v,
                        Literal::F32(v) => v as u32,
                        Literal::Bool(v) => v as u32,
                        Literal::F64(_) => return Err(ConstantSolvingError::InvalidCastArg),
                    }),
                    (ScalarKind::Float, 4) => Literal::F32(match literal {
                        Literal::I32(v) => v as f32,
                        Literal::U32(v) => v as f32,
                        Literal::F32(v) => v,
                        Literal::Bool(v) => v as u32 as f32,
                        Literal::F64(_) => return Err(ConstantSolvingError::InvalidCastArg),
                    }),
                    (ScalarKind::Bool, crate::BOOL_WIDTH) => Literal::Bool(match literal {
                        Literal::I32(v) => v != 0,
                        Literal::U32(v) => v != 0,
                        Literal::F32(v) => v != 0.0,
                        Literal::Bool(v) => v,
                        Literal::F64(_) => return Err(ConstantSolvingError::InvalidCastArg),
                    }),
                    _ => return Err(ConstantSolvingError::InvalidCastArg),
                };
                Expression::Literal(literal)
            }
            Expression::Compose {
                ty,
                components: ref src_components,
            } => {
                match self.types[ty].inner {
                    TypeInner::Vector { .. } | TypeInner::Matrix { .. } => (),
                    _ => return Err(ConstantSolvingError::InvalidCastArg),
                }

                let mut components = src_components.clone();
                for component in &mut components {
                    *component = self.cast(*component, kind, target_width, span)?;
                }

                Expression::Compose { ty, components }
            }
            _ => return Err(ConstantSolvingError::InvalidCastArg),
        };

        Ok(self.register_constant(expr, span))
    }

    fn unary_op(
        &mut self,
        op: UnaryOperator,
        expr: Handle<Expression>,
        span: crate::Span,
    ) -> Result<Handle<Expression>, ConstantSolvingError> {
        let expr = self.eval_zero_value(expr, span)?;

        let expr = match self.const_expressions[expr] {
            Expression::Literal(value) => Expression::Literal(match op {
                UnaryOperator::Negate => match value {
                    Literal::I32(v) => Literal::I32(-v),
                    Literal::F32(v) => Literal::F32(-v),
                    _ => return Err(ConstantSolvingError::InvalidUnaryOpArg),
                },
                UnaryOperator::Not => match value {
                    Literal::I32(v) => Literal::I32(!v),
                    Literal::U32(v) => Literal::U32(!v),
                    Literal::Bool(v) => Literal::Bool(!v),
                    _ => return Err(ConstantSolvingError::InvalidUnaryOpArg),
                },
            }),
            Expression::Compose {
                ty,
                components: ref src_components,
            } => {
                match self.types[ty].inner {
                    TypeInner::Vector { .. } | TypeInner::Matrix { .. } => (),
                    _ => return Err(ConstantSolvingError::InvalidUnaryOpArg),
                }

                let mut components = src_components.clone();
                for component in &mut components {
                    *component = self.unary_op(op, *component, span)?;
                }

                Expression::Compose { ty, components }
            }
            _ => return Err(ConstantSolvingError::InvalidUnaryOpArg),
        };

        Ok(self.register_constant(expr, span))
    }

    fn binary_op(
        &mut self,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
        span: crate::Span,
    ) -> Result<Handle<Expression>, ConstantSolvingError> {
        let left = self.eval_zero_value(left, span)?;
        let right = self.eval_zero_value(right, span)?;

        let expr = match (
            &self.const_expressions[left],
            &self.const_expressions[right],
        ) {
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
                            _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                        }),
                        (Literal::I32(a), Literal::U32(b)) => Literal::I32(match op {
                            BinaryOperator::ShiftLeft => a.wrapping_shl(b),
                            BinaryOperator::ShiftRight => a.wrapping_shr(b),
                            _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
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
                            _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                        }),
                        (Literal::F32(a), Literal::F32(b)) => Literal::F32(match op {
                            BinaryOperator::Add => a + b,
                            BinaryOperator::Subtract => a - b,
                            BinaryOperator::Multiply => a * b,
                            BinaryOperator::Divide => a / b,
                            BinaryOperator::Modulo => a - b * (a / b).floor(),
                            _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                        }),
                        (Literal::Bool(a), Literal::Bool(b)) => Literal::Bool(match op {
                            BinaryOperator::LogicalAnd => a && b,
                            BinaryOperator::LogicalOr => a || b,
                            _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                        }),
                        _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
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
            _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
        };

        Ok(self.register_constant(expr, span))
    }

    fn register_constant(&mut self, expr: Expression, span: crate::Span) -> Handle<Expression> {
        self.const_expressions.append(expr, span)
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

    use super::ConstantSolver;

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
        let mut expressions = Arena::new();
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

        let expr = expressions.append(Expression::Constant(h), Default::default());
        let expr1 = expressions.append(Expression::Constant(vec_h), Default::default());

        let root1 = expressions.append(
            Expression::Unary {
                op: UnaryOperator::Negate,
                expr,
            },
            Default::default(),
        );

        let root2 = expressions.append(
            Expression::Unary {
                op: UnaryOperator::Not,
                expr,
            },
            Default::default(),
        );

        let root3 = expressions.append(
            Expression::Unary {
                op: UnaryOperator::Not,
                expr: expr1,
            },
            Default::default(),
        );

        let mut solver = ConstantSolver {
            types: &mut types,
            expressions: &expressions,
            constants: &mut constants,
            const_expressions: &mut const_expressions,
        };

        let res1 = solver.solve(root1).unwrap();
        let res2 = solver.solve(root2).unwrap();
        let res3 = solver.solve(root3).unwrap();

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
        let mut expressions = Arena::new();
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

        let expr = expressions.append(Expression::Constant(h), Default::default());

        let root = expressions.append(
            Expression::As {
                expr,
                kind: ScalarKind::Bool,
                convert: Some(crate::BOOL_WIDTH),
            },
            Default::default(),
        );

        let mut solver = ConstantSolver {
            types: &mut types,
            expressions: &expressions,
            constants: &mut constants,
            const_expressions: &mut const_expressions,
        };

        let res = solver.solve(root).unwrap();

        assert_eq!(
            const_expressions[res],
            Expression::Literal(Literal::Bool(true))
        );
    }

    #[test]
    fn access() {
        let mut types = UniqueArena::new();
        let mut expressions = Arena::new();
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

        let base = expressions.append(Expression::Constant(h), Default::default());
        let root1 = expressions.append(
            Expression::AccessIndex { base, index: 1 },
            Default::default(),
        );
        let root2 = expressions.append(
            Expression::AccessIndex {
                base: root1,
                index: 2,
            },
            Default::default(),
        );

        let mut solver = ConstantSolver {
            types: &mut types,
            expressions: &expressions,
            constants: &mut constants,
            const_expressions: &mut const_expressions,
        };

        let res1 = solver.solve(root1).unwrap();
        let res2 = solver.solve(root2).unwrap();

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
}
