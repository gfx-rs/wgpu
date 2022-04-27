use crate::{
    arena::{Arena, Handle, UniqueArena},
    BinaryOperator, Constant, ConstantInner, Expression, ScalarKind, ScalarValue, Type, TypeInner,
    UnaryOperator,
};

#[derive(Debug)]
pub struct ConstantSolver<'a> {
    pub types: &'a mut UniqueArena<Type>,
    pub expressions: &'a Arena<Expression>,
    pub constants: &'a mut Arena<Constant>,
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
    #[error("Not implemented as constant expression: {0}")]
    NotImplemented(String),
}

impl<'a> ConstantSolver<'a> {
    pub fn solve(
        &mut self,
        expr: Handle<Expression>,
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
        let span = self.expressions.get_span(expr);
        match self.expressions[expr] {
            Expression::Constant(constant) => Ok(constant),
            Expression::AccessIndex { base, index } => self.access(base, index as usize),
            Expression::Access { base, index } => {
                let index = self.solve(index)?;

                self.access(base, self.constant_index(index)?)
            }
            Expression::Splat {
                size,
                value: splat_value,
            } => {
                let value_constant = self.solve(splat_value)?;
                let ty = match self.constants[value_constant].inner {
                    ConstantInner::Scalar { ref value, width } => {
                        let kind = value.scalar_kind();
                        self.types.insert(
                            Type {
                                name: None,
                                inner: TypeInner::Vector { size, kind, width },
                            },
                            span,
                        )
                    }
                    ConstantInner::Composite { .. } => {
                        return Err(ConstantSolvingError::SplatScalarOnly);
                    }
                };

                let inner = ConstantInner::Composite {
                    ty,
                    components: vec![value_constant; size as usize],
                };
                Ok(self.register_constant(inner, span))
            }
            Expression::Swizzle {
                size,
                vector: src_vector,
                pattern,
            } => {
                let src_constant = self.solve(src_vector)?;
                let (ty, src_components) = match self.constants[src_constant].inner {
                    ConstantInner::Scalar { .. } => {
                        return Err(ConstantSolvingError::SwizzleVectorOnly);
                    }
                    ConstantInner::Composite {
                        ty,
                        components: ref src_components,
                    } => match self.types[ty].inner {
                        crate::TypeInner::Vector {
                            size: _,
                            kind,
                            width,
                        } => {
                            let dst_ty = self.types.insert(
                                Type {
                                    name: None,
                                    inner: crate::TypeInner::Vector { size, kind, width },
                                },
                                span,
                            );
                            (dst_ty, &src_components[..])
                        }
                        _ => {
                            return Err(ConstantSolvingError::SwizzleVectorOnly);
                        }
                    },
                };

                let components = pattern
                    .iter()
                    .map(|&sc| src_components[sc as usize])
                    .collect();
                let inner = ConstantInner::Composite { ty, components };

                Ok(self.register_constant(inner, span))
            }
            Expression::Compose { ty, ref components } => {
                let components = components
                    .iter()
                    .map(|c| self.solve(*c))
                    .collect::<Result<_, _>>()?;
                let inner = ConstantInner::Composite { ty, components };

                Ok(self.register_constant(inner, span))
            }
            Expression::Unary { expr, op } => {
                let expr_constant = self.solve(expr)?;

                self.unary_op(op, expr_constant, span)
            }
            Expression::Binary { left, right, op } => {
                let left_constant = self.solve(left)?;
                let right_constant = self.solve(right)?;

                self.binary_op(op, left_constant, right_constant, span)
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                ..
            } => {
                let arg = self.solve(arg)?;
                let arg1 = arg1.map(|arg| self.solve(arg)).transpose()?;
                let arg2 = arg2.map(|arg| self.solve(arg)).transpose()?;

                let const0 = &self.constants[arg].inner;
                let const1 = arg1.map(|arg| &self.constants[arg].inner);
                let const2 = arg2.map(|arg| &self.constants[arg].inner);

                match fun {
                    crate::MathFunction::Pow => {
                        let (value, width) = match (const0, const1.unwrap()) {
                            (
                                &ConstantInner::Scalar {
                                    width,
                                    value: value0,
                                },
                                &ConstantInner::Scalar { value: value1, .. },
                            ) => (
                                match (value0, value1) {
                                    (ScalarValue::Sint(a), ScalarValue::Sint(b)) => {
                                        ScalarValue::Sint(a.pow(b as u32))
                                    }
                                    (ScalarValue::Uint(a), ScalarValue::Uint(b)) => {
                                        ScalarValue::Uint(a.pow(b as u32))
                                    }
                                    (ScalarValue::Float(a), ScalarValue::Float(b)) => {
                                        ScalarValue::Float(a.powf(b))
                                    }
                                    _ => return Err(ConstantSolvingError::InvalidMathArg),
                                },
                                width,
                            ),
                            _ => return Err(ConstantSolvingError::InvalidMathArg),
                        };

                        let inner = ConstantInner::Scalar { width, value };
                        Ok(self.register_constant(inner, span))
                    }
                    crate::MathFunction::Clamp => {
                        let (value, width) = match (const0, const1.unwrap(), const2.unwrap()) {
                            (
                                &ConstantInner::Scalar {
                                    width,
                                    value: value0,
                                },
                                &ConstantInner::Scalar { value: value1, .. },
                                &ConstantInner::Scalar { value: value2, .. },
                            ) => (
                                match (value0, value1, value2) {
                                    (
                                        ScalarValue::Sint(a),
                                        ScalarValue::Sint(b),
                                        ScalarValue::Sint(c),
                                    ) => ScalarValue::Sint(a.max(b).min(c)),
                                    (
                                        ScalarValue::Uint(a),
                                        ScalarValue::Uint(b),
                                        ScalarValue::Uint(c),
                                    ) => ScalarValue::Uint(a.max(b).min(c)),
                                    (
                                        ScalarValue::Float(a),
                                        ScalarValue::Float(b),
                                        ScalarValue::Float(c),
                                    ) => ScalarValue::Float(glsl_float_clamp(a, b, c)),
                                    _ => return Err(ConstantSolvingError::InvalidMathArg),
                                },
                                width,
                            ),
                            _ => {
                                return Err(ConstantSolvingError::NotImplemented(format!(
                                    "{:?} applied to vector values",
                                    fun
                                )))
                            }
                        };

                        let inner = ConstantInner::Scalar { width, value };
                        Ok(self.register_constant(inner, span))
                    }
                    _ => Err(ConstantSolvingError::NotImplemented(format!("{:?}", fun))),
                }
            }
            Expression::As {
                convert,
                expr,
                kind,
            } => {
                let expr_constant = self.solve(expr)?;

                match convert {
                    Some(width) => self.cast(expr_constant, kind, width, span),
                    None => Err(ConstantSolvingError::Bitcast),
                }
            }
            Expression::ArrayLength(expr) => {
                let array = self.solve(expr)?;

                match self.constants[array].inner {
                    ConstantInner::Scalar { .. } => {
                        Err(ConstantSolvingError::InvalidArrayLengthArg)
                    }
                    ConstantInner::Composite { ty, .. } => match self.types[ty].inner {
                        TypeInner::Array { size, .. } => match size {
                            crate::ArraySize::Constant(constant) => Ok(constant),
                            crate::ArraySize::Dynamic => {
                                Err(ConstantSolvingError::ArrayLengthDynamic)
                            }
                        },
                        _ => Err(ConstantSolvingError::InvalidArrayLengthArg),
                    },
                }
            }

            Expression::Load { .. } => Err(ConstantSolvingError::Load),
            Expression::Select { .. } => Err(ConstantSolvingError::Select),
            Expression::LocalVariable(_) => Err(ConstantSolvingError::LocalVariable),
            Expression::Derivative { .. } => Err(ConstantSolvingError::Derivative),
            Expression::Relational { .. } => Err(ConstantSolvingError::Relational),
            Expression::CallResult { .. } => Err(ConstantSolvingError::Call),
            Expression::AtomicResult { .. } => Err(ConstantSolvingError::Atomic),
            Expression::FunctionArgument(_) => Err(ConstantSolvingError::FunctionArg),
            Expression::GlobalVariable(_) => Err(ConstantSolvingError::GlobalVariable),
            Expression::ImageSample { .. }
            | Expression::ImageLoad { .. }
            | Expression::ImageQuery { .. } => Err(ConstantSolvingError::ImageExpression),
        }
    }

    fn access(
        &mut self,
        base: Handle<Expression>,
        index: usize,
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
        let base = self.solve(base)?;

        match self.constants[base].inner {
            ConstantInner::Scalar { .. } => Err(ConstantSolvingError::InvalidAccessBase),
            ConstantInner::Composite { ty, ref components } => {
                match self.types[ty].inner {
                    TypeInner::Vector { .. }
                    | TypeInner::Matrix { .. }
                    | TypeInner::Array { .. }
                    | TypeInner::Struct { .. } => (),
                    _ => return Err(ConstantSolvingError::InvalidAccessBase),
                }

                components
                    .get(index)
                    .copied()
                    .ok_or(ConstantSolvingError::InvalidAccessIndex)
            }
        }
    }

    fn constant_index(&self, constant: Handle<Constant>) -> Result<usize, ConstantSolvingError> {
        match self.constants[constant].inner {
            ConstantInner::Scalar {
                value: ScalarValue::Uint(index),
                ..
            } => Ok(index as usize),
            _ => Err(ConstantSolvingError::InvalidAccessIndexTy),
        }
    }

    fn cast(
        &mut self,
        constant: Handle<Constant>,
        kind: ScalarKind,
        target_width: crate::Bytes,
        span: crate::Span,
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
        let mut inner = self.constants[constant].inner.clone();

        match inner {
            ConstantInner::Scalar {
                ref mut value,
                ref mut width,
            } => {
                *width = target_width;
                *value = match kind {
                    ScalarKind::Sint => ScalarValue::Sint(match *value {
                        ScalarValue::Sint(v) => v,
                        ScalarValue::Uint(v) => v as i64,
                        ScalarValue::Float(v) => v as i64,
                        ScalarValue::Bool(v) => v as i64,
                    }),
                    ScalarKind::Uint => ScalarValue::Uint(match *value {
                        ScalarValue::Sint(v) => v as u64,
                        ScalarValue::Uint(v) => v,
                        ScalarValue::Float(v) => v as u64,
                        ScalarValue::Bool(v) => v as u64,
                    }),
                    ScalarKind::Float => ScalarValue::Float(match *value {
                        ScalarValue::Sint(v) => v as f64,
                        ScalarValue::Uint(v) => v as f64,
                        ScalarValue::Float(v) => v,
                        ScalarValue::Bool(v) => v as u64 as f64,
                    }),
                    ScalarKind::Bool => ScalarValue::Bool(match *value {
                        ScalarValue::Sint(v) => v != 0,
                        ScalarValue::Uint(v) => v != 0,
                        ScalarValue::Float(v) => v != 0.0,
                        ScalarValue::Bool(v) => v,
                    }),
                }
            }
            ConstantInner::Composite {
                ty,
                ref mut components,
            } => {
                match self.types[ty].inner {
                    TypeInner::Vector { .. } | TypeInner::Matrix { .. } => (),
                    _ => return Err(ConstantSolvingError::InvalidCastArg),
                }

                for component in components {
                    *component = self.cast(*component, kind, target_width, span)?;
                }
            }
        }

        Ok(self.register_constant(inner, span))
    }

    fn unary_op(
        &mut self,
        op: UnaryOperator,
        constant: Handle<Constant>,
        span: crate::Span,
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
        let mut inner = self.constants[constant].inner.clone();

        match inner {
            ConstantInner::Scalar { ref mut value, .. } => match op {
                UnaryOperator::Negate => match *value {
                    ScalarValue::Sint(ref mut v) => *v = -*v,
                    ScalarValue::Float(ref mut v) => *v = -*v,
                    _ => return Err(ConstantSolvingError::InvalidUnaryOpArg),
                },
                UnaryOperator::Not => match *value {
                    ScalarValue::Sint(ref mut v) => *v = !*v,
                    ScalarValue::Uint(ref mut v) => *v = !*v,
                    ScalarValue::Bool(ref mut v) => *v = !*v,
                    _ => return Err(ConstantSolvingError::InvalidUnaryOpArg),
                },
            },
            ConstantInner::Composite {
                ty,
                ref mut components,
            } => {
                match self.types[ty].inner {
                    TypeInner::Vector { .. } | TypeInner::Matrix { .. } => (),
                    _ => return Err(ConstantSolvingError::InvalidCastArg),
                }

                for component in components {
                    *component = self.unary_op(op, *component, span)?
                }
            }
        }

        Ok(self.register_constant(inner, span))
    }

    fn binary_op(
        &mut self,
        op: BinaryOperator,
        left: Handle<Constant>,
        right: Handle<Constant>,
        span: crate::Span,
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
        let left_inner = &self.constants[left].inner;
        let right_inner = &self.constants[right].inner;

        let inner = match (left_inner, right_inner) {
            (
                &ConstantInner::Scalar {
                    value: left_value,
                    width,
                },
                &ConstantInner::Scalar {
                    value: right_value,
                    width: _,
                },
            ) => {
                let value = match op {
                    BinaryOperator::Equal => ScalarValue::Bool(left_value == right_value),
                    BinaryOperator::NotEqual => ScalarValue::Bool(left_value != right_value),
                    BinaryOperator::Less => ScalarValue::Bool(left_value < right_value),
                    BinaryOperator::LessEqual => ScalarValue::Bool(left_value <= right_value),
                    BinaryOperator::Greater => ScalarValue::Bool(left_value > right_value),
                    BinaryOperator::GreaterEqual => ScalarValue::Bool(left_value >= right_value),

                    _ => match (left_value, right_value) {
                        (ScalarValue::Sint(a), ScalarValue::Sint(b)) => {
                            ScalarValue::Sint(match op {
                                BinaryOperator::Add => a.wrapping_add(b),
                                BinaryOperator::Subtract => a.wrapping_sub(b),
                                BinaryOperator::Multiply => a.wrapping_mul(b),
                                BinaryOperator::Divide => a.checked_div(b).unwrap_or(0),
                                BinaryOperator::Modulo => a.checked_rem(b).unwrap_or(0),
                                BinaryOperator::And => a & b,
                                BinaryOperator::ExclusiveOr => a ^ b,
                                BinaryOperator::InclusiveOr => a | b,
                                _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                            })
                        }
                        (ScalarValue::Sint(a), ScalarValue::Uint(b)) => {
                            ScalarValue::Sint(match op {
                                BinaryOperator::ShiftLeft => a.wrapping_shl(b as u32),
                                BinaryOperator::ShiftRight => a.wrapping_shr(b as u32),
                                _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                            })
                        }
                        (ScalarValue::Uint(a), ScalarValue::Uint(b)) => {
                            ScalarValue::Uint(match op {
                                BinaryOperator::Add => a.wrapping_add(b),
                                BinaryOperator::Subtract => a.wrapping_sub(b),
                                BinaryOperator::Multiply => a.wrapping_mul(b),
                                BinaryOperator::Divide => a.checked_div(b).unwrap_or(0),
                                BinaryOperator::Modulo => a.checked_rem(b).unwrap_or(0),
                                BinaryOperator::And => a & b,
                                BinaryOperator::ExclusiveOr => a ^ b,
                                BinaryOperator::InclusiveOr => a | b,
                                BinaryOperator::ShiftLeft => a.wrapping_shl(b as u32),
                                BinaryOperator::ShiftRight => a.wrapping_shr(b as u32),
                                _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                            })
                        }
                        (ScalarValue::Float(a), ScalarValue::Float(b)) => {
                            ScalarValue::Float(match op {
                                BinaryOperator::Add => a + b,
                                BinaryOperator::Subtract => a - b,
                                BinaryOperator::Multiply => a * b,
                                BinaryOperator::Divide => a / b,
                                BinaryOperator::Modulo => a - b * (a / b).floor(),
                                _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                            })
                        }
                        (ScalarValue::Bool(a), ScalarValue::Bool(b)) => {
                            ScalarValue::Bool(match op {
                                BinaryOperator::LogicalAnd => a && b,
                                BinaryOperator::LogicalOr => a || b,
                                _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                            })
                        }
                        _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                    },
                };

                ConstantInner::Scalar { value, width }
            }
            (&ConstantInner::Composite { ref components, ty }, &ConstantInner::Scalar { .. }) => {
                let mut components = components.clone();
                for comp in components.iter_mut() {
                    *comp = self.binary_op(op, *comp, right, span)?;
                }
                ConstantInner::Composite { ty, components }
            }
            (&ConstantInner::Scalar { .. }, &ConstantInner::Composite { ref components, ty }) => {
                let mut components = components.clone();
                for comp in components.iter_mut() {
                    *comp = self.binary_op(op, left, *comp, span)?;
                }
                ConstantInner::Composite { ty, components }
            }
            _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
        };

        Ok(self.register_constant(inner, span))
    }

    fn register_constant(&mut self, inner: ConstantInner, span: crate::Span) -> Handle<Constant> {
        self.constants.fetch_or_append(
            Constant {
                name: None,
                specialization: None,
                inner,
            },
            span,
        )
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
fn glsl_float_max(x: f64, y: f64) -> f64 {
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
fn glsl_float_min(x: f64, y: f64) -> f64 {
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
fn glsl_float_clamp(value: f64, min: f64, max: f64) -> f64 {
    glsl_float_min(glsl_float_max(value, min), max)
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::{
        Arena, Constant, ConstantInner, Expression, ScalarKind, ScalarValue, Type, TypeInner,
        UnaryOperator, UniqueArena, VectorSize,
    };

    use super::ConstantSolver;

    #[test]
    fn nan_handling() {
        assert!(super::glsl_float_max(f64::NAN, 2.0).is_nan());
        assert!(!super::glsl_float_max(2.0, f64::NAN).is_nan());

        assert!(super::glsl_float_min(f64::NAN, 2.0).is_nan());
        assert!(!super::glsl_float_min(2.0, f64::NAN).is_nan());

        assert!(super::glsl_float_clamp(f64::NAN, 1.0, 2.0).is_nan());
        assert!(!super::glsl_float_clamp(1.0, f64::NAN, 2.0).is_nan());
        assert!(!super::glsl_float_clamp(1.0, 2.0, f64::NAN).is_nan());
    }

    #[test]
    fn unary_op() {
        let mut types = UniqueArena::new();
        let mut expressions = Arena::new();
        let mut constants = Arena::new();

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
                specialization: None,
                inner: ConstantInner::Scalar {
                    width: 4,
                    value: ScalarValue::Sint(4),
                },
            },
            Default::default(),
        );

        let h1 = constants.append(
            Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Scalar {
                    width: 4,
                    value: ScalarValue::Sint(8),
                },
            },
            Default::default(),
        );

        let vec_h = constants.append(
            Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Composite {
                    ty: vec_ty,
                    components: vec![h, h1],
                },
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
        };

        let res1 = solver.solve(root1).unwrap();
        let res2 = solver.solve(root2).unwrap();
        let res3 = solver.solve(root3).unwrap();

        assert_eq!(
            constants[res1].inner,
            ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Sint(-4),
            },
        );

        assert_eq!(
            constants[res2].inner,
            ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Sint(!4),
            },
        );

        let res3_inner = &constants[res3].inner;

        match res3_inner {
            ConstantInner::Composite { ty, components } => {
                assert_eq!(*ty, vec_ty);
                let mut components_iter = components.iter().copied();
                assert_eq!(
                    constants[components_iter.next().unwrap()].inner,
                    ConstantInner::Scalar {
                        width: 4,
                        value: ScalarValue::Sint(!4),
                    },
                );
                assert_eq!(
                    constants[components_iter.next().unwrap()].inner,
                    ConstantInner::Scalar {
                        width: 4,
                        value: ScalarValue::Sint(!8),
                    },
                );
                assert!(components_iter.next().is_none());
            }
            _ => panic!("Expected vector"),
        }
    }

    #[test]
    fn cast() {
        let mut expressions = Arena::new();
        let mut constants = Arena::new();

        let h = constants.append(
            Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Scalar {
                    width: 4,
                    value: ScalarValue::Sint(4),
                },
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
            types: &mut UniqueArena::new(),
            expressions: &expressions,
            constants: &mut constants,
        };

        let res = solver.solve(root).unwrap();

        assert_eq!(
            constants[res].inner,
            ConstantInner::Scalar {
                width: crate::BOOL_WIDTH,
                value: ScalarValue::Bool(true),
            },
        );
    }

    #[test]
    fn access() {
        let mut types = UniqueArena::new();
        let mut expressions = Arena::new();
        let mut constants = Arena::new();

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
            let h = constants.append(
                Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Scalar {
                        width: 4,
                        value: ScalarValue::Float(i as f64),
                    },
                },
                Default::default(),
            );

            vec1_components.push(h)
        }

        for i in 3..6 {
            let h = constants.append(
                Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Scalar {
                        width: 4,
                        value: ScalarValue::Float(i as f64),
                    },
                },
                Default::default(),
            );

            vec2_components.push(h)
        }

        let vec1 = constants.append(
            Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Composite {
                    ty: vec_ty,
                    components: vec1_components,
                },
            },
            Default::default(),
        );

        let vec2 = constants.append(
            Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Composite {
                    ty: vec_ty,
                    components: vec2_components,
                },
            },
            Default::default(),
        );

        let h = constants.append(
            Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Composite {
                    ty: matrix_ty,
                    components: vec![vec1, vec2],
                },
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
        };

        let res1 = solver.solve(root1).unwrap();
        let res2 = solver.solve(root2).unwrap();

        let res1_inner = &constants[res1].inner;

        match res1_inner {
            ConstantInner::Composite { ty, components } => {
                assert_eq!(*ty, vec_ty);
                let mut components_iter = components.iter().copied();
                assert_eq!(
                    constants[components_iter.next().unwrap()].inner,
                    ConstantInner::Scalar {
                        width: 4,
                        value: ScalarValue::Float(3.),
                    },
                );
                assert_eq!(
                    constants[components_iter.next().unwrap()].inner,
                    ConstantInner::Scalar {
                        width: 4,
                        value: ScalarValue::Float(4.),
                    },
                );
                assert_eq!(
                    constants[components_iter.next().unwrap()].inner,
                    ConstantInner::Scalar {
                        width: 4,
                        value: ScalarValue::Float(5.),
                    },
                );
                assert!(components_iter.next().is_none());
            }
            _ => panic!("Expected vector"),
        }

        assert_eq!(
            constants[res2].inner,
            ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Float(5.),
            },
        );
    }
}
