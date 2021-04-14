use crate::{
    arena::{Arena, Handle},
    BinaryOperator, Constant, ConstantInner, Expression, ScalarKind, ScalarValue, Type, TypeInner,
    UnaryOperator,
};

#[derive(Debug)]
pub struct ConstantSolver<'a> {
    pub types: &'a Arena<Type>,
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
    #[error("Splat type is not registered")]
    SplatType,
}

impl<'a> ConstantSolver<'a> {
    pub fn solve(
        &mut self,
        expr: Handle<Expression>,
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
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
                let tgt = self.solve(splat_value)?;
                let ty = match self.constants[tgt].inner {
                    ConstantInner::Scalar { ref value, width } => {
                        let kind = value.scalar_kind();
                        self.types
                            .fetch_if(|t| t.inner == crate::TypeInner::Vector { size, kind, width })
                    }
                    ConstantInner::Composite { .. } => None,
                };
                Ok(self.constants.fetch_or_append(Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Composite {
                        ty: ty.ok_or(ConstantSolvingError::SplatType)?,
                        components: vec![tgt; size as usize],
                    },
                }))
            }
            Expression::Compose { ty, ref components } => {
                let components = components
                    .iter()
                    .map(|c| self.solve(*c))
                    .collect::<Result<_, _>>()?;

                Ok(self.constants.fetch_or_append(Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Composite { ty, components },
                }))
            }
            Expression::Unary { expr, op } => {
                let tgt = self.solve(expr)?;

                self.unary_op(op, tgt)
            }
            Expression::Binary { left, right, op } => {
                let left = self.solve(left)?;
                let right = self.solve(right)?;

                self.binary_op(op, left, right)
            }
            Expression::Math { .. } => todo!(),
            Expression::As {
                convert,
                expr,
                kind,
            } => {
                let tgt = self.solve(expr)?;

                if convert {
                    self.cast(tgt, kind)
                } else {
                    Err(ConstantSolvingError::Bitcast)
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
            Expression::Call { .. } => Err(ConstantSolvingError::Call),
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
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
        fn inner_cast<A: num_traits::FromPrimitive>(value: ScalarValue) -> A {
            match value {
                ScalarValue::Sint(v) => A::from_i64(v),
                ScalarValue::Uint(v) => A::from_u64(v),
                ScalarValue::Float(v) => A::from_f64(v),
                ScalarValue::Bool(v) => A::from_u64(v as u64),
            }
            .unwrap()
        }

        let mut inner = self.constants[constant].inner.clone();

        match inner {
            ConstantInner::Scalar { ref mut value, .. } => {
                *value = match kind {
                    ScalarKind::Sint => ScalarValue::Sint(inner_cast(*value)),
                    ScalarKind::Uint => ScalarValue::Uint(inner_cast(*value)),
                    ScalarKind::Float => ScalarValue::Float(inner_cast(*value)),
                    ScalarKind::Bool => ScalarValue::Bool(inner_cast::<u64>(*value) != 0),
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
                    *component = self.cast(*component, kind)?;
                }
            }
        }

        Ok(self.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner,
        }))
    }

    fn unary_op(
        &mut self,
        op: UnaryOperator,
        constant: Handle<Constant>,
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
                    *component = self.unary_op(op, *component)?
                }
            }
        }

        Ok(self.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner,
        }))
    }

    fn binary_op(
        &mut self,
        op: BinaryOperator,
        left: Handle<Constant>,
        right: Handle<Constant>,
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
        let left = &self.constants[left].inner;
        let right = &self.constants[right].inner;

        let inner = match (left, right) {
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
                                BinaryOperator::Add => a + b,
                                BinaryOperator::Subtract => a - b,
                                BinaryOperator::Multiply => a * b,
                                BinaryOperator::Divide => a / b,
                                BinaryOperator::Modulo => a % b,
                                BinaryOperator::And => a & b,
                                BinaryOperator::ExclusiveOr => a ^ b,
                                BinaryOperator::InclusiveOr => a | b,
                                BinaryOperator::ShiftLeft => a << b,
                                BinaryOperator::ShiftRight => a >> b,
                                _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                            })
                        }
                        (ScalarValue::Uint(a), ScalarValue::Uint(b)) => {
                            ScalarValue::Uint(match op {
                                BinaryOperator::Add => a + b,
                                BinaryOperator::Subtract => a - b,
                                BinaryOperator::Multiply => a * b,
                                BinaryOperator::Divide => a / b,
                                BinaryOperator::Modulo => a % b,
                                BinaryOperator::And => a & b,
                                BinaryOperator::ExclusiveOr => a ^ b,
                                BinaryOperator::InclusiveOr => a | b,
                                BinaryOperator::ShiftLeft => a << b,
                                BinaryOperator::ShiftRight => a >> b,
                                _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
                            })
                        }
                        (ScalarValue::Float(a), ScalarValue::Float(b)) => {
                            ScalarValue::Float(match op {
                                BinaryOperator::Add => a + b,
                                BinaryOperator::Subtract => a - b,
                                BinaryOperator::Multiply => a * b,
                                BinaryOperator::Divide => a / b,
                                BinaryOperator::Modulo => a % b,
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
            _ => return Err(ConstantSolvingError::InvalidBinaryOpArgs),
        };

        Ok(self.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::{
        Arena, Constant, ConstantInner, Expression, ScalarKind, ScalarValue, Type, TypeInner,
        UnaryOperator, VectorSize,
    };

    use super::ConstantSolver;

    #[test]
    fn unary_op() {
        let mut types = Arena::new();
        let mut expressions = Arena::new();
        let mut constants = Arena::new();

        let vec_ty = types.append(Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Bi,
                kind: ScalarKind::Sint,
                width: 4,
            },
        });

        let h = constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Sint(4),
            },
        });

        let h1 = constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Sint(8),
            },
        });

        let vec_h = constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Composite {
                ty: vec_ty,
                components: vec![h, h1],
            },
        });

        let expr = expressions.append(Expression::Constant(h));
        let expr1 = expressions.append(Expression::Constant(vec_h));

        let root1 = expressions.append(Expression::Unary {
            op: UnaryOperator::Negate,
            expr,
        });

        let root2 = expressions.append(Expression::Unary {
            op: UnaryOperator::Not,
            expr,
        });

        let root3 = expressions.append(Expression::Unary {
            op: UnaryOperator::Not,
            expr: expr1,
        });

        let mut solver = ConstantSolver {
            types: &types,
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

        let h = constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Sint(4),
            },
        });

        let expr = expressions.append(Expression::Constant(h));

        let root = expressions.append(Expression::As {
            expr,
            kind: ScalarKind::Bool,
            convert: true,
        });

        let mut solver = ConstantSolver {
            types: &Arena::new(),
            expressions: &expressions,
            constants: &mut constants,
        };

        let res = solver.solve(root).unwrap();

        assert_eq!(
            constants[res].inner,
            ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Bool(true),
            },
        );
    }

    #[test]
    fn access() {
        let mut types = Arena::new();
        let mut expressions = Arena::new();
        let mut constants = Arena::new();

        let matrix_ty = types.append(Type {
            name: None,
            inner: TypeInner::Matrix {
                columns: VectorSize::Bi,
                rows: VectorSize::Tri,
                width: 4,
            },
        });

        let vec_ty = types.append(Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Tri,
                kind: ScalarKind::Float,
                width: 4,
            },
        });

        let mut vec1_components = Vec::with_capacity(3);
        let mut vec2_components = Vec::with_capacity(3);

        for i in 0..3 {
            let h = constants.append(Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Scalar {
                    width: 4,
                    value: ScalarValue::Float(i as f64),
                },
            });

            vec1_components.push(h)
        }

        for i in 3..6 {
            let h = constants.append(Constant {
                name: None,
                specialization: None,
                inner: ConstantInner::Scalar {
                    width: 4,
                    value: ScalarValue::Float(i as f64),
                },
            });

            vec2_components.push(h)
        }

        let vec1 = constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Composite {
                ty: vec_ty,
                components: vec1_components,
            },
        });

        let vec2 = constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Composite {
                ty: vec_ty,
                components: vec2_components,
            },
        });

        let h = constants.append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Composite {
                ty: matrix_ty,
                components: vec![vec1, vec2],
            },
        });

        let base = expressions.append(Expression::Constant(h));
        let root1 = expressions.append(Expression::AccessIndex { base, index: 1 });
        let root2 = expressions.append(Expression::AccessIndex {
            base: root1,
            index: 2,
        });

        let mut solver = ConstantSolver {
            types: &types,
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
