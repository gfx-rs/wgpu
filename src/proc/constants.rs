use crate::{
    arena::{Arena, Handle},
    ArraySize, Constant, ConstantInner, Expression, ScalarKind, ScalarValue, Type,
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
            Expression::Unary { .. } => todo!(),
            Expression::Binary { .. } => todo!(),
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
                    crate::ConstantInner::Scalar { .. } => {
                        Err(ConstantSolvingError::InvalidArrayLengthArg)
                    }
                    crate::ConstantInner::Composite { ty, .. } => match self.types[ty].inner {
                        crate::TypeInner::Array { size, .. } => match size {
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
            Expression::ImageSample { .. } => Err(ConstantSolvingError::ImageExpression),
            Expression::ImageLoad { .. } => Err(ConstantSolvingError::ImageExpression),
        }
    }

    fn access(
        &mut self,
        base: Handle<Expression>,
        index: usize,
    ) -> Result<Handle<Constant>, ConstantSolvingError> {
        let base = self.solve(base)?;

        match self.constants[base].inner {
            crate::ConstantInner::Scalar { .. } => Err(ConstantSolvingError::InvalidAccessBase),
            crate::ConstantInner::Composite { ty, ref components } => match self.types[ty].inner {
                crate::TypeInner::Vector { size, .. } => {
                    if size as usize <= index {
                        Err(ConstantSolvingError::InvalidAccessIndex)
                    } else {
                        Ok(components[index])
                    }
                }
                crate::TypeInner::Matrix { .. } => todo!(),
                crate::TypeInner::Array { size, .. } => match size {
                    ArraySize::Constant(constant) => {
                        let size = self.constant_index(constant)?;

                        if size <= index {
                            Err(ConstantSolvingError::InvalidAccessIndex)
                        } else {
                            Ok(components[index])
                        }
                    }
                    ArraySize::Dynamic => Err(ConstantSolvingError::ArrayLengthDynamic),
                },
                crate::TypeInner::Struct { ref members, .. } => {
                    if members.len() <= index {
                        Err(ConstantSolvingError::InvalidAccessIndex)
                    } else {
                        Ok(components[index])
                    }
                }
                _ => Err(ConstantSolvingError::InvalidAccessBase),
            },
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
                let intial = value.clone();

                match kind {
                    ScalarKind::Sint => *value = ScalarValue::Sint(inner_cast(intial)),
                    ScalarKind::Uint => *value = ScalarValue::Uint(inner_cast(intial)),
                    ScalarKind::Float => *value = ScalarValue::Float(inner_cast(intial)),
                    ScalarKind::Bool => *value = ScalarValue::Bool(inner_cast::<u64>(intial) == 0),
                }
            }
            ConstantInner::Composite {
                ty,
                ref mut components,
            } => {
                match self.types[ty].inner {
                    crate::TypeInner::Vector { .. } | crate::TypeInner::Matrix { .. } => (),
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
}
