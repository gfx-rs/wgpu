use crate::arena::{Arena, Handle};

use thiserror::Error;

pub struct Typifier {
    types: Vec<Handle<crate::Type>>,
}

#[derive(Clone, Debug, Error)]
pub enum ResolveError {
    #[error("Invalid index into array")]
    InvalidAccessIndex,
    #[error("Function {name} not defined")]
    FunctionNotDefined { name: String },
    #[error("Function without return type")]
    FunctionReturnsVoid,
}

impl Typifier {
    pub fn new() -> Self {
        Typifier { types: Vec::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        arena: &mut Arena<crate::Type>,
        constants: &Arena<crate::Constant>,
        global_vars: &Arena<crate::GlobalVariable>,
        local_vars: &Arena<crate::LocalVariable>,
        functions: &Arena<crate::Function>,
        parameter_types: &[Handle<crate::Type>],
    ) -> Result<Handle<crate::Type>, ResolveError> {
        #[derive(Debug)]
        enum Resolution {
            Handle(crate::Handle<crate::Type>),
            Value(crate::TypeInner),
        }

        if self.types.len() <= expr_handle.index() {
            for (eh, expr) in expressions.iter().skip(self.types.len()) {
                let resolution = match *expr {
                    crate::Expression::Access { base, .. } => {
                        match arena[self.types[base.index()]].inner {
                            crate::TypeInner::Array { base, .. } => Resolution::Handle(base),
                            ref other => panic!("Can't access into {:?}", other),
                        }
                    }
                    crate::Expression::AccessIndex { base, index } => {
                        match arena[self.types[base.index()]].inner {
                            crate::TypeInner::Vector { size, kind, width } => {
                                if index >= size as u32 {
                                    return Err(ResolveError::InvalidAccessIndex);
                                }
                                Resolution::Value(crate::TypeInner::Scalar { kind, width })
                            }
                            crate::TypeInner::Matrix {
                                columns,
                                rows,
                                kind,
                                width,
                            } => {
                                if index >= columns as u32 {
                                    return Err(ResolveError::InvalidAccessIndex);
                                }
                                Resolution::Value(crate::TypeInner::Vector {
                                    size: rows,
                                    kind,
                                    width,
                                })
                            }
                            crate::TypeInner::Array { base, .. } => Resolution::Handle(base),
                            crate::TypeInner::Struct { ref members } => {
                                let member = members
                                    .get(index as usize)
                                    .ok_or(ResolveError::InvalidAccessIndex)?;
                                Resolution::Handle(member.ty)
                            }
                            ref other => panic!("Can't access into {:?}", other),
                        }
                    }
                    crate::Expression::Constant(h) => Resolution::Handle(constants[h].ty),
                    crate::Expression::Compose { ty, .. } => Resolution::Handle(ty),
                    crate::Expression::FunctionParameter(index) => {
                        Resolution::Handle(parameter_types[index as usize])
                    }
                    crate::Expression::GlobalVariable(h) => Resolution::Handle(global_vars[h].ty),
                    crate::Expression::LocalVariable(h) => Resolution::Handle(local_vars[h].ty),
                    crate::Expression::Load { .. } => unimplemented!(),
                    crate::Expression::ImageSample { image, .. }
                    | crate::Expression::ImageLoad { image, .. } => {
                        let image = self.resolve(
                            image,
                            expressions,
                            arena,
                            constants,
                            global_vars,
                            local_vars,
                            functions,
                            parameter_types,
                        )?;

                        Resolution::Value(match arena[image].inner {
                            crate::TypeInner::Image {
                                kind,
                                class: crate::ImageClass::Depth,
                                ..
                            } => crate::TypeInner::Scalar { kind, width: 4 },
                            crate::TypeInner::Image { kind, .. } => crate::TypeInner::Vector {
                                kind,
                                width: 4,
                                size: crate::VectorSize::Quad,
                            },
                            _ => unreachable!(),
                        })
                    }
                    crate::Expression::Unary { expr, .. } => {
                        Resolution::Handle(self.types[expr.index()])
                    }
                    crate::Expression::Binary { op, left, right } => match op {
                        crate::BinaryOperator::Add
                        | crate::BinaryOperator::Subtract
                        | crate::BinaryOperator::Divide
                        | crate::BinaryOperator::Modulo => {
                            Resolution::Handle(self.types[left.index()])
                        }
                        crate::BinaryOperator::Multiply => {
                            let ty_left = self.types[left.index()];
                            let ty_right = self.types[right.index()];
                            if ty_left == ty_right {
                                Resolution::Handle(ty_left)
                            } else if let crate::TypeInner::Scalar { .. } = arena[ty_right].inner {
                                Resolution::Handle(ty_left)
                            } else if let crate::TypeInner::Scalar { .. } = arena[ty_left].inner {
                                Resolution::Handle(ty_right)
                            } else if let crate::TypeInner::Matrix {
                                columns,
                                kind,
                                width,
                                ..
                            } = arena[ty_left].inner
                            {
                                Resolution::Value(crate::TypeInner::Vector {
                                    size: columns,
                                    kind,
                                    width,
                                })
                            } else {
                                panic!(
                                    "Incompatible arguments {:?} x {:?}",
                                    arena[ty_left], arena[ty_right]
                                );
                            }
                        }
                        crate::BinaryOperator::Equal
                        | crate::BinaryOperator::NotEqual
                        | crate::BinaryOperator::Less
                        | crate::BinaryOperator::LessEqual
                        | crate::BinaryOperator::Greater
                        | crate::BinaryOperator::GreaterEqual
                        | crate::BinaryOperator::LogicalAnd
                        | crate::BinaryOperator::LogicalOr => {
                            Resolution::Handle(self.types[left.index()])
                        }
                        crate::BinaryOperator::And
                        | crate::BinaryOperator::ExclusiveOr
                        | crate::BinaryOperator::InclusiveOr
                        | crate::BinaryOperator::ShiftLeftLogical
                        | crate::BinaryOperator::ShiftRightLogical
                        | crate::BinaryOperator::ShiftRightArithmetic => {
                            Resolution::Handle(self.types[left.index()])
                        }
                    },
                    crate::Expression::Intrinsic { .. } => unimplemented!(),
                    crate::Expression::Transpose(expr) => {
                        let ty_handle = self.types[expr.index()];
                        match arena[ty_handle].inner {
                            crate::TypeInner::Matrix {
                                columns,
                                rows,
                                kind,
                                width,
                            } => Resolution::Value(crate::TypeInner::Matrix {
                                columns: rows,
                                rows: columns,
                                kind,
                                width,
                            }),
                            ref other => panic!("incompatible transpose of {:?}", other),
                        }
                    }
                    crate::Expression::DotProduct(left_expr, _) => {
                        let left_ty = self.types[left_expr.index()];
                        match arena[left_ty].inner {
                            crate::TypeInner::Vector {
                                kind,
                                size: _,
                                width,
                            } => Resolution::Value(crate::TypeInner::Scalar { kind, width }),
                            ref other => panic!("incompatible dot of {:?}", other),
                        }
                    }
                    crate::Expression::CrossProduct(_, _) => unimplemented!(),
                    crate::Expression::As(expr, kind) => {
                        let ty_handle = self.types[expr.index()];
                        match arena[ty_handle].inner {
                            crate::TypeInner::Scalar { kind: _, width } => {
                                Resolution::Value(crate::TypeInner::Scalar { kind, width })
                            }
                            crate::TypeInner::Vector {
                                kind: _,
                                size,
                                width,
                            } => Resolution::Value(crate::TypeInner::Vector { kind, size, width }),
                            ref other => panic!("incompatible as of {:?}", other),
                        }
                    }
                    crate::Expression::Derivative { .. } => unimplemented!(),
                    crate::Expression::Call {
                        origin: crate::FunctionOrigin::External(ref name),
                        ref arguments,
                    } => match name.as_str() {
                        "distance" | "length" | "dot" => {
                            let ty_handle = self.types[arguments[0].index()];
                            match arena[ty_handle].inner {
                                crate::TypeInner::Vector { kind, width, .. } => {
                                    Resolution::Value(crate::TypeInner::Scalar { kind, width })
                                }
                                ref other => panic!("Unexpected argument {:?}", other),
                            }
                        }
                        "normalize" | "fclamp" | "max" | "reflect" | "pow" | "clamp" | "mix" => {
                            Resolution::Handle(self.types[arguments[0].index()])
                        }
                        _ => return Err(ResolveError::FunctionNotDefined { name: name.clone() }),
                    },
                    crate::Expression::Call {
                        origin: crate::FunctionOrigin::Local(handle),
                        arguments: _,
                    } => {
                        let ty = functions[handle]
                            .return_type
                            .ok_or(ResolveError::FunctionReturnsVoid)?;
                        Resolution::Handle(ty)
                    }
                };
                log::debug!("Resolving {:?} = {:?} : {:?}", eh, expr, resolution);
                self.types.push(match resolution {
                    Resolution::Handle(h) => h,
                    Resolution::Value(inner) => arena
                        .fetch_if_or_append(crate::Type { name: None, inner }, |a, b| {
                            a.inner == b.inner
                        }),
                });
            }
        }
        Ok(self.types[expr_handle.index()])
    }
}

#[derive(Clone, Debug, Error)]
#[error("mismatched constant type {0:?} expected {1:?}")]
pub struct UnexpectedConstantTypeError(crate::ConstantInner, crate::TypeInner);

pub fn check_constant_types(
    inner: &crate::ConstantInner,
    type_inner: &crate::TypeInner,
) -> Result<(), UnexpectedConstantTypeError> {
    match (inner, type_inner) {
        (
            crate::ConstantInner::Sint(_),
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Sint,
                width: _,
            },
        ) => Ok(()),
        (
            crate::ConstantInner::Uint(_),
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Uint,
                width: _,
            },
        ) => Ok(()),
        (
            crate::ConstantInner::Float(_),
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Float,
                width: _,
            },
        ) => Ok(()),
        (
            crate::ConstantInner::Bool(_),
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Bool,
                width: _,
            },
        ) => Ok(()),
        (crate::ConstantInner::Composite(_inner), _) => Ok(()), // TODO recursively check composite types
        (other_inner, other_type_inner) => Err(UnexpectedConstantTypeError(
            other_inner.clone(),
            other_type_inner.clone(),
        )),
    }
}
