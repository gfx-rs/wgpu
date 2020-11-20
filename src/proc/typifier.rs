use crate::arena::{Arena, Handle};

use thiserror::Error;

#[derive(Debug, PartialEq)]
enum Resolution {
    Handle(Handle<crate::Type>),
    Value(crate::TypeInner),
}

// Clone is only implemented for numeric variants of `TypeInner`.
impl Clone for Resolution {
    fn clone(&self) -> Self {
        match *self {
            Resolution::Handle(handle) => Resolution::Handle(handle),
            Resolution::Value(ref v) => Resolution::Value(match *v {
                crate::TypeInner::Scalar { kind, width } => {
                    crate::TypeInner::Scalar { kind, width }
                }
                crate::TypeInner::Vector { size, kind, width } => {
                    crate::TypeInner::Vector { size, kind, width }
                }
                crate::TypeInner::Matrix {
                    rows,
                    columns,
                    width,
                } => crate::TypeInner::Matrix {
                    rows,
                    columns,
                    width,
                },
                #[allow(clippy::panic)]
                _ => panic!("Unepxected clone type: {:?}", v),
            }),
        }
    }
}

#[derive(Debug)]
pub struct Typifier {
    resolutions: Vec<Resolution>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ResolveError {
    #[error("Invalid index into array")]
    InvalidAccessIndex,
    #[error("Function {name} not defined")]
    FunctionNotDefined { name: String },
    #[error("Function without return type")]
    FunctionReturnsVoid,
    #[error("Type is not found in the given immutable arena")]
    TypeNotFound,
    #[error("Incompatible operand: {op} {operand}")]
    IncompatibleOperand { op: String, operand: String },
    #[error("Incompatible operands: {left} {op} {right}")]
    IncompatibleOperands {
        op: String,
        left: String,
        right: String,
    },
}

pub struct ResolveContext<'a> {
    pub constants: &'a Arena<crate::Constant>,
    pub global_vars: &'a Arena<crate::GlobalVariable>,
    pub local_vars: &'a Arena<crate::LocalVariable>,
    pub functions: &'a Arena<crate::Function>,
    pub arguments: &'a [crate::FunctionArgument],
}

impl Typifier {
    pub fn new() -> Self {
        Typifier {
            resolutions: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.resolutions.clear()
    }

    pub fn get<'a>(
        &'a self,
        expr_handle: Handle<crate::Expression>,
        types: &'a Arena<crate::Type>,
    ) -> &'a crate::TypeInner {
        match self.resolutions[expr_handle.index()] {
            Resolution::Handle(ty_handle) => &types[ty_handle].inner,
            Resolution::Value(ref inner) => inner,
        }
    }

    pub fn get_handle(
        &self,
        expr_handle: Handle<crate::Expression>,
    ) -> Option<Handle<crate::Type>> {
        match self.resolutions[expr_handle.index()] {
            Resolution::Handle(ty_handle) => Some(ty_handle),
            Resolution::Value(_) => None,
        }
    }

    fn resolve_impl(
        &self,
        expr: &crate::Expression,
        types: &Arena<crate::Type>,
        ctx: &ResolveContext,
    ) -> Result<Resolution, ResolveError> {
        Ok(match *expr {
            crate::Expression::Access { base, .. } => match *self.get(base, types) {
                crate::TypeInner::Array { base, .. } => Resolution::Handle(base),
                crate::TypeInner::Vector {
                    size: _,
                    kind,
                    width,
                } => Resolution::Value(crate::TypeInner::Scalar { kind, width }),
                crate::TypeInner::Matrix {
                    rows: size,
                    columns: _,
                    width,
                } => Resolution::Value(crate::TypeInner::Vector {
                    size,
                    kind: crate::ScalarKind::Float,
                    width,
                }),
                ref other => {
                    return Err(ResolveError::IncompatibleOperand {
                        op: "access".to_string(),
                        operand: format!("{:?}", other),
                    })
                }
            },
            crate::Expression::AccessIndex { base, index } => match *self.get(base, types) {
                crate::TypeInner::Vector { size, kind, width } => {
                    if index >= size as u32 {
                        return Err(ResolveError::InvalidAccessIndex);
                    }
                    Resolution::Value(crate::TypeInner::Scalar { kind, width })
                }
                crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                } => {
                    if index >= columns as u32 {
                        return Err(ResolveError::InvalidAccessIndex);
                    }
                    Resolution::Value(crate::TypeInner::Vector {
                        size: rows,
                        kind: crate::ScalarKind::Float,
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
                ref other => {
                    return Err(ResolveError::IncompatibleOperand {
                        op: "access index".to_string(),
                        operand: format!("{:?}", other),
                    })
                }
            },
            crate::Expression::Constant(h) => Resolution::Handle(ctx.constants[h].ty),
            crate::Expression::Compose { ty, .. } => Resolution::Handle(ty),
            crate::Expression::FunctionArgument(index) => {
                Resolution::Handle(ctx.arguments[index as usize].ty)
            }
            crate::Expression::GlobalVariable(h) => Resolution::Handle(ctx.global_vars[h].ty),
            crate::Expression::LocalVariable(h) => Resolution::Handle(ctx.local_vars[h].ty),
            crate::Expression::Load { .. } => unimplemented!(),
            crate::Expression::ImageSample { image, .. }
            | crate::Expression::ImageLoad { image, .. } => match *self.get(image, types) {
                crate::TypeInner::Image { class, .. } => Resolution::Value(match class {
                    crate::ImageClass::Depth => crate::TypeInner::Scalar {
                        kind: crate::ScalarKind::Float,
                        width: 4,
                    },
                    crate::ImageClass::Sampled { kind, multi: _ } => crate::TypeInner::Vector {
                        kind,
                        width: 4,
                        size: crate::VectorSize::Quad,
                    },
                    crate::ImageClass::Storage(format) => crate::TypeInner::Vector {
                        kind: format.into(),
                        width: 4,
                        size: crate::VectorSize::Quad,
                    },
                }),
                _ => unreachable!(),
            },
            crate::Expression::Unary { expr, .. } => self.resolutions[expr.index()].clone(),
            crate::Expression::Binary { op, left, right } => match op {
                crate::BinaryOperator::Add
                | crate::BinaryOperator::Subtract
                | crate::BinaryOperator::Divide
                | crate::BinaryOperator::Modulo => self.resolutions[left.index()].clone(),
                crate::BinaryOperator::Multiply => {
                    let ty_left = self.get(left, types);
                    let ty_right = self.get(right, types);
                    if ty_left == ty_right {
                        self.resolutions[left.index()].clone()
                    } else if let crate::TypeInner::Scalar { .. } = *ty_right {
                        self.resolutions[left.index()].clone()
                    } else {
                        match *ty_left {
                            crate::TypeInner::Scalar { .. } => {
                                self.resolutions[right.index()].clone()
                            }
                            crate::TypeInner::Matrix {
                                columns,
                                rows: _,
                                width,
                            } => Resolution::Value(crate::TypeInner::Vector {
                                size: columns,
                                kind: crate::ScalarKind::Float,
                                width,
                            }),
                            _ => {
                                return Err(ResolveError::IncompatibleOperands {
                                    op: "x".to_string(),
                                    left: format!("{:?}", ty_left),
                                    right: format!("{:?}", ty_right),
                                })
                            }
                        }
                    }
                }
                crate::BinaryOperator::Equal
                | crate::BinaryOperator::NotEqual
                | crate::BinaryOperator::Less
                | crate::BinaryOperator::LessEqual
                | crate::BinaryOperator::Greater
                | crate::BinaryOperator::GreaterEqual
                | crate::BinaryOperator::LogicalAnd
                | crate::BinaryOperator::LogicalOr => self.resolutions[left.index()].clone(),
                crate::BinaryOperator::And
                | crate::BinaryOperator::ExclusiveOr
                | crate::BinaryOperator::InclusiveOr
                | crate::BinaryOperator::ShiftLeft
                | crate::BinaryOperator::ShiftRight => self.resolutions[left.index()].clone(),
            },
            crate::Expression::Select { accept, .. } => self.resolutions[accept.index()].clone(),
            crate::Expression::Intrinsic { .. } => unimplemented!(),
            crate::Expression::Transpose(expr) => match *self.get(expr, types) {
                crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                } => Resolution::Value(crate::TypeInner::Matrix {
                    columns: rows,
                    rows: columns,
                    width,
                }),
                ref other => {
                    return Err(ResolveError::IncompatibleOperand {
                        op: "transpose".to_string(),
                        operand: format!("{:?}", other),
                    })
                }
            },
            crate::Expression::DotProduct(left_expr, _) => match *self.get(left_expr, types) {
                crate::TypeInner::Vector {
                    kind,
                    size: _,
                    width,
                } => Resolution::Value(crate::TypeInner::Scalar { kind, width }),
                ref other => {
                    return Err(ResolveError::IncompatibleOperand {
                        op: "dot product".to_string(),
                        operand: format!("{:?}", other),
                    })
                }
            },
            crate::Expression::CrossProduct(_, _) => unimplemented!(),
            crate::Expression::As {
                expr,
                kind,
                convert: _,
            } => match *self.get(expr, types) {
                crate::TypeInner::Scalar { kind: _, width } => {
                    Resolution::Value(crate::TypeInner::Scalar { kind, width })
                }
                crate::TypeInner::Vector {
                    kind: _,
                    size,
                    width,
                } => Resolution::Value(crate::TypeInner::Vector { kind, size, width }),
                ref other => {
                    return Err(ResolveError::IncompatibleOperand {
                        op: "as".to_string(),
                        operand: format!("{:?}", other),
                    })
                }
            },
            crate::Expression::Derivative { .. } => unimplemented!(),
            crate::Expression::Call {
                origin: crate::FunctionOrigin::External(ref name),
                ref arguments,
            } => match name.as_str() {
                "distance" | "length" => match *self.get(arguments[0], types) {
                    crate::TypeInner::Vector { kind, width, .. }
                    | crate::TypeInner::Scalar { kind, width } => {
                        Resolution::Value(crate::TypeInner::Scalar { kind, width })
                    }
                    ref other => {
                        return Err(ResolveError::IncompatibleOperand {
                            op: name.clone(),
                            operand: format!("{:?}", other),
                        })
                    }
                },
                "dot" => match *self.get(arguments[0], types) {
                    crate::TypeInner::Vector { kind, width, .. } => {
                        Resolution::Value(crate::TypeInner::Scalar { kind, width })
                    }
                    ref other => {
                        return Err(ResolveError::IncompatibleOperand {
                            op: name.clone(),
                            operand: format!("{:?}", other),
                        })
                    }
                },
                //Note: `cross` is here too, we still need to figure out what to do with it
                "abs" | "atan2" | "cos" | "sin" | "floor" | "inverse" | "normalize" | "min"
                | "max" | "reflect" | "pow" | "clamp" | "fclamp" | "mix" | "step"
                | "smoothstep" | "cross" => self.resolutions[arguments[0].index()].clone(),
                _ => return Err(ResolveError::FunctionNotDefined { name: name.clone() }),
            },
            crate::Expression::Call {
                origin: crate::FunctionOrigin::Local(handle),
                arguments: _,
            } => {
                let ty = ctx.functions[handle]
                    .return_type
                    .ok_or(ResolveError::FunctionReturnsVoid)?;
                Resolution::Handle(ty)
            }
            crate::Expression::ArrayLength(_) => Resolution::Value(crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Uint,
                width: 4,
            }),
        })
    }

    pub fn grow(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        types: &mut Arena<crate::Type>,
        ctx: &ResolveContext,
    ) -> Result<(), ResolveError> {
        if self.resolutions.len() <= expr_handle.index() {
            for (eh, expr) in expressions.iter().skip(self.resolutions.len()) {
                let resolution = self.resolve_impl(expr, types, ctx)?;
                log::debug!("Resolving {:?} = {:?} : {:?}", eh, expr, resolution);

                let ty_handle = match resolution {
                    Resolution::Handle(h) => h,
                    Resolution::Value(inner) => types
                        .fetch_if_or_append(crate::Type { name: None, inner }, |a, b| {
                            a.inner == b.inner
                        }),
                };
                self.resolutions.push(Resolution::Handle(ty_handle));
            }
        }
        Ok(())
    }

    pub fn resolve_all(
        &mut self,
        expressions: &Arena<crate::Expression>,
        types: &Arena<crate::Type>,
        ctx: &ResolveContext,
    ) -> Result<(), ResolveError> {
        self.clear();
        for (_, expr) in expressions.iter() {
            let resolution = self.resolve_impl(expr, types, ctx)?;
            self.resolutions.push(resolution);
        }
        Ok(())
    }
}

pub fn check_constant_type(inner: &crate::ConstantInner, type_inner: &crate::TypeInner) -> bool {
    match (inner, type_inner) {
        (
            crate::ConstantInner::Sint(_),
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Sint,
                width: _,
            },
        ) => true,
        (
            crate::ConstantInner::Uint(_),
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Uint,
                width: _,
            },
        ) => true,
        (
            crate::ConstantInner::Float(_),
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Float,
                width: _,
            },
        ) => true,
        (
            crate::ConstantInner::Bool(_),
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Bool,
                width: _,
            },
        ) => true,
        (crate::ConstantInner::Composite(_inner), _) => true, // TODO recursively check composite types
        (_, _) => false,
    }
}
