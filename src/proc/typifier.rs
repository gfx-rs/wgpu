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
        use crate::TypeInner as Ti;
        match *self {
            Resolution::Handle(handle) => Resolution::Handle(handle),
            Resolution::Value(ref v) => Resolution::Value(match *v {
                Ti::Scalar { kind, width } => Ti::Scalar { kind, width },
                Ti::Vector { size, kind, width } => Ti::Vector { size, kind, width },
                Ti::Matrix {
                    rows,
                    columns,
                    width,
                } => Ti::Matrix {
                    rows,
                    columns,
                    width,
                },
                Ti::Pointer { base, class } => Ti::Pointer { base, class },
                Ti::ValuePointer {
                    size,
                    kind,
                    width,
                    class,
                } => Ti::ValuePointer {
                    size,
                    kind,
                    width,
                    class,
                },
                _ => unreachable!("Unexpected clone type: {:?}", v),
            }),
        }
    }
}

/// Helper processor that derives the types of all expressions.
#[derive(Debug)]
pub struct Typifier {
    resolutions: Vec<Resolution>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ResolveError {
    #[error("Index {index} is out of bounds for expression {expr:?}")]
    OutOfBoundsIndex {
        expr: Handle<crate::Expression>,
        index: u32,
    },
    #[error("Invalid access into expression {expr:?}, indexed: {indexed}")]
    InvalidAccess {
        expr: Handle<crate::Expression>,
        indexed: bool,
    },
    #[error("Invalid sub-access into type {ty:?}, indexed: {indexed}")]
    InvalidSubAccess {
        ty: Handle<crate::Type>,
        indexed: bool,
    },
    #[error("Invalid pointer {0:?}")]
    InvalidPointer(Handle<crate::Expression>),
    #[error("Invalid image {0:?}")]
    InvalidImage(Handle<crate::Expression>),
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

#[derive(Clone, Debug, Error, PartialEq)]
#[error("Type resolution of {0:?} failed")]
pub struct TypifyError(Handle<crate::Expression>, #[source] ResolveError);

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

    pub fn try_get<'a>(
        &'a self,
        expr_handle: Handle<crate::Expression>,
        types: &'a Arena<crate::Type>,
    ) -> Option<&'a crate::TypeInner> {
        let resolution = self.resolutions.get(expr_handle.index())?;
        Some(match *resolution {
            Resolution::Handle(ty_handle) => &types[ty_handle].inner,
            Resolution::Value(ref inner) => inner,
        })
    }

    pub fn get_handle(
        &self,
        expr_handle: Handle<crate::Expression>,
    ) -> Result<Handle<crate::Type>, &crate::TypeInner> {
        match self.resolutions[expr_handle.index()] {
            Resolution::Handle(ty_handle) => Ok(ty_handle),
            Resolution::Value(ref inner) => Err(inner),
        }
    }

    fn resolve_impl(
        &self,
        expr: &crate::Expression,
        types: &Arena<crate::Type>,
        ctx: &ResolveContext,
    ) -> Result<Resolution, ResolveError> {
        use crate::TypeInner as Ti;
        Ok(match *expr {
            crate::Expression::Access { base, .. } => match *self.get(base, types) {
                Ti::Array { base, .. } => Resolution::Handle(base),
                Ti::Vector {
                    size: _,
                    kind,
                    width,
                } => Resolution::Value(Ti::Scalar { kind, width }),
                Ti::ValuePointer {
                    size: Some(_),
                    kind,
                    width,
                    class,
                } => Resolution::Value(Ti::ValuePointer {
                    size: None,
                    kind,
                    width,
                    class,
                }),
                Ti::Pointer { base, class } => Resolution::Value(match types[base].inner {
                    Ti::Array { base, .. } => Ti::Pointer { base, class },
                    Ti::Vector {
                        size: _,
                        kind,
                        width,
                    } => Ti::ValuePointer {
                        size: None,
                        kind,
                        width,
                        class,
                    },
                    ref other => {
                        log::error!("Access sub-type {:?}", other);
                        return Err(ResolveError::InvalidSubAccess {
                            ty: base,
                            indexed: false,
                        });
                    }
                }),
                ref other => {
                    log::error!("Access type {:?}", other);
                    return Err(ResolveError::InvalidAccess {
                        expr: base,
                        indexed: false,
                    });
                }
            },
            crate::Expression::AccessIndex { base, index } => match *self.get(base, types) {
                Ti::Vector { size, kind, width } => {
                    if index >= size as u32 {
                        return Err(ResolveError::OutOfBoundsIndex { expr: base, index });
                    }
                    Resolution::Value(Ti::Scalar { kind, width })
                }
                Ti::Matrix {
                    columns,
                    rows,
                    width,
                } => {
                    if index >= columns as u32 {
                        return Err(ResolveError::OutOfBoundsIndex { expr: base, index });
                    }
                    Resolution::Value(crate::TypeInner::Vector {
                        size: rows,
                        kind: crate::ScalarKind::Float,
                        width,
                    })
                }
                Ti::Array { base, .. } => Resolution::Handle(base),
                Ti::Struct {
                    block: _,
                    ref members,
                } => {
                    let member = members
                        .get(index as usize)
                        .ok_or(ResolveError::OutOfBoundsIndex { expr: base, index })?;
                    Resolution::Handle(member.ty)
                }
                Ti::ValuePointer {
                    size: Some(size),
                    kind,
                    width,
                    class,
                } => {
                    if index >= size as u32 {
                        return Err(ResolveError::OutOfBoundsIndex { expr: base, index });
                    }
                    Resolution::Value(Ti::ValuePointer {
                        size: None,
                        kind,
                        width,
                        class,
                    })
                }
                Ti::Pointer {
                    base: ty_base,
                    class,
                } => Resolution::Value(match types[ty_base].inner {
                    Ti::Array { base, .. } => Ti::Pointer { base, class },
                    Ti::Vector { size, kind, width } => {
                        if index >= size as u32 {
                            return Err(ResolveError::OutOfBoundsIndex { expr: base, index });
                        }
                        Ti::ValuePointer {
                            size: None,
                            kind,
                            width,
                            class,
                        }
                    }
                    Ti::Matrix {
                        rows,
                        columns,
                        width,
                    } => {
                        if index >= columns as u32 {
                            return Err(ResolveError::OutOfBoundsIndex { expr: base, index });
                        }
                        Ti::ValuePointer {
                            size: Some(rows),
                            kind: crate::ScalarKind::Float,
                            width,
                            class,
                        }
                    }
                    Ti::Struct {
                        block: _,
                        ref members,
                    } => {
                        let member = members
                            .get(index as usize)
                            .ok_or(ResolveError::OutOfBoundsIndex { expr: base, index })?;
                        Ti::Pointer {
                            base: member.ty,
                            class,
                        }
                    }
                    ref other => {
                        log::error!("Access index sub-type {:?}", other);
                        return Err(ResolveError::InvalidSubAccess {
                            ty: ty_base,
                            indexed: true,
                        });
                    }
                }),
                ref other => {
                    log::error!("Access index type {:?}", other);
                    return Err(ResolveError::InvalidAccess {
                        expr: base,
                        indexed: true,
                    });
                }
            },
            crate::Expression::Constant(h) => match ctx.constants[h].inner {
                crate::ConstantInner::Scalar { width, ref value } => {
                    Resolution::Value(Ti::Scalar {
                        kind: value.scalar_kind(),
                        width,
                    })
                }
                crate::ConstantInner::Composite { ty, components: _ } => Resolution::Handle(ty),
            },
            crate::Expression::Compose { ty, .. } => Resolution::Handle(ty),
            crate::Expression::FunctionArgument(index) => {
                Resolution::Handle(ctx.arguments[index as usize].ty)
            }
            crate::Expression::GlobalVariable(h) => {
                let var = &ctx.global_vars[h];
                if var.class == crate::StorageClass::Handle {
                    Resolution::Handle(var.ty)
                } else {
                    Resolution::Value(Ti::Pointer {
                        base: var.ty,
                        class: var.class,
                    })
                }
            }
            crate::Expression::LocalVariable(h) => {
                let var = &ctx.local_vars[h];
                Resolution::Value(Ti::Pointer {
                    base: var.ty,
                    class: crate::StorageClass::Function,
                })
            }
            crate::Expression::Load { pointer } => match *self.get(pointer, types) {
                Ti::Pointer { base, class: _ } => Resolution::Handle(base),
                Ti::ValuePointer {
                    size,
                    kind,
                    width,
                    class: _,
                } => Resolution::Value(match size {
                    Some(size) => Ti::Vector { size, kind, width },
                    None => Ti::Scalar { kind, width },
                }),
                ref other => {
                    log::error!("Pointer type {:?}", other);
                    return Err(ResolveError::InvalidPointer(pointer));
                }
            },
            crate::Expression::ImageSample { image, .. }
            | crate::Expression::ImageLoad { image, .. } => match *self.get(image, types) {
                Ti::Image { class, .. } => Resolution::Value(match class {
                    crate::ImageClass::Depth => Ti::Scalar {
                        kind: crate::ScalarKind::Float,
                        width: 4,
                    },
                    crate::ImageClass::Sampled { kind, multi: _ } => Ti::Vector {
                        kind,
                        width: 4,
                        size: crate::VectorSize::Quad,
                    },
                    crate::ImageClass::Storage(format) => Ti::Vector {
                        kind: format.into(),
                        width: 4,
                        size: crate::VectorSize::Quad,
                    },
                }),
                ref other => {
                    log::error!("Image type {:?}", other);
                    return Err(ResolveError::InvalidImage(image));
                }
            },
            crate::Expression::ImageQuery { image, query } => Resolution::Value(match query {
                crate::ImageQuery::Size { level: _ } => match *self.get(image, types) {
                    Ti::Image { dim, .. } => match dim {
                        crate::ImageDimension::D1 => Ti::Scalar {
                            kind: crate::ScalarKind::Sint,
                            width: 4,
                        },
                        crate::ImageDimension::D2 => Ti::Vector {
                            size: crate::VectorSize::Bi,
                            kind: crate::ScalarKind::Sint,
                            width: 4,
                        },
                        crate::ImageDimension::D3 | crate::ImageDimension::Cube => Ti::Vector {
                            size: crate::VectorSize::Tri,
                            kind: crate::ScalarKind::Sint,
                            width: 4,
                        },
                    },
                    ref other => {
                        log::error!("Image type {:?}", other);
                        return Err(ResolveError::InvalidImage(image));
                    }
                },
                crate::ImageQuery::NumLevels
                | crate::ImageQuery::NumLayers
                | crate::ImageQuery::NumSamples => Ti::Scalar {
                    kind: crate::ScalarKind::Sint,
                    width: 4,
                },
            }),
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
                    } else if let Ti::Scalar { .. } = *ty_left {
                        self.resolutions[right.index()].clone()
                    } else if let Ti::Scalar { .. } = *ty_right {
                        self.resolutions[left.index()].clone()
                    } else if let Ti::Matrix {
                        columns: _,
                        rows,
                        width,
                    } = *ty_left
                    {
                        Resolution::Value(Ti::Vector {
                            size: rows,
                            kind: crate::ScalarKind::Float,
                            width,
                        })
                    } else if let Ti::Matrix {
                        columns,
                        rows: _,
                        width,
                    } = *ty_right
                    {
                        Resolution::Value(Ti::Vector {
                            size: columns,
                            kind: crate::ScalarKind::Float,
                            width,
                        })
                    } else {
                        return Err(ResolveError::IncompatibleOperands {
                            op: "x".to_string(),
                            left: format!("{:?}", ty_left),
                            right: format!("{:?}", ty_right),
                        });
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
                    let kind = crate::ScalarKind::Bool;
                    let width = 1;
                    let inner = match *self.get(left, types) {
                        Ti::Scalar { .. } => Ti::Scalar { kind, width },
                        Ti::Vector { size, .. } => Ti::Vector { size, kind, width },
                        ref other => {
                            return Err(ResolveError::IncompatibleOperand {
                                op: "logical".to_string(),
                                operand: format!("{:?}", other),
                            })
                        }
                    };
                    Resolution::Value(inner)
                }
                crate::BinaryOperator::And
                | crate::BinaryOperator::ExclusiveOr
                | crate::BinaryOperator::InclusiveOr
                | crate::BinaryOperator::ShiftLeft
                | crate::BinaryOperator::ShiftRight => self.resolutions[left.index()].clone(),
            },
            crate::Expression::Select { accept, .. } => self.resolutions[accept.index()].clone(),
            crate::Expression::Derivative { axis: _, expr } => {
                self.resolutions[expr.index()].clone()
            }
            crate::Expression::Relational { .. } => Resolution::Value(Ti::Scalar {
                kind: crate::ScalarKind::Bool,
                width: 4,
            }),
            crate::Expression::Math {
                fun,
                arg,
                arg1,
                arg2: _,
            } => {
                use crate::MathFunction as Mf;
                match fun {
                    // comparison
                    Mf::Abs |
                    Mf::Min |
                    Mf::Max |
                    Mf::Clamp |
                    // trigonometry
                    Mf::Cos |
                    Mf::Cosh |
                    Mf::Sin |
                    Mf::Sinh |
                    Mf::Tan |
                    Mf::Tanh |
                    Mf::Acos |
                    Mf::Asin |
                    Mf::Atan |
                    Mf::Atan2 |
                    // decomposition
                    Mf::Ceil |
                    Mf::Floor |
                    Mf::Round |
                    Mf::Fract |
                    Mf::Trunc |
                    Mf::Modf |
                    Mf::Frexp |
                    Mf::Ldexp |
                    // exponent
                    Mf::Exp |
                    Mf::Exp2 |
                    Mf::Log |
                    Mf::Log2 |
                    Mf::Pow => self.resolutions[arg.index()].clone(),
                    // geometry
                    Mf::Dot => match *self.get(arg, types) {
                        Ti::Vector {
                            kind,
                            size: _,
                            width,
                        } => Resolution::Value(Ti::Scalar { kind, width }),
                        ref other => {
                            return Err(ResolveError::IncompatibleOperand {
                                op: "dot product".to_string(),
                                operand: format!("{:?}", other),
                            })
                        }
                    },
                    Mf::Outer => {
                        let arg1 = arg1.ok_or_else(|| ResolveError::IncompatibleOperand {
                            op: "outer product".to_string(),
                            operand: "".to_string(),
                        })?;
                        match (self.get(arg, types), self.get(arg1,types)) {
                            (&Ti::Vector {kind: _, size: columns,width}, &Ti::Vector{ size: rows, .. }) => Resolution::Value(Ti::Matrix { columns, rows, width }),
                            (left, right) => {
                                return Err(ResolveError::IncompatibleOperands {
                                    op: "outer product".to_string(),
                                    left: format!("{:?}", left),
                                    right: format!("{:?}", right),
                                })
                            }
                        }
                    },
                    Mf::Cross => self.resolutions[arg.index()].clone(),
                    Mf::Distance |
                    Mf::Length => match *self.get(arg, types) {
                        Ti::Scalar {width,kind} |
                        Ti::Vector {width,kind,size:_} => Resolution::Value(Ti::Scalar { kind, width }),
                        ref other => {
                            return Err(ResolveError::IncompatibleOperand {
                                op: format!("{:?}", fun),
                                operand: format!("{:?}", other),
                            })
                        }
                    },
                    Mf::Normalize |
                    Mf::FaceForward |
                    Mf::Reflect => self.resolutions[arg.index()].clone(),
                    // computational
                    Mf::Sign |
                    Mf::Fma |
                    Mf::Mix |
                    Mf::Step |
                    Mf::SmoothStep |
                    Mf::Sqrt |
                    Mf::InverseSqrt => self.resolutions[arg.index()].clone(),
                    Mf::Transpose => match *self.get(arg, types) {
                        Ti::Matrix {
                            columns,
                            rows,
                            width,
                        } => Resolution::Value(Ti::Matrix {
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
                    Mf::Inverse => match *self.get(arg, types) {
                        Ti::Matrix {
                            columns,
                            rows,
                            width,
                        } if columns == rows => Resolution::Value(Ti::Matrix {
                            columns,
                            rows,
                            width,
                        }),
                        ref other => {
                            return Err(ResolveError::IncompatibleOperand {
                                op: "inverse".to_string(),
                                operand: format!("{:?}", other),
                            })
                        }
                    },
                    Mf::Determinant => match *self.get(arg, types) {
                        Ti::Matrix {
                            width,
                            ..
                        } => Resolution::Value(Ti::Scalar { kind: crate::ScalarKind::Float, width }),
                        ref other => {
                            return Err(ResolveError::IncompatibleOperand {
                                op: "determinant".to_string(),
                                operand: format!("{:?}", other),
                            })
                        }
                    },
                    // bits
                    Mf::CountOneBits |
                    Mf::ReverseBits => self.resolutions[arg.index()].clone(),
                }
            }
            crate::Expression::As {
                expr,
                kind,
                convert: _,
            } => match *self.get(expr, types) {
                Ti::Scalar { kind: _, width } => Resolution::Value(Ti::Scalar { kind, width }),
                Ti::Vector {
                    kind: _,
                    size,
                    width,
                } => Resolution::Value(Ti::Vector { kind, size, width }),
                ref other => {
                    return Err(ResolveError::IncompatibleOperand {
                        op: "as".to_string(),
                        operand: format!("{:?}", other),
                    })
                }
            },
            crate::Expression::Call(function) => {
                let result = ctx.functions[function]
                    .result
                    .as_ref()
                    .ok_or(ResolveError::FunctionReturnsVoid)?;
                Resolution::Handle(result.ty)
            }
            crate::Expression::ArrayLength(_) => Resolution::Value(Ti::Scalar {
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
                self.resolutions.push(resolution);
            }
        }
        Ok(())
    }

    pub fn resolve_all(
        &mut self,
        expressions: &Arena<crate::Expression>,
        types: &Arena<crate::Type>,
        ctx: &ResolveContext,
    ) -> Result<(), TypifyError> {
        self.clear();
        for (handle, expr) in expressions.iter() {
            let resolution = self
                .resolve_impl(expr, types, ctx)
                .map_err(|err| TypifyError(handle, err))?;
            self.resolutions.push(resolution);
        }
        Ok(())
    }
}
