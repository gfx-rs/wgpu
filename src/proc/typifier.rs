use crate::{
    arena::{Arena, Handle},
    FastHashMap, Type, TypeInner, VectorSize,
};

pub struct Typifier {
    types: Vec<Handle<crate::Type>>,
}

#[derive(Debug)]
pub enum ResolveError {
    InvalidAccessIndex,
    FunctionNotDefined,
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
        types: &mut Arena<crate::Type>,
        constants: &Arena<crate::Constant>,
        global_vars: &Arena<crate::GlobalVariable>,
        local_vars: &Arena<crate::LocalVariable>,
        functions: &Arena<crate::Function>,
        function_lookup: &FastHashMap<String, Handle<crate::Function>>,
    ) -> Result<Handle<crate::Type>, ResolveError> {
        if self.types.len() <= expr_handle.index() {
            for (eh, expr) in expressions.iter().skip(self.types.len()) {
                let ty = match *expr {
                    crate::Expression::Access { base, .. } => {
                        match types[self.types[base.index()]].inner {
                            crate::TypeInner::Array { base, .. } => base,
                            ref other => panic!("Can't access into {:?}", other),
                        }
                    }
                    crate::Expression::AccessIndex { base, index } => {
                        match types[self.types[base.index()]].inner {
                            crate::TypeInner::Vector { size, kind, width } => {
                                if index >= size as u32 {
                                    return Err(ResolveError::InvalidAccessIndex);
                                }
                                let inner = crate::TypeInner::Scalar { kind, width };
                                Self::deduce_type_handle(inner, types)
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
                                let inner = crate::TypeInner::Vector {
                                    size: rows,
                                    kind,
                                    width,
                                };
                                Self::deduce_type_handle(inner, types)
                            }
                            crate::TypeInner::Array { base, .. } => base,
                            crate::TypeInner::Struct { ref members } => {
                                members
                                    .get(index as usize)
                                    .ok_or(ResolveError::InvalidAccessIndex)?
                                    .ty
                            }
                            ref other => panic!("Can't access into {:?}", other),
                        }
                    }
                    crate::Expression::Constant(h) => constants[h].ty,
                    crate::Expression::Compose { ty, .. } => ty,
                    crate::Expression::FunctionParameter(_) => unimplemented!(),
                    crate::Expression::GlobalVariable(h) => global_vars[h].ty,
                    crate::Expression::LocalVariable(h) => local_vars[h].ty,
                    crate::Expression::Load { .. } => unimplemented!(),
                    crate::Expression::ImageSample { image, .. } => {
                        let image = self.resolve(
                            image,
                            expressions,
                            types,
                            constants,
                            global_vars,
                            local_vars,
                            functions,
                            function_lookup,
                        )?;

                        let (kind, width) = match types[image].inner {
                            TypeInner::Image { base, .. } => match types[base].inner {
                                TypeInner::Scalar { kind, width } => (kind, width),
                                _ => unimplemented!(),
                            },
                            _ => unreachable!(),
                        };

                        types.fetch_or_append(Type {
                            name: None,
                            inner: TypeInner::Vector {
                                kind,
                                width,
                                size: VectorSize::Quad,
                            },
                        })
                    }
                    crate::Expression::Unary { expr, .. } => self.types[expr.index()],
                    crate::Expression::Binary { op, left, right } => match op {
                        crate::BinaryOperator::Add
                        | crate::BinaryOperator::Subtract
                        | crate::BinaryOperator::Divide
                        | crate::BinaryOperator::Modulo => self.types[left.index()],
                        crate::BinaryOperator::Multiply => {
                            let ty_left = self.types[left.index()];
                            let ty_right = self.types[right.index()];
                            if ty_left == ty_right {
                                ty_left
                            } else if let crate::TypeInner::Scalar { .. } = types[ty_right].inner {
                                ty_left
                            } else if let crate::TypeInner::Matrix {
                                columns,
                                kind,
                                width,
                                ..
                            } = types[ty_left].inner
                            {
                                let inner = crate::TypeInner::Vector {
                                    size: columns,
                                    kind,
                                    width,
                                };
                                Self::deduce_type_handle(inner, types)
                            } else {
                                panic!(
                                    "Incompatible arguments {:?} x {:?}",
                                    types[ty_left], types[ty_right]
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
                        | crate::BinaryOperator::LogicalOr => self.types[left.index()],
                        crate::BinaryOperator::And
                        | crate::BinaryOperator::ExclusiveOr
                        | crate::BinaryOperator::InclusiveOr
                        | crate::BinaryOperator::ShiftLeftLogical
                        | crate::BinaryOperator::ShiftRightLogical
                        | crate::BinaryOperator::ShiftRightArithmetic => self.types[left.index()],
                    },
                    crate::Expression::Intrinsic { .. } => unimplemented!(),
                    crate::Expression::DotProduct(_, _) => unimplemented!(),
                    crate::Expression::CrossProduct(_, _) => unimplemented!(),
                    crate::Expression::Derivative { .. } => unimplemented!(),
                    crate::Expression::Call {
                        ref name,
                        ref arguments,
                    } => match name.as_str() {
                        "distance" | "length" => {
                            let ty_handle = self.types[arguments[0].index()];
                            let inner = match types[ty_handle].inner {
                                crate::TypeInner::Vector { kind, width, .. } => {
                                    crate::TypeInner::Scalar { kind, width }
                                }
                                ref other => panic!("Unexpected argument {:?}", other),
                            };
                            Self::deduce_type_handle(inner, types)
                        }
                        "normalize" | "fclamp" => self.types[arguments[0].index()],
                        other => functions[*function_lookup
                            .get(other)
                            .ok_or(ResolveError::FunctionNotDefined)?]
                        .return_type
                        .ok_or(ResolveError::FunctionReturnsVoid)?,
                    },
                };
                log::debug!("Resolving {:?} = {:?} : {:?}", eh, expr, ty);
                self.types.push(ty);
            }
        }
        Ok(self.types[expr_handle.index()])
    }

    pub fn deduce_type_handle(
        inner: crate::TypeInner,
        arena: &mut Arena<crate::Type>,
    ) -> Handle<crate::Type> {
        if let Some((token, _)) = arena.iter().find(|(_, ty)| ty.inner == inner) {
            return token;
        }
        arena.append(crate::Type { name: None, inner })
    }
}

#[derive(Debug)]
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
