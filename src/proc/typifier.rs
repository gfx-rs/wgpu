use crate::arena::{Arena, Handle};

pub struct Typifier {
    types: Vec<Handle<crate::Type>>,
}

#[derive(Debug)]
pub enum ResolveError {
    InvalidAccessIndex,
}

impl Typifier {
    pub fn new() -> Self {
        Typifier {
            types: Vec::new(),
        }
    }

    pub fn resolve(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        types: &mut Arena<crate::Type>,
        constants: &Arena<crate::Constant>,
        global_vars: &Arena<crate::GlobalVariable>,
        local_vars: &Arena<crate::LocalVariable>,
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
                                    return Err(ResolveError::InvalidAccessIndex)
                                }
                                let inner = crate::TypeInner::Scalar { kind, width };
                                Self::deduce_type_handle(inner, types)
                            }
                            crate::TypeInner::Matrix { columns, rows, kind, width } => {
                                if index >= columns as u32 {
                                    return Err(ResolveError::InvalidAccessIndex)
                                }
                                let inner = crate::TypeInner::Vector { size: rows, kind, width };
                                Self::deduce_type_handle(inner, types)
                            }
                            crate::TypeInner::Array { base, .. } => base,
                            crate::TypeInner::Struct { ref members } => {
                                members.get(index as usize)
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
                    crate::Expression::ImageSample { .. } => unimplemented!(),
                    crate::Expression::Unary { expr, .. } => self.types[expr.index()],
                    crate::Expression::Binary { op, left, right } => {
                        match op {
                            crate::BinaryOperator::Add |
                            crate::BinaryOperator::Subtract |
                            crate::BinaryOperator::Divide |
                            crate::BinaryOperator::Modulo => {
                                self.types[left.index()]
                            }
                            crate::BinaryOperator::Multiply => {
                                let ty_left = self.types[left.index()];
                                let ty_right = self.types[right.index()];
                                if ty_left == ty_right {
                                    ty_left
                                } else if let crate::TypeInner::Scalar { .. } = types[ty_right].inner {
                                    ty_left
                                } else if let crate::TypeInner::Matrix { columns, kind, width, .. } = types[ty_left].inner {
                                    let inner = crate::TypeInner::Vector { size: columns, kind, width};
                                    Self::deduce_type_handle(inner, types)
                                } else {
                                    panic!("Incompatible arguments {:?} x {:?}", types[ty_left], types[ty_right]);
                                }
                            }
                            crate::BinaryOperator::Equal |
                            crate::BinaryOperator::NotEqual |
                            crate::BinaryOperator::Less |
                            crate::BinaryOperator::LessEqual |
                            crate::BinaryOperator::Greater |
                            crate::BinaryOperator::GreaterEqual |
                            crate::BinaryOperator::LogicalAnd |
                            crate::BinaryOperator::LogicalOr => {
                                self.types[left.index()]
                            }
                            crate::BinaryOperator::And |
                            crate::BinaryOperator::ExclusiveOr |
                            crate::BinaryOperator::InclusiveOr |
                            crate::BinaryOperator::ShiftLeftLogical |
                            crate::BinaryOperator::ShiftRightLogical |
                            crate::BinaryOperator::ShiftRightArithmetic => {
                                self.types[left.index()]
                            }
                        }
                    }
                    crate::Expression::Intrinsic { .. } => unimplemented!(),
                    crate::Expression::DotProduct(_, _) => unimplemented!(),
                    crate::Expression::CrossProduct(_, _) => unimplemented!(),
                    crate::Expression::Derivative { .. } => unimplemented!(),
                    crate::Expression::Call { ref name, ref arguments } => {
                        match name.as_str() {
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
                            _ => panic!("Unknown '{}' call", name),
                        }
                    }
                };
                log::debug!("Resolving {:?} = {:?} : {:?}", eh, expr, ty);
                self.types.push(ty);
            };
        }
        Ok(self.types[expr_handle.index()])
    }

    pub fn deduce_type_handle(
        inner: crate::TypeInner,
        arena: &mut Arena<crate::Type>,
    ) -> Handle<crate::Type> {
        if let Some((token, _)) = arena
            .iter()
            .find(|(_, ty)| ty.inner == inner)
        {
            return token;
        }
        arena.append(crate::Type {
            name: None,
            inner,
        })
    }
}
