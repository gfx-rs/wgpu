use crate::{
    arena::{Arena, Handle},
    proc::TypeResolution,
};

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ComposeError {
    #[error("Compose type {0:?} doesn't exist")]
    TypeDoesntExist(Handle<crate::Type>),
    #[error("Composing of type {0:?} can't be done")]
    Type(Handle<crate::Type>),
    #[error("Composing expects {expected} components but {given} were given")]
    ComponentCount { given: u32, expected: u32 },
    #[error("Composing {index}'s component type is not expected")]
    ComponentType { index: u32 },
}

pub fn validate_compose(
    self_ty_handle: Handle<crate::Type>,
    constant_arena: &Arena<crate::Constant>,
    type_arena: &Arena<crate::Type>,
    component_resolutions: impl ExactSizeIterator<Item = TypeResolution>,
) -> Result<(), ComposeError> {
    use crate::TypeInner as Ti;

    let self_ty = type_arena
        .try_get(self_ty_handle)
        .ok_or(ComposeError::TypeDoesntExist(self_ty_handle))?;
    match self_ty.inner {
        // vectors are composed from scalars or other vectors
        Ti::Vector { size, kind, width } => {
            let mut total = 0;
            for (index, comp_res) in component_resolutions.enumerate() {
                total += match *comp_res.inner_with(type_arena) {
                    Ti::Scalar {
                        kind: comp_kind,
                        width: comp_width,
                    } if comp_kind == kind && comp_width == width => 1,
                    Ti::Vector {
                        size: comp_size,
                        kind: comp_kind,
                        width: comp_width,
                    } if comp_kind == kind && comp_width == width => comp_size as u32,
                    ref other => {
                        log::error!("Vector component[{}] type {:?}", index, other);
                        return Err(ComposeError::ComponentType {
                            index: index as u32,
                        });
                    }
                };
            }
            if size as u32 != total {
                return Err(ComposeError::ComponentCount {
                    expected: size as u32,
                    given: total,
                });
            }
        }
        // matrix are composed from column vectors
        Ti::Matrix {
            columns,
            rows,
            width,
        } => {
            let inner = Ti::Vector {
                size: rows,
                kind: crate::ScalarKind::Float,
                width,
            };
            if columns as usize != component_resolutions.len() {
                return Err(ComposeError::ComponentCount {
                    expected: columns as u32,
                    given: component_resolutions.len() as u32,
                });
            }
            for (index, comp_res) in component_resolutions.enumerate() {
                if comp_res.inner_with(type_arena) != &inner {
                    log::error!("Matrix component[{}] type {:?}", index, comp_res);
                    return Err(ComposeError::ComponentType {
                        index: index as u32,
                    });
                }
            }
        }
        Ti::Array {
            base,
            size: crate::ArraySize::Constant(handle),
            stride: _,
        } => {
            let count = constant_arena[handle].to_array_length().unwrap();
            if count as usize != component_resolutions.len() {
                return Err(ComposeError::ComponentCount {
                    expected: count,
                    given: component_resolutions.len() as u32,
                });
            }
            for (index, comp_res) in component_resolutions.enumerate() {
                if comp_res.inner_with(type_arena) != &type_arena[base].inner {
                    log::error!("Array component[{}] type {:?}", index, comp_res);
                    return Err(ComposeError::ComponentType {
                        index: index as u32,
                    });
                }
            }
        }
        Ti::Struct { ref members, .. } => {
            if members.len() != component_resolutions.len() {
                return Err(ComposeError::ComponentCount {
                    given: component_resolutions.len() as u32,
                    expected: members.len() as u32,
                });
            }
            for (index, (member, comp_res)) in members.iter().zip(component_resolutions).enumerate()
            {
                if comp_res.inner_with(type_arena) != &type_arena[member.ty].inner {
                    log::error!("Struct component[{}] type {:?}", index, comp_res);
                    return Err(ComposeError::ComponentType {
                        index: index as u32,
                    });
                }
            }
        }
        ref other => {
            log::error!("Composing of {:?}", other);
            return Err(ComposeError::Type(self_ty_handle));
        }
    }

    Ok(())
}
