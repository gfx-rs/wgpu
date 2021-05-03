use super::Error;
use crate::arena::{Arena, Handle};

fn make_scalar_inner(kind: crate::ScalarKind, width: crate::Bytes) -> crate::ConstantInner {
    crate::ConstantInner::Scalar {
        width,
        value: match kind {
            crate::ScalarKind::Uint => crate::ScalarValue::Uint(0),
            crate::ScalarKind::Sint => crate::ScalarValue::Sint(0),
            crate::ScalarKind::Float => crate::ScalarValue::Float(0.0),
            crate::ScalarKind::Bool => crate::ScalarValue::Bool(false),
        },
    }
}

pub fn generate_null_constant(
    ty: Handle<crate::Type>,
    type_arena: &mut Arena<crate::Type>,
    constant_arena: &mut Arena<crate::Constant>,
) -> Result<crate::ConstantInner, Error> {
    let inner = match type_arena[ty].inner {
        crate::TypeInner::Scalar { kind, width } => make_scalar_inner(kind, width),
        crate::TypeInner::Vector { size, kind, width } => {
            let mut components = Vec::with_capacity(size as usize);
            for _ in 0..size as usize {
                components.push(constant_arena.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner: make_scalar_inner(kind, width),
                }));
            }
            crate::ConstantInner::Composite { ty, components }
        }
        crate::TypeInner::Matrix {
            columns,
            rows,
            width,
        } => {
            let vector_ty = type_arena.fetch_or_append(crate::Type {
                name: None,
                inner: crate::TypeInner::Vector {
                    kind: crate::ScalarKind::Float,
                    size: rows,
                    width,
                },
            });
            let vector_inner = generate_null_constant(vector_ty, type_arena, constant_arena)?;
            let vector_handle = constant_arena.fetch_or_append(crate::Constant {
                name: None,
                specialization: None,
                inner: vector_inner,
            });
            crate::ConstantInner::Composite {
                ty,
                components: vec![vector_handle; columns as usize],
            }
        }
        crate::TypeInner::Struct { ref members, .. } => {
            let mut components = Vec::with_capacity(members.len());
            // copy out the types to avoid borrowing `members`
            let member_tys = members.iter().map(|member| member.ty).collect::<Vec<_>>();
            for member_ty in member_tys {
                let inner = generate_null_constant(member_ty, type_arena, constant_arena)?;
                components.push(constant_arena.fetch_or_append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner,
                }));
            }
            crate::ConstantInner::Composite { ty, components }
        }
        crate::TypeInner::Array {
            base,
            size: crate::ArraySize::Constant(handle),
            ..
        } => {
            let size = constant_arena[handle]
                .to_array_length()
                .ok_or(Error::InvalidArraySize(handle))?;
            let inner = generate_null_constant(base, type_arena, constant_arena)?;
            let value = constant_arena.fetch_or_append(crate::Constant {
                name: None,
                specialization: None,
                inner,
            });
            crate::ConstantInner::Composite {
                ty,
                components: vec![value; size as usize],
            }
        }
        ref other => {
            log::warn!("null constant type {:?}", other);
            return Err(Error::UnsupportedType(ty));
        }
    };
    Ok(inner)
}

/// Create a default value for an output built-in.
pub fn generate_default_built_in(
    built_in: Option<crate::BuiltIn>,
    ty: Handle<crate::Type>,
    type_arena: &mut Arena<crate::Type>,
    constant_arena: &mut Arena<crate::Constant>,
) -> Result<Handle<crate::Constant>, Error> {
    let inner = match built_in {
        Some(crate::BuiltIn::Position) => {
            let zero = constant_arena.fetch_or_append(crate::Constant {
                name: None,
                specialization: None,
                inner: crate::ConstantInner::Scalar {
                    value: crate::ScalarValue::Float(0.0),
                    width: 4,
                },
            });
            let one = constant_arena.fetch_or_append(crate::Constant {
                name: None,
                specialization: None,
                inner: crate::ConstantInner::Scalar {
                    value: crate::ScalarValue::Float(1.0),
                    width: 4,
                },
            });
            crate::ConstantInner::Composite {
                ty,
                components: vec![zero, zero, zero, one],
            }
        }
        Some(crate::BuiltIn::PointSize) => crate::ConstantInner::Scalar {
            value: crate::ScalarValue::Float(1.0),
            width: 4,
        },
        Some(crate::BuiltIn::FragDepth) => crate::ConstantInner::Scalar {
            value: crate::ScalarValue::Float(0.0),
            width: 4,
        },
        Some(crate::BuiltIn::SampleMask) => crate::ConstantInner::Scalar {
            value: crate::ScalarValue::Uint(!0),
            width: 4,
        },
        //Note: `crate::BuiltIn::ClipDistance` is intentionally left for the default path
        _ => generate_null_constant(ty, type_arena, constant_arena)?,
    };
    Ok(constant_arena.fetch_or_append(crate::Constant {
        name: None,
        specialization: None,
        inner,
    }))
}
