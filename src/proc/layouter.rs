use crate::arena::{Arena, Handle};
use std::{num::NonZeroU32, ops};

pub type Alignment = NonZeroU32;

/// Alignment information for a type.
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct TypeLayout {
    pub size: u32,
    pub alignment: Alignment,
}

/// Helper processor that derives the sizes of all types.
/// It uses the default layout algorithm/table, described in
/// https://github.com/gpuweb/gpuweb/issues/1393
#[derive(Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct Layouter {
    layouts: Vec<TypeLayout>,
}

impl ops::Index<Handle<crate::Type>> for Layouter {
    type Output = TypeLayout;
    fn index(&self, handle: Handle<crate::Type>) -> &TypeLayout {
        &self.layouts[handle.index()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("Base type {0:?} is out of bounds")]
pub struct InvalidBaseType(pub Handle<crate::Type>);

impl Layouter {
    pub fn clear(&mut self) {
        self.layouts.clear();
    }

    pub fn round_up(alignment: Alignment, offset: u32) -> u32 {
        match offset & (alignment.get() - 1) {
            0 => offset,
            other => offset + alignment.get() - other,
        }
    }

    pub fn member_placement(
        &self,
        offset: u32,
        ty: Handle<crate::Type>,
        align: Option<Alignment>,
        size: Option<NonZeroU32>,
    ) -> (ops::Range<u32>, Alignment) {
        let layout = self.layouts[ty.index()];
        let alignment = align.unwrap_or(layout.alignment);
        let start = Self::round_up(alignment, offset);
        let span = match size {
            Some(size) => size.get(),
            None => layout.size,
        };
        (start..start + span, alignment)
    }

    pub fn update(
        &mut self,
        types: &Arena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> Result<(), InvalidBaseType> {
        use crate::TypeInner as Ti;

        for (ty_handle, ty) in types.iter().skip(self.layouts.len()) {
            let size = ty.inner.span(constants);
            let layout = match ty.inner {
                Ti::Scalar { width, .. } => TypeLayout {
                    size,
                    alignment: Alignment::new(width as u32).unwrap(),
                },
                Ti::Vector {
                    size: vec_size,
                    width,
                    ..
                } => TypeLayout {
                    size,
                    alignment: {
                        let count = if vec_size >= crate::VectorSize::Tri {
                            4
                        } else {
                            2
                        };
                        Alignment::new((count * width) as u32).unwrap()
                    },
                },
                Ti::Matrix {
                    columns: _,
                    rows,
                    width,
                } => TypeLayout {
                    size,
                    alignment: {
                        let count = if rows >= crate::VectorSize::Tri { 4 } else { 2 };
                        Alignment::new((count * width) as u32).unwrap()
                    },
                },
                Ti::Pointer { .. } | Ti::ValuePointer { .. } => TypeLayout {
                    size,
                    alignment: Alignment::new(1).unwrap(),
                },
                Ti::Array {
                    base,
                    stride: _,
                    size: _,
                } => TypeLayout {
                    size,
                    alignment: if base < ty_handle {
                        self[base].alignment
                    } else {
                        return Err(InvalidBaseType(base));
                    },
                },
                Ti::Struct {
                    top_level: _,
                    span,
                    ref members,
                } => {
                    let mut alignment = Alignment::new(1).unwrap();
                    for member in members {
                        alignment = if member.ty < ty_handle {
                            alignment.max(self[member.ty].alignment)
                        } else {
                            return Err(InvalidBaseType(member.ty));
                        };
                    }
                    TypeLayout {
                        size: span,
                        alignment,
                    }
                }
                Ti::Image { .. } | Ti::Sampler { .. } => TypeLayout {
                    size,
                    alignment: Alignment::new(1).unwrap(),
                },
            };
            debug_assert!(ty.inner.span(constants) <= layout.size);
            self.layouts.push(layout);
        }

        Ok(())
    }
}
