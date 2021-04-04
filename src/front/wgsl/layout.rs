use crate::arena::{Arena, Handle};
use std::num::NonZeroU32;

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

pub struct Placement {
    pub pad: crate::Span,
    pub span: crate::Span,
}

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
    ) -> Placement {
        let layout = self.layouts[ty.index()];
        let alignment = align.unwrap_or(layout.alignment);
        let start = Self::round_up(alignment, offset);
        let span = match size {
            Some(size) => size.get(),
            None => layout.size,
        };
        Placement {
            pad: start - offset,
            span,
        }
    }

    pub fn update(&mut self, types: &Arena<crate::Type>, constants: &Arena<crate::Constant>) {
        use crate::TypeInner as Ti;
        for (_, ty) in types.iter().skip(self.layouts.len()) {
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
                Ti::Array { stride, .. } => TypeLayout {
                    size,
                    alignment: Alignment::new(stride).unwrap(),
                },
                Ti::Struct {
                    block: _,
                    ref members,
                } => {
                    let mut total = 0;
                    let mut biggest_alignment = Alignment::new(1).unwrap();
                    for member in members {
                        let layout = self.layouts[member.ty.index()];
                        biggest_alignment = biggest_alignment.max(layout.alignment);
                        total += member.span;
                    }
                    TypeLayout {
                        size: Self::round_up(biggest_alignment, total),
                        alignment: biggest_alignment,
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
    }
}
