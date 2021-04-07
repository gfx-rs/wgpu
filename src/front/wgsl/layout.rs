use crate::arena::{Arena, Handle};
use std::{num::NonZeroU32, ops};

/// Alignment information for a type.
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct TypeLayout {
    pub size: u32,
    pub alignment: crate::Alignment,
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

impl Layouter {
    pub fn clear(&mut self) {
        self.layouts.clear();
    }

    pub fn round_up(alignment: crate::Alignment, offset: u32) -> u32 {
        match offset & (alignment.get() - 1) {
            0 => offset,
            other => offset + alignment.get() - other,
        }
    }

    pub fn member_placement(
        &self,
        offset: u32,
        ty: Handle<crate::Type>,
        align: Option<crate::Alignment>,
        size: Option<NonZeroU32>,
    ) -> (ops::Range<u32>, crate::Alignment) {
        let layout = self.layouts[ty.index()];
        let alignment = align.unwrap_or(layout.alignment);
        let start = Self::round_up(alignment, offset);
        let span = match size {
            Some(size) => size.get(),
            None => layout.size,
        };
        (start..start + span, alignment)
    }

    pub fn update(&mut self, types: &Arena<crate::Type>, constants: &Arena<crate::Constant>) {
        use crate::TypeInner as Ti;
        for (_, ty) in types.iter().skip(self.layouts.len()) {
            let size = ty.inner.span(constants);
            let layout = match ty.inner {
                Ti::Scalar { width, .. } => TypeLayout {
                    size,
                    alignment: crate::Alignment::new(width as u32).unwrap(),
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
                        crate::Alignment::new((count * width) as u32).unwrap()
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
                        crate::Alignment::new((count * width) as u32).unwrap()
                    },
                },
                Ti::Pointer { .. } | Ti::ValuePointer { .. } => TypeLayout {
                    size,
                    alignment: crate::Alignment::new(1).unwrap(),
                },
                Ti::Array { stride, .. } => TypeLayout {
                    size,
                    alignment: crate::Alignment::new(stride).unwrap(),
                },
                Ti::Struct {
                    ref level,
                    members: _,
                    span,
                } => TypeLayout {
                    size: span,
                    alignment: match *level {
                        crate::StructLevel::Root => crate::Alignment::new(1).unwrap(),
                        crate::StructLevel::Normal { alignment } => alignment,
                    },
                },
                Ti::Image { .. } | Ti::Sampler { .. } => TypeLayout {
                    size,
                    alignment: crate::Alignment::new(1).unwrap(),
                },
            };
            debug_assert!(ty.inner.span(constants) <= layout.size);
            self.layouts.push(layout);
        }
    }
}
