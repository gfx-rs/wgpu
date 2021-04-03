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

impl Layouter {
    pub fn new(types: &Arena<crate::Type>, constants: &Arena<crate::Constant>) -> Self {
        let mut this = Layouter::default();
        this.initialize(types, constants);
        this
    }

    pub fn round_up(alignment: NonZeroU32, offset: u32) -> u32 {
        match offset & (alignment.get() - 1) {
            0 => offset,
            other => offset + alignment.get() - other,
        }
    }

    pub fn member_placement(
        &self,
        offset: u32,
        member: &crate::StructMember,
    ) -> (ops::Range<u32>, NonZeroU32) {
        let layout = self.layouts[member.ty.index()];
        let alignment = member.align.unwrap_or(layout.alignment);
        let start = Self::round_up(alignment, offset);
        let end = start
            + match member.size {
                Some(size) => size.get(),
                None => layout.size,
            };
        (start..end, alignment)
    }

    pub fn initialize(&mut self, types: &Arena<crate::Type>, constants: &Arena<crate::Constant>) {
        use crate::TypeInner as Ti;

        self.layouts.clear();
        self.layouts.reserve(types.len());

        for (_, ty) in types.iter() {
            self.layouts.push(match ty.inner {
                Ti::Scalar { kind: _, width } => TypeLayout {
                    size: width as u32,
                    alignment: Alignment::new(width as u32).unwrap(),
                },
                Ti::Vector {
                    size,
                    kind: _,
                    width,
                } => TypeLayout {
                    size: (size as u8 * width) as u32,
                    alignment: {
                        let count = if size >= crate::VectorSize::Tri { 4 } else { 2 };
                        Alignment::new((count * width) as u32).unwrap()
                    },
                },
                Ti::Matrix {
                    columns,
                    rows,
                    width,
                } => TypeLayout {
                    size: (columns as u8 * rows as u8 * width) as u32,
                    alignment: {
                        let count = if rows >= crate::VectorSize::Tri { 4 } else { 2 };
                        Alignment::new((count * width) as u32).unwrap()
                    },
                },
                Ti::Pointer { .. } | Ti::ValuePointer { .. } => TypeLayout {
                    size: 4,
                    alignment: Alignment::new(1).unwrap(),
                },
                Ti::Array { base, size, stride } => {
                    let count = match size {
                        crate::ArraySize::Constant(handle) => {
                            constants[handle].to_array_length().unwrap()
                        }
                        // A dynamically-sized array has to have at least one element
                        crate::ArraySize::Dynamic => 1,
                    };
                    let stride = match stride {
                        Some(value) => value,
                        None => {
                            let layout = &self.layouts[base.index()];
                            let stride = Self::round_up(layout.alignment, layout.size);
                            Alignment::new(stride).unwrap()
                        }
                    };
                    TypeLayout {
                        size: count * stride.get(),
                        alignment: stride,
                    }
                }
                Ti::Struct {
                    block: _,
                    ref members,
                } => {
                    let mut total = 0;
                    let mut biggest_alignment = Alignment::new(1).unwrap();
                    for member in members {
                        let (placement, alignment) = self.member_placement(total, member);
                        biggest_alignment = biggest_alignment.max(alignment);
                        total = placement.end;
                    }
                    TypeLayout {
                        size: Self::round_up(biggest_alignment, total),
                        alignment: biggest_alignment,
                    }
                }
                Ti::Image { .. } | Ti::Sampler { .. } => TypeLayout {
                    size: 0,
                    alignment: Alignment::new(1).unwrap(),
                },
            });
        }
    }
}

impl ops::Index<Handle<crate::Type>> for Layouter {
    type Output = TypeLayout;
    fn index(&self, handle: Handle<crate::Type>) -> &TypeLayout {
        &self.layouts[handle.index()]
    }
}
