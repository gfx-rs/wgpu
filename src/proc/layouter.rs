use crate::arena::Arena;
use std::num::NonZeroU32;

pub type Alignment = NonZeroU32;

/// Alignment information for a type.
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct TypeLayout {
    pub size: u32,
    pub alignment: Alignment,
}

impl TypeLayout {
    /// Return padding to this type given an offset.
    pub fn pad(&self, offset: u32) -> u32 {
        match offset & self.alignment.get() {
            0 => 0,
            other => self.alignment.get() - other,
        }
    }
}

/// Helper processor that derives the sizes of all types.
#[derive(Debug, Default)]
pub struct Layouter {
    layouts: Vec<TypeLayout>,
}

impl Layouter {
    pub fn new(types: &Arena<crate::Type>, constants: &Arena<crate::Constant>) -> Self {
        let mut this = Layouter::default();
        this.initialize(types, constants);
        this
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
                    //TODO: reconsider if this needs to match the size
                    alignment: Alignment::new(width as u32).unwrap(),
                },
                Ti::Matrix {
                    columns,
                    rows,
                    width,
                } => TypeLayout {
                    size: (columns as u8 * rows as u8 * width) as u32,
                    alignment: Alignment::new((columns as u8 * width) as u32).unwrap(),
                },
                Ti::Pointer { .. } => TypeLayout {
                    size: 4,
                    alignment: Alignment::new(1).unwrap(),
                },
                Ti::Array { base, size, stride } => {
                    let count = match size {
                        crate::ArraySize::Constant(handle) => match constants[handle].inner {
                            crate::ConstantInner::Scalar {
                                width: _,
                                value: crate::ScalarValue::Uint(value),
                            } => value as u32,
                            ref other => unreachable!("Unexpected array size {:?}", other),
                        },
                        crate::ArraySize::Dynamic => 1,
                    };
                    let stride = match stride {
                        Some(value) => value,
                        None => Alignment::new(self.layouts[base.index()].size.max(1)).unwrap(),
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
                        let member_layout = self.layouts[member.ty.index()];
                        biggest_alignment = biggest_alignment.max(member_layout.alignment);
                        // align up first
                        total += member_layout.pad(total);
                        // then add the size
                        total += match member.span {
                            Some(span) => span.get(),
                            None => member_layout.size,
                        };
                    }
                    TypeLayout {
                        size: total,
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

    pub fn resolve(&self, handle: crate::Handle<crate::Type>) -> TypeLayout {
        self.layouts[handle.index()]
    }
}
