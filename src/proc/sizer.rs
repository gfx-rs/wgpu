use crate::arena::Arena;

/// Helper processor that derives the sizes of all types.
#[derive(Debug)]
pub struct Sizer {
    sizes: Vec<u32>,
}

impl Sizer {
    pub fn new(types: &Arena<crate::Type>, constants: &Arena<crate::Constant>) -> Self {
        use crate::TypeInner as Ti;

        let mut sizes = Vec::with_capacity(types.len());
        for (_, ty) in types.iter() {
            sizes.push(match ty.inner {
                Ti::Scalar { kind: _, width } => width as u32,
                Ti::Vector {
                    size,
                    kind: _,
                    width,
                } => (size as u8 * width) as u32,
                Ti::Matrix {
                    columns,
                    rows,
                    width,
                } => (columns as u8 * rows as u8 * width) as u32,
                Ti::Pointer { .. } => 0,
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
                        Some(value) => value.get(),
                        None => sizes[base.index()],
                    };
                    count * stride
                }
                Ti::Struct {
                    block: _,
                    ref members,
                } => {
                    let mut total = 0;
                    for member in members {
                        total += match member.span {
                            Some(span) => span.get(),
                            None => sizes[member.ty.index()],
                        };
                    }
                    total
                }
                Ti::Image { .. } | Ti::Sampler { .. } => 0,
            });
        }

        Sizer { sizes }
    }

    pub fn resolve(&self, handle: crate::Handle<crate::Type>) -> u32 {
        self.sizes[handle.index()]
    }
}
