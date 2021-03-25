use crate::{
    arena::{Arena, Handle},
    proc::Layouter,
};

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct TypeFlags: u8 {
        /// Can be used for data variables.
        const DATA = 0x1;
        /// The data type has known size.
        const SIZED = 0x2;
        /// Can be be used for interfacing between pipeline stages.
        const INTERFACE = 0x4;
        /// Can be used for host-shareable structures.
        const HOST_SHARED = 0x8;
        /// This is a top-level host-shareable type.
        const BLOCK = 0x10;
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum Disalignment {
    #[error("The array stride {stride} is not a multiple of the required alignment {alignment}")]
    ArrayStride { stride: u32, alignment: u32 },
    #[error("The struct size {size}, is not a multiple of the required alignment {alignment}")]
    StructSize { size: u32, alignment: u32 },
    #[error("The struct member[{index}] offset {offset} is not a multiple of the required alignment {alignment}")]
    Member {
        index: u32,
        offset: u32,
        alignment: u32,
    },
    #[error("The struct member[{index}] is not statically sized")]
    UnsizedMember { index: u32 },
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum TypeError {
    #[error("The {0:?} scalar width {1} is not supported")]
    InvalidWidth(crate::ScalarKind, crate::Bytes),
    #[error("The base handle {0:?} can not be resolved")]
    UnresolvedBase(Handle<crate::Type>),
    #[error("Expected data type, found {0:?}")]
    InvalidData(Handle<crate::Type>),
    #[error("Structure type {0:?} can not be a block structure")]
    InvalidBlockType(Handle<crate::Type>),
    #[error("Base type {0:?} for the array is invalid")]
    InvalidArrayBaseType(Handle<crate::Type>),
    #[error("The constant {0:?} can not be used for an array size")]
    InvalidArraySizeConstant(Handle<crate::Constant>),
    #[error(
        "Array stride {stride} is not a multiple of the base element alignment {base_alignment}"
    )]
    UnalignedArrayStride { stride: u32, base_alignment: u32 },
    #[error("Array stride {stride} is smaller than the base element size {base_size}")]
    InsufficientArrayStride { stride: u32, base_size: u32 },
    #[error("Field '{0}' can't be dynamically-sized, has type {1:?}")]
    InvalidDynamicArray(String, Handle<crate::Type>),
    #[error("Structure member[{index}] size {size} is not a sufficient to hold {base_size}")]
    InsufficientMemberSize {
        index: u32,
        size: u32,
        base_size: u32,
    },
    #[error("The composite type contains a block structure")]
    NestedBlock,
}

// Only makes sense if `flags.contains(HOST_SHARED)`
type LayoutCompatibility = Result<(), (Handle<crate::Type>, Disalignment)>;

// For the uniform buffer alignment, array strides and struct sizes must be multiples of 16.
const UNIFORM_LAYOUT_ALIGNMENT_MASK: u32 = 0xF;

#[derive(Clone, Debug)]
pub(super) struct TypeInfo {
    pub flags: TypeFlags,
    pub uniform_layout: LayoutCompatibility,
    pub storage_layout: LayoutCompatibility,
}

impl TypeInfo {
    fn new() -> Self {
        TypeInfo {
            flags: TypeFlags::empty(),
            uniform_layout: Ok(()),
            storage_layout: Ok(()),
        }
    }

    fn from_flags(flags: TypeFlags) -> Self {
        TypeInfo {
            flags,
            uniform_layout: Ok(()),
            storage_layout: Ok(()),
        }
    }
}

impl super::Validator {
    pub(super) fn check_width(kind: crate::ScalarKind, width: crate::Bytes) -> bool {
        match kind {
            crate::ScalarKind::Bool => width == crate::BOOL_WIDTH,
            _ => width == 4,
        }
    }

    pub(super) fn reset_types(&mut self, size: usize) {
        self.types.clear();
        self.types.resize(size, TypeInfo::new());
    }

    pub(super) fn validate_type(
        &self,
        ty: &crate::Type,
        handle: Handle<crate::Type>,
        constants: &Arena<crate::Constant>,
        layouter: &Layouter,
    ) -> Result<TypeInfo, TypeError> {
        use crate::TypeInner as Ti;
        Ok(match ty.inner {
            Ti::Scalar { kind, width } | Ti::Vector { kind, width, .. } => {
                if !Self::check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                TypeInfo::from_flags(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED,
                )
            }
            Ti::Matrix { width, .. } => {
                if !Self::check_width(crate::ScalarKind::Float, width) {
                    return Err(TypeError::InvalidWidth(crate::ScalarKind::Float, width));
                }
                TypeInfo::from_flags(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED,
                )
            }
            Ti::Pointer { base, class: _ } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                TypeInfo::from_flags(TypeFlags::DATA | TypeFlags::SIZED)
            }
            Ti::ValuePointer {
                size: _,
                kind,
                width,
                class: _,
            } => {
                if !Self::check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                TypeInfo::from_flags(TypeFlags::SIZED)
            }
            Ti::Array { base, size, stride } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                let base_info = &self.types[base.index()];
                if !base_info.flags.contains(TypeFlags::DATA | TypeFlags::SIZED) {
                    return Err(TypeError::InvalidArrayBaseType(base));
                }
                if base_info.flags.contains(TypeFlags::BLOCK) {
                    return Err(TypeError::NestedBlock);
                }

                let base_layout = &layouter[base];
                if let Some(stride) = stride {
                    if stride.get() % base_layout.alignment.get() != 0 {
                        return Err(TypeError::UnalignedArrayStride {
                            stride: stride.get(),
                            base_alignment: base_layout.alignment.get(),
                        });
                    }
                    if stride.get() < base_layout.size {
                        return Err(TypeError::InsufficientArrayStride {
                            stride: stride.get(),
                            base_size: base_layout.size,
                        });
                    }
                }

                let (sized_flag, uniform_layout) = match size {
                    crate::ArraySize::Constant(const_handle) => {
                        match constants.try_get(const_handle) {
                            Some(&crate::Constant {
                                inner:
                                    crate::ConstantInner::Scalar {
                                        width: _,
                                        value: crate::ScalarValue::Uint(_),
                                    },
                                ..
                            }) => {}
                            // Accept a signed integer size to avoid
                            // requiring an explicit uint
                            // literal. Type inference should make
                            // this unnecessary.
                            Some(&crate::Constant {
                                inner:
                                    crate::ConstantInner::Scalar {
                                        width: _,
                                        value: crate::ScalarValue::Sint(_),
                                    },
                                ..
                            }) => {}
                            other => {
                                log::warn!("Array size {:?}", other);
                                return Err(TypeError::InvalidArraySizeConstant(const_handle));
                            }
                        }

                        let effective_stride = match stride {
                            Some(stride) => stride.get(),
                            None => base_layout.size,
                        };
                        let uniform_layout =
                            if effective_stride & UNIFORM_LAYOUT_ALIGNMENT_MASK == 0 {
                                base_info.uniform_layout.clone()
                            } else {
                                Err((
                                    handle,
                                    Disalignment::ArrayStride {
                                        stride: effective_stride,
                                        alignment: UNIFORM_LAYOUT_ALIGNMENT_MASK + 1,
                                    },
                                ))
                            };
                        (TypeFlags::SIZED, uniform_layout)
                    }
                    //Note: this will be detected at the struct level
                    crate::ArraySize::Dynamic => (TypeFlags::empty(), Ok(())),
                };

                let base_mask = TypeFlags::HOST_SHARED | TypeFlags::INTERFACE;
                TypeInfo {
                    flags: TypeFlags::DATA | (base_info.flags & base_mask) | sized_flag,
                    uniform_layout,
                    storage_layout: base_info.storage_layout.clone(),
                }
            }
            Ti::Struct { block, ref members } => {
                let mut flags = TypeFlags::DATA
                    | TypeFlags::SIZED
                    | TypeFlags::HOST_SHARED
                    | TypeFlags::INTERFACE;
                let mut uniform_layout = Ok(());
                let mut storage_layout = Ok(());
                let mut offset = 0;
                for (i, member) in members.iter().enumerate() {
                    if member.ty >= handle {
                        return Err(TypeError::UnresolvedBase(member.ty));
                    }
                    let base_info = &self.types[member.ty.index()];
                    if !base_info.flags.contains(TypeFlags::DATA) {
                        return Err(TypeError::InvalidData(member.ty));
                    }
                    if block && !base_info.flags.contains(TypeFlags::INTERFACE) {
                        return Err(TypeError::InvalidBlockType(member.ty));
                    }
                    if base_info.flags.contains(TypeFlags::BLOCK) {
                        return Err(TypeError::NestedBlock);
                    }
                    flags &= base_info.flags;

                    let base_layout = &layouter[member.ty];
                    let (range, _alignment) = layouter.member_placement(offset, member);
                    if range.end - range.start < base_layout.size {
                        return Err(TypeError::InsufficientMemberSize {
                            index: i as u32,
                            size: range.end - range.start,
                            base_size: base_layout.size,
                        });
                    }
                    if range.start % base_layout.alignment.get() != 0 {
                        let result = Err((
                            handle,
                            Disalignment::Member {
                                index: i as u32,
                                offset: range.start,
                                alignment: base_layout.alignment.get(),
                            },
                        ));
                        uniform_layout = uniform_layout.or_else(|_| result.clone());
                        storage_layout = storage_layout.or(result);
                    }
                    offset = range.end;

                    // only the last field can be unsized
                    if !base_info.flags.contains(TypeFlags::SIZED) {
                        if i + 1 != members.len() {
                            let name = member.name.clone().unwrap_or_default();
                            return Err(TypeError::InvalidDynamicArray(name, member.ty));
                        }
                        if uniform_layout.is_ok() {
                            uniform_layout =
                                Err((handle, Disalignment::UnsizedMember { index: i as u32 }));
                        }
                    }

                    uniform_layout = uniform_layout.or_else(|_| base_info.uniform_layout.clone());
                    storage_layout = storage_layout.or_else(|_| base_info.storage_layout.clone());
                }
                if block {
                    flags |= TypeFlags::BLOCK;
                }

                // disabled temporarily, see https://github.com/gpuweb/gpuweb/issues/1558
                const CHECK_STRUCT_SIZE: bool = false;
                if CHECK_STRUCT_SIZE
                    && uniform_layout.is_ok()
                    && offset & UNIFORM_LAYOUT_ALIGNMENT_MASK != 0
                {
                    uniform_layout = Err((
                        handle,
                        Disalignment::StructSize {
                            size: offset,
                            alignment: UNIFORM_LAYOUT_ALIGNMENT_MASK + 1,
                        },
                    ));
                }
                TypeInfo {
                    flags,
                    uniform_layout,
                    storage_layout,
                }
            }
            Ti::Image { .. } | Ti::Sampler { .. } => TypeInfo::from_flags(TypeFlags::empty()),
        })
    }
}
