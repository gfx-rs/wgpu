use crate::arena::{Arena, Handle};

pub type Alignment = u32;

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

#[derive(Clone, Copy, Debug, thiserror::Error)]
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
type LayoutCompatibility = Result<Alignment, (Handle<crate::Type>, Disalignment)>;

// For the uniform buffer alignment, array strides and struct sizes must be multiples of 16.
const UNIFORM_LAYOUT_ALIGNMENT_MASK: u32 = 0xF;

#[derive(Clone, Debug)]
pub(super) struct TypeInfo {
    pub flags: TypeFlags,
    pub uniform_layout: LayoutCompatibility,
    pub storage_layout: LayoutCompatibility,
}

impl TypeInfo {
    fn dummy() -> Self {
        TypeInfo {
            flags: TypeFlags::empty(),
            uniform_layout: Ok(0),
            storage_layout: Ok(0),
        }
    }

    fn new(flags: TypeFlags, alignment: crate::Span) -> Self {
        TypeInfo {
            flags,
            uniform_layout: Ok(alignment),
            storage_layout: Ok(alignment),
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
        self.types.resize(size, TypeInfo::dummy());
    }

    pub(super) fn validate_type(
        &self,
        handle: Handle<crate::Type>,
        types: &Arena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> Result<TypeInfo, TypeError> {
        use crate::TypeInner as Ti;
        Ok(match types[handle].inner {
            Ti::Scalar { kind, width } => {
                if !Self::check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED,
                    width as u32,
                )
            }
            Ti::Vector { size, kind, width } => {
                if !Self::check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                let count = if size >= crate::VectorSize::Tri { 4 } else { 2 };
                TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED,
                    count * (width as u32),
                )
            }
            Ti::Matrix {
                columns: _,
                rows,
                width,
            } => {
                if !Self::check_width(crate::ScalarKind::Float, width) {
                    return Err(TypeError::InvalidWidth(crate::ScalarKind::Float, width));
                }
                let count = if rows >= crate::VectorSize::Tri { 4 } else { 2 };
                TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED,
                    count * (width as u32),
                )
            }
            Ti::Pointer { base, class: _ } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                TypeInfo::new(TypeFlags::DATA | TypeFlags::SIZED, 0)
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
                TypeInfo::new(TypeFlags::SIZED, 0)
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

                let base_size = types[base].inner.span(constants);
                if stride < base_size {
                    return Err(TypeError::InsufficientArrayStride { stride, base_size });
                }

                let uniform_layout = match base_info.uniform_layout {
                    Ok(base_alignment) => {
                        // combine the alignment requirements
                        let alignment = ((base_alignment - 1) | UNIFORM_LAYOUT_ALIGNMENT_MASK) + 1;
                        if stride % alignment != 0 {
                            Err((handle, Disalignment::ArrayStride { stride, alignment }))
                        } else {
                            Ok(alignment)
                        }
                    }
                    Err(e) => Err(e),
                };
                let storage_layout = match base_info.storage_layout {
                    Ok(alignment) => {
                        if stride % alignment != 0 {
                            Err((handle, Disalignment::ArrayStride { stride, alignment }))
                        } else {
                            Ok(alignment)
                        }
                    }
                    Err(e) => Err(e),
                };

                let sized_flag = match size {
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

                        TypeFlags::SIZED
                    }
                    //Note: this will be detected at the struct level
                    crate::ArraySize::Dynamic => TypeFlags::empty(),
                };

                let base_mask = TypeFlags::HOST_SHARED | TypeFlags::INTERFACE;
                TypeInfo {
                    flags: TypeFlags::DATA | (base_info.flags & base_mask) | sized_flag,
                    uniform_layout,
                    storage_layout,
                }
            }
            Ti::Struct { block, ref members } => {
                let mut flags = TypeFlags::DATA
                    | TypeFlags::SIZED
                    | TypeFlags::HOST_SHARED
                    | TypeFlags::INTERFACE;
                let mut uniform_layout = Ok(1);
                let mut storage_layout = Ok(1);
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

                    let base_size = types[member.ty].inner.span(constants);
                    if member.span < base_size {
                        return Err(TypeError::InsufficientMemberSize {
                            index: i as u32,
                            size: member.span,
                            base_size,
                        });
                    }

                    uniform_layout = match (uniform_layout, base_info.uniform_layout) {
                        (Ok(cur_alignment), Ok(alignment)) => {
                            if offset % alignment != 0 {
                                Err((
                                    handle,
                                    Disalignment::Member {
                                        index: i as u32,
                                        offset,
                                        alignment,
                                    },
                                ))
                            } else {
                                let combined_alignment =
                                    ((cur_alignment - 1) | (alignment - 1)) + 1;
                                Ok(combined_alignment)
                            }
                        }
                        (Err(e), _) | (_, Err(e)) => Err(e),
                    };
                    storage_layout = match (storage_layout, base_info.storage_layout) {
                        (Ok(cur_alignment), Ok(alignment)) => {
                            if offset % alignment != 0 {
                                Err((
                                    handle,
                                    Disalignment::Member {
                                        index: i as u32,
                                        offset,
                                        alignment,
                                    },
                                ))
                            } else {
                                let combined_alignment =
                                    ((cur_alignment - 1) | (alignment - 1)) + 1;
                                Ok(combined_alignment)
                            }
                        }
                        (Err(e), _) | (_, Err(e)) => Err(e),
                    };
                    offset += member.span;

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
            Ti::Image { .. } | Ti::Sampler { .. } => TypeInfo::new(TypeFlags::empty(), 0),
        })
    }
}
