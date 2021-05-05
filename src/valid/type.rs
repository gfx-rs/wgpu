use super::Capabilities;
use crate::arena::{Arena, Handle};

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
    #[error("The struct span {span}, is not a multiple of the required alignment {alignment}")]
    StructSpan { span: u32, alignment: u32 },
    #[error("The struct span {alignment}, is not a multiple of the member[{member_index}] alignment {member_alignment}")]
    StructAlignment {
        alignment: u32,
        member_index: u32,
        member_alignment: u32,
    },
    #[error("The struct member[{index}] offset {offset} is not a multiple of the required alignment {alignment}")]
    MemberOffset {
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
    #[error("Structure member[{index}] at {offset} overlaps the previous member")]
    MemberOverlap { index: u32, offset: u32 },
    #[error(
        "Structure member[{index}] at {offset} and size {size} crosses the structure boundary"
    )]
    MemberOutOfBounds { index: u32, offset: u32, size: u32 },
    #[error("The composite type contains a block structure")]
    NestedBlock,
}

// Only makes sense if `flags.contains(HOST_SHARED)`
type LayoutCompatibility = Result<Option<crate::Alignment>, (Handle<crate::Type>, Disalignment)>;

fn check_member_layout(
    accum: &mut LayoutCompatibility,
    member: &crate::StructMember,
    member_index: u32,
    member_layout: LayoutCompatibility,
    struct_level: crate::StructLevel,
    ty_handle: Handle<crate::Type>,
) {
    *accum = match (*accum, member_layout) {
        (Ok(cur_alignment), Ok(align)) => {
            let align = align.unwrap().get();
            if member.offset % align != 0 {
                Err((
                    ty_handle,
                    Disalignment::MemberOffset {
                        index: member_index,
                        offset: member.offset,
                        alignment: align,
                    },
                ))
            } else {
                match struct_level {
                    crate::StructLevel::Normal { alignment } if alignment.get() % align != 0 => {
                        Err((
                            ty_handle,
                            Disalignment::StructAlignment {
                                alignment: alignment.get(),
                                member_index,
                                member_alignment: align,
                            },
                        ))
                    }
                    _ => {
                        let combined_alignment =
                            ((cur_alignment.unwrap().get() - 1) | (align - 1)) + 1;
                        Ok(crate::Alignment::new(combined_alignment))
                    }
                }
            }
        }
        (Err(e), _) | (_, Err(e)) => Err(e),
    };
}

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
            uniform_layout: Ok(None),
            storage_layout: Ok(None),
        }
    }

    fn new(flags: TypeFlags, align: u32) -> Self {
        let alignment = crate::Alignment::new(align);
        TypeInfo {
            flags,
            uniform_layout: Ok(alignment),
            storage_layout: Ok(alignment),
        }
    }
}

impl super::Validator {
    pub(super) fn check_width(&self, kind: crate::ScalarKind, width: crate::Bytes) -> bool {
        match kind {
            crate::ScalarKind::Bool => width == crate::BOOL_WIDTH,
            crate::ScalarKind::Float => {
                width == 4 || (width == 8 && self.capabilities.contains(Capabilities::FLOAT64))
            }
            crate::ScalarKind::Sint | crate::ScalarKind::Uint => width == 4,
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
                if !self.check_width(kind, width) {
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
                if !self.check_width(kind, width) {
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
                if !self.check_width(crate::ScalarKind::Float, width) {
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
                if !self.check_width(kind, width) {
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
                        let align = ((base_alignment.unwrap().get() - 1)
                            | UNIFORM_LAYOUT_ALIGNMENT_MASK)
                            + 1;
                        if stride % align != 0 {
                            Err((
                                handle,
                                Disalignment::ArrayStride {
                                    stride,
                                    alignment: align,
                                },
                            ))
                        } else {
                            Ok(crate::Alignment::new(align))
                        }
                    }
                    Err(e) => Err(e),
                };
                let storage_layout = match base_info.storage_layout {
                    Ok(base_alignment) => {
                        let align = base_alignment.unwrap().get();
                        if stride % align != 0 {
                            Err((
                                handle,
                                Disalignment::ArrayStride {
                                    stride,
                                    alignment: align,
                                },
                            ))
                        } else {
                            Ok(base_alignment)
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
            Ti::Struct {
                level,
                ref members,
                span,
            } => {
                let mut ti = TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::HOST_SHARED
                        | TypeFlags::INTERFACE,
                    1,
                );
                let mut min_offset = 0;
                for (i, member) in members.iter().enumerate() {
                    if member.ty >= handle {
                        return Err(TypeError::UnresolvedBase(member.ty));
                    }
                    let base_info = &self.types[member.ty.index()];
                    if !base_info.flags.contains(TypeFlags::DATA) {
                        return Err(TypeError::InvalidData(member.ty));
                    }
                    if level == crate::StructLevel::Root
                        && !base_info.flags.contains(TypeFlags::INTERFACE)
                    {
                        return Err(TypeError::InvalidBlockType(member.ty));
                    }
                    if base_info.flags.contains(TypeFlags::BLOCK) {
                        return Err(TypeError::NestedBlock);
                    }
                    ti.flags &= base_info.flags;

                    if member.offset < min_offset {
                        //HACK: this could be nicer. We want to allow some structures
                        // to not bother with offsets/alignments if they are never
                        // used for host sharing.
                        if member.offset == 0 {
                            ti.flags.set(TypeFlags::HOST_SHARED, false);
                        } else {
                            return Err(TypeError::MemberOverlap {
                                index: i as u32,
                                offset: member.offset,
                            });
                        }
                    }
                    let base_size = types[member.ty].inner.span(constants);
                    min_offset = member.offset + base_size;
                    if min_offset > span {
                        return Err(TypeError::MemberOutOfBounds {
                            index: i as u32,
                            offset: member.offset,
                            size: base_size,
                        });
                    }

                    check_member_layout(
                        &mut ti.uniform_layout,
                        member,
                        i as u32,
                        base_info.uniform_layout,
                        level,
                        handle,
                    );
                    check_member_layout(
                        &mut ti.storage_layout,
                        member,
                        i as u32,
                        base_info.storage_layout,
                        level,
                        handle,
                    );

                    // only the last field can be unsized
                    if !base_info.flags.contains(TypeFlags::SIZED) {
                        if i + 1 != members.len() {
                            let name = member.name.clone().unwrap_or_default();
                            return Err(TypeError::InvalidDynamicArray(name, member.ty));
                        }
                        if ti.uniform_layout.is_ok() {
                            ti.uniform_layout =
                                Err((handle, Disalignment::UnsizedMember { index: i as u32 }));
                        }
                    }
                }
                if let crate::StructLevel::Root = level {
                    ti.flags |= TypeFlags::BLOCK;
                }

                // disabled temporarily, see https://github.com/gpuweb/gpuweb/issues/1558
                const CHECK_STRUCT_SIZE: bool = false;
                if CHECK_STRUCT_SIZE
                    && ti.uniform_layout.is_ok()
                    && span & UNIFORM_LAYOUT_ALIGNMENT_MASK != 0
                {
                    ti.uniform_layout = Err((
                        handle,
                        Disalignment::StructSpan {
                            span,
                            alignment: UNIFORM_LAYOUT_ALIGNMENT_MASK + 1,
                        },
                    ));
                }
                ti
            }
            Ti::Image { .. } | Ti::Sampler { .. } => TypeInfo::new(TypeFlags::empty(), 0),
        })
    }
}
