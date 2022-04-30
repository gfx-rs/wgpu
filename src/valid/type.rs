use super::Capabilities;
use crate::{
    arena::{Arena, BadHandle, Handle, UniqueArena},
    proc::Alignment,
};

const UNIFORM_MIN_ALIGNMENT: Alignment = unsafe { Alignment::new_unchecked(16) };

bitflags::bitflags! {
    /// Flags associated with [`Type`]s by [`Validator`].
    ///
    /// [`Type`]: crate::Type
    /// [`Validator`]: crate::valid::Validator
    #[repr(transparent)]
    pub struct TypeFlags: u8 {
        /// Can be used for data variables.
        ///
        /// This flag is required on types of local variables, function
        /// arguments, array elements, and struct members.
        ///
        /// This includes all types except `Image`, `Sampler`,
        /// and some `Pointer` types.
        const DATA = 0x1;

        /// The data type has a size known by pipeline creation time.
        ///
        /// Unsized types are quite restricted. The only unsized types permitted
        /// by Naga, other than the non-[`DATA`] types like [`Image`] and
        /// [`Sampler`], are dynamically-sized [`Array`s], and [`Struct`s] whose
        /// last members are such arrays. See the documentation for those types
        /// for details.
        ///
        /// [`DATA`]: TypeFlags::DATA
        /// [`Image`]: crate::Type::Image
        /// [`Sampler`]: crate::Type::Sampler
        /// [`Array`]: crate::Type::Array
        /// [`Struct`]: crate::Type::struct
        const SIZED = 0x2;

        /// The data can be copied around.
        const COPY = 0x4;

        /// Can be be used for user-defined IO between pipeline stages.
        ///
        /// This covers anything that can be in [`Location`] binding:
        /// non-bool scalars and vectors, matrices, and structs and
        /// arrays containing only interface types.
        const IO_SHAREABLE = 0x8;

        /// Can be used for host-shareable structures.
        const HOST_SHAREABLE = 0x10;

        /// This type can be passed as a function argument.
        const ARGUMENT = 0x40;
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum Disalignment {
    #[error("The array stride {stride} is not a multiple of the required alignment {alignment}")]
    ArrayStride { stride: u32, alignment: u32 },
    #[error("The struct span {span}, is not a multiple of the required alignment {alignment}")]
    StructSpan { span: u32, alignment: u32 },
    #[error("The struct member[{index}] offset {offset} is not a multiple of the required alignment {alignment}")]
    MemberOffset {
        index: u32,
        offset: u32,
        alignment: u32,
    },
    #[error("The struct member[{index}] offset {offset} must be at least {expected}")]
    MemberOffsetAfterStruct {
        index: u32,
        offset: u32,
        expected: u32,
    },
    #[error("The struct member[{index}] is not statically sized")]
    UnsizedMember { index: u32 },
    #[error("The type is not host-shareable")]
    NonHostShareable,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum TypeError {
    #[error(transparent)]
    BadHandle(#[from] BadHandle),
    #[error("The {0:?} scalar width {1} is not supported")]
    InvalidWidth(crate::ScalarKind, crate::Bytes),
    #[error("The {0:?} scalar width {1} is not supported for an atomic")]
    InvalidAtomicWidth(crate::ScalarKind, crate::Bytes),
    #[error("The base handle {0:?} can not be resolved")]
    UnresolvedBase(Handle<crate::Type>),
    #[error("Invalid type for pointer target {0:?}")]
    InvalidPointerBase(Handle<crate::Type>),
    #[error("Unsized types like {base:?} must be in the `Storage` address space, not `{space:?}`")]
    InvalidPointerToUnsized {
        base: Handle<crate::Type>,
        space: crate::AddressSpace,
    },
    #[error("Expected data type, found {0:?}")]
    InvalidData(Handle<crate::Type>),
    #[error("Base type {0:?} for the array is invalid")]
    InvalidArrayBaseType(Handle<crate::Type>),
    #[error("The constant {0:?} can not be used for an array size")]
    InvalidArraySizeConstant(Handle<crate::Constant>),
    #[error("The constant {0:?} is specialized, and cannot be used as an array size")]
    UnsupportedSpecializedArrayLength(Handle<crate::Constant>),
    #[error("Array type {0:?} must have a length of one or more")]
    NonPositiveArrayLength(Handle<crate::Constant>),
    #[error("Array stride {stride} does not match the expected {expected}")]
    InvalidArrayStride { stride: u32, expected: u32 },
    #[error("Field '{0}' can't be dynamically-sized, has type {1:?}")]
    InvalidDynamicArray(String, Handle<crate::Type>),
    #[error("Structure member[{index}] at {offset} overlaps the previous member")]
    MemberOverlap { index: u32, offset: u32 },
    #[error(
        "Structure member[{index}] at {offset} and size {size} crosses the structure boundary of size {span}"
    )]
    MemberOutOfBounds {
        index: u32,
        offset: u32,
        size: u32,
        span: u32,
    },
    #[error("Structure types must have at least one member")]
    EmptyStruct,
}

// Only makes sense if `flags.contains(HOST_SHARED)`
type LayoutCompatibility = Result<Option<Alignment>, (Handle<crate::Type>, Disalignment)>;

fn check_member_layout(
    accum: &mut LayoutCompatibility,
    member: &crate::StructMember,
    member_index: u32,
    member_layout: LayoutCompatibility,
    parent_handle: Handle<crate::Type>,
) {
    *accum = match (*accum, member_layout) {
        (Ok(cur_alignment), Ok(align)) => {
            let align = align.unwrap().get();
            if member.offset % align != 0 {
                Err((
                    parent_handle,
                    Disalignment::MemberOffset {
                        index: member_index,
                        offset: member.offset,
                        alignment: align,
                    },
                ))
            } else {
                let combined_alignment = ((cur_alignment.unwrap().get() - 1) | (align - 1)) + 1;
                Ok(Alignment::new(combined_alignment))
            }
        }
        (Err(e), _) | (_, Err(e)) => Err(e),
    };
}

/// Determine whether a pointer in `space` can be passed as an argument.
///
/// If a pointer in `space` is permitted to be passed as an argument to a
/// user-defined function, return `TypeFlags::ARGUMENT`. Otherwise, return
/// `TypeFlags::empty()`.
///
/// Pointers passed as arguments to user-defined functions must be in the
/// `Function`, `Private`, or `Workgroup` storage space.
const fn ptr_space_argument_flag(space: crate::AddressSpace) -> TypeFlags {
    use crate::AddressSpace as As;
    match space {
        As::Function | As::Private | As::WorkGroup => TypeFlags::ARGUMENT,
        As::Uniform | As::Storage { .. } | As::Handle | As::PushConstant => TypeFlags::empty(),
    }
}

#[derive(Clone, Debug)]
pub(super) struct TypeInfo {
    pub flags: TypeFlags,
    pub uniform_layout: LayoutCompatibility,
    pub storage_layout: LayoutCompatibility,
}

impl TypeInfo {
    const fn dummy() -> Self {
        TypeInfo {
            flags: TypeFlags::empty(),
            uniform_layout: Ok(None),
            storage_layout: Ok(None),
        }
    }

    const fn new(flags: TypeFlags, align: u32) -> Self {
        let alignment = Alignment::new(align);
        TypeInfo {
            flags,
            uniform_layout: Ok(alignment),
            storage_layout: Ok(alignment),
        }
    }
}

impl super::Validator {
    pub(super) const fn check_width(&self, kind: crate::ScalarKind, width: crate::Bytes) -> bool {
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
        self.layouter.clear();
    }

    pub(super) fn validate_type(
        &self,
        handle: Handle<crate::Type>,
        types: &UniqueArena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> Result<TypeInfo, TypeError> {
        use crate::TypeInner as Ti;
        Ok(match types[handle].inner {
            Ti::Scalar { kind, width } => {
                if !self.check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                let shareable = if kind.is_numeric() {
                    TypeFlags::IO_SHAREABLE | TypeFlags::HOST_SHAREABLE
                } else {
                    TypeFlags::empty()
                };
                TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::COPY
                        | TypeFlags::ARGUMENT
                        | shareable,
                    width as u32,
                )
            }
            Ti::Vector { size, kind, width } => {
                if !self.check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }
                let shareable = if kind.is_numeric() {
                    TypeFlags::IO_SHAREABLE | TypeFlags::HOST_SHAREABLE
                } else {
                    TypeFlags::empty()
                };
                let count = if size >= crate::VectorSize::Tri { 4 } else { 2 };
                TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::COPY
                        | TypeFlags::HOST_SHAREABLE
                        | TypeFlags::ARGUMENT
                        | shareable,
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
                        | TypeFlags::COPY
                        | TypeFlags::HOST_SHAREABLE
                        | TypeFlags::ARGUMENT,
                    count * (width as u32),
                )
            }
            Ti::Atomic { kind, width } => {
                let good = match kind {
                    crate::ScalarKind::Bool | crate::ScalarKind::Float => false,
                    crate::ScalarKind::Sint | crate::ScalarKind::Uint => width == 4,
                };
                if !good {
                    return Err(TypeError::InvalidAtomicWidth(kind, width));
                }
                TypeInfo::new(
                    TypeFlags::DATA | TypeFlags::SIZED | TypeFlags::HOST_SHAREABLE,
                    width as u32,
                )
            }
            Ti::Pointer { base, space } => {
                use crate::AddressSpace as As;

                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }

                let base_info = &self.types[base.index()];
                if !base_info.flags.contains(TypeFlags::DATA) {
                    return Err(TypeError::InvalidPointerBase(base));
                }

                // Runtime-sized values can only live in the `Storage` storage
                // space, so it's useless to have a pointer to such a type in
                // any other space.
                //
                // Detecting this problem here prevents the definition of
                // functions like:
                //
                //     fn f(p: ptr<workgroup, UnsizedType>) -> ... { ... }
                //
                // which would otherwise be permitted, but uncallable. (They
                // may also present difficulties in code generation).
                if !base_info.flags.contains(TypeFlags::SIZED) {
                    match space {
                        As::Storage { .. } => {}
                        _ => {
                            return Err(TypeError::InvalidPointerToUnsized { base, space });
                        }
                    }
                }

                // `Validator::validate_function` actually checks the storage
                // space of pointer arguments explicitly before checking the
                // `ARGUMENT` flag, to give better error messages. But it seems
                // best to set `ARGUMENT` accurately anyway.
                let argument_flag = ptr_space_argument_flag(space);

                // Pointers cannot be stored in variables, structure members, or
                // array elements, so we do not mark them as `DATA`.
                TypeInfo::new(argument_flag | TypeFlags::SIZED | TypeFlags::COPY, 0)
            }
            Ti::ValuePointer {
                size: _,
                kind,
                width,
                space,
            } => {
                // ValuePointer should be treated the same way as the equivalent
                // Pointer / Scalar / Vector combination, so each step in those
                // variants' match arms should have a counterpart here.
                //
                // However, some cases are trivial: All our implicit base types
                // are DATA and SIZED, so we can never return
                // `InvalidPointerBase` or `InvalidPointerToUnsized`.
                if !self.check_width(kind, width) {
                    return Err(TypeError::InvalidWidth(kind, width));
                }

                // `Validator::validate_function` actually checks the storage
                // space of pointer arguments explicitly before checking the
                // `ARGUMENT` flag, to give better error messages. But it seems
                // best to set `ARGUMENT` accurately anyway.
                let argument_flag = ptr_space_argument_flag(space);

                // Pointers cannot be stored in variables, structure members, or
                // array elements, so we do not mark them as `DATA`.
                TypeInfo::new(argument_flag | TypeFlags::SIZED | TypeFlags::COPY, 0)
            }
            Ti::Array { base, size, stride } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                let base_info = &self.types[base.index()];
                if !base_info.flags.contains(TypeFlags::DATA | TypeFlags::SIZED) {
                    return Err(TypeError::InvalidArrayBaseType(base));
                }

                let base_layout = self.layouter[base];
                let expected_stride = base_layout.to_stride();
                if stride != expected_stride {
                    return Err(TypeError::InvalidArrayStride {
                        stride,
                        expected: expected_stride,
                    });
                }

                let general_alignment = base_layout.alignment.get();
                let uniform_layout = match base_info.uniform_layout {
                    Ok(base_alignment) => {
                        // combine the alignment requirements
                        let align = base_alignment
                            .unwrap()
                            .get()
                            .max(general_alignment)
                            .max(UNIFORM_MIN_ALIGNMENT.get());
                        if stride % align != 0 {
                            Err((
                                handle,
                                Disalignment::ArrayStride {
                                    stride,
                                    alignment: align,
                                },
                            ))
                        } else {
                            Ok(Alignment::new(align))
                        }
                    }
                    Err(e) => Err(e),
                };
                let storage_layout = match base_info.storage_layout {
                    Ok(base_alignment) => {
                        let align = base_alignment.unwrap().get().max(general_alignment);
                        if stride % align != 0 {
                            Err((
                                handle,
                                Disalignment::ArrayStride {
                                    stride,
                                    alignment: align,
                                },
                            ))
                        } else {
                            Ok(Alignment::new(align))
                        }
                    }
                    Err(e) => Err(e),
                };

                let sized_flag = match size {
                    crate::ArraySize::Constant(const_handle) => {
                        let constant = constants.try_get(const_handle)?;
                        let length_is_positive = match *constant {
                            crate::Constant {
                                specialization: Some(_),
                                ..
                            } => {
                                // Many of our back ends don't seem to support
                                // specializable array lengths. If you want to try to make
                                // this work, be sure to address all uses of
                                // `Constant::to_array_length`, which ignores
                                // specialization.
                                return Err(TypeError::UnsupportedSpecializedArrayLength(
                                    const_handle,
                                ));
                            }
                            crate::Constant {
                                inner:
                                    crate::ConstantInner::Scalar {
                                        width: _,
                                        value: crate::ScalarValue::Uint(length),
                                    },
                                ..
                            } => length > 0,
                            // Accept a signed integer size to avoid
                            // requiring an explicit uint
                            // literal. Type inference should make
                            // this unnecessary.
                            crate::Constant {
                                inner:
                                    crate::ConstantInner::Scalar {
                                        width: _,
                                        value: crate::ScalarValue::Sint(length),
                                    },
                                ..
                            } => length > 0,
                            _ => {
                                log::warn!("Array size {:?}", constant);
                                return Err(TypeError::InvalidArraySizeConstant(const_handle));
                            }
                        };

                        if !length_is_positive {
                            return Err(TypeError::NonPositiveArrayLength(const_handle));
                        }

                        TypeFlags::SIZED | TypeFlags::ARGUMENT
                    }
                    crate::ArraySize::Dynamic => {
                        // Non-SIZED types may only appear as the last element of a structure.
                        // This is enforced by checks for SIZED-ness for all compound types,
                        // and a special case for structs.
                        TypeFlags::empty()
                    }
                };

                let base_mask = TypeFlags::COPY | TypeFlags::HOST_SHAREABLE;
                TypeInfo {
                    flags: TypeFlags::DATA | (base_info.flags & base_mask) | sized_flag,
                    uniform_layout,
                    storage_layout,
                }
            }
            Ti::Struct { ref members, span } => {
                if members.is_empty() {
                    return Err(TypeError::EmptyStruct);
                }

                let mut ti = TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::COPY
                        | TypeFlags::HOST_SHAREABLE
                        | TypeFlags::IO_SHAREABLE
                        | TypeFlags::ARGUMENT,
                    1,
                );
                ti.uniform_layout = Ok(Some(UNIFORM_MIN_ALIGNMENT));

                let mut min_offset = 0;

                let mut prev_struct_data: Option<(u32, u32)> = None;

                for (i, member) in members.iter().enumerate() {
                    if member.ty >= handle {
                        return Err(TypeError::UnresolvedBase(member.ty));
                    }
                    let base_info = &self.types[member.ty.index()];
                    if !base_info.flags.contains(TypeFlags::DATA) {
                        return Err(TypeError::InvalidData(member.ty));
                    }
                    if !base_info.flags.contains(TypeFlags::HOST_SHAREABLE) {
                        if ti.uniform_layout.is_ok() {
                            ti.uniform_layout = Err((member.ty, Disalignment::NonHostShareable));
                        }
                        if ti.storage_layout.is_ok() {
                            ti.storage_layout = Err((member.ty, Disalignment::NonHostShareable));
                        }
                    }
                    ti.flags &= base_info.flags;

                    if member.offset < min_offset {
                        //HACK: this could be nicer. We want to allow some structures
                        // to not bother with offsets/alignments if they are never
                        // used for host sharing.
                        if member.offset == 0 {
                            ti.flags.set(TypeFlags::HOST_SHAREABLE, false);
                        } else {
                            return Err(TypeError::MemberOverlap {
                                index: i as u32,
                                offset: member.offset,
                            });
                        }
                    }

                    //Note: `unwrap()` is fine because `Layouter` goes first and checks this
                    let base_size = types[member.ty].inner.size(constants);
                    min_offset = member.offset + base_size;
                    if min_offset > span {
                        return Err(TypeError::MemberOutOfBounds {
                            index: i as u32,
                            offset: member.offset,
                            size: base_size,
                            span,
                        });
                    }

                    check_member_layout(
                        &mut ti.uniform_layout,
                        member,
                        i as u32,
                        base_info.uniform_layout,
                        handle,
                    );
                    check_member_layout(
                        &mut ti.storage_layout,
                        member,
                        i as u32,
                        base_info.storage_layout,
                        handle,
                    );

                    // Validate rule: If a structure member itself has a structure type S,
                    // then the number of bytes between the start of that member and
                    // the start of any following member must be at least roundUp(16, SizeOf(S)).
                    if let Some((span, offset)) = prev_struct_data {
                        let diff = member.offset - offset;
                        let min = crate::valid::Layouter::round_up(UNIFORM_MIN_ALIGNMENT, span);
                        if diff < min {
                            ti.uniform_layout = Err((
                                handle,
                                Disalignment::MemberOffsetAfterStruct {
                                    index: i as u32,
                                    offset: member.offset,
                                    expected: offset + min,
                                },
                            ));
                        }
                    };

                    prev_struct_data = match types[member.ty].inner {
                        crate::TypeInner::Struct { span, .. } => Some((span, member.offset)),
                        _ => None,
                    };

                    // The last field may be an unsized array.
                    if !base_info.flags.contains(TypeFlags::SIZED) {
                        let is_array = match types[member.ty].inner {
                            crate::TypeInner::Array { .. } => true,
                            _ => false,
                        };
                        if !is_array || i + 1 != members.len() {
                            let name = member.name.clone().unwrap_or_default();
                            return Err(TypeError::InvalidDynamicArray(name, member.ty));
                        }
                        if ti.uniform_layout.is_ok() {
                            ti.uniform_layout =
                                Err((handle, Disalignment::UnsizedMember { index: i as u32 }));
                        }
                    }
                }

                let alignment = self.layouter[handle].alignment.get();
                if span % alignment != 0 {
                    ti.uniform_layout = Err((handle, Disalignment::StructSpan { span, alignment }));
                    ti.storage_layout = Err((handle, Disalignment::StructSpan { span, alignment }));
                }

                ti
            }
            Ti::Image { .. } | Ti::Sampler { .. } => TypeInfo::new(TypeFlags::ARGUMENT, 0),
            Ti::BindingArray { .. } => TypeInfo::new(TypeFlags::empty(), 0),
        })
    }
}
