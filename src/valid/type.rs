use super::Capabilities;
use crate::{
    arena::{Arena, Handle, UniqueArena},
    proc::Alignment,
};

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

        /// Can be be used for interfacing between pipeline stages.
        ///
        /// This includes non-bool scalars and vectors, matrices, and structs
        /// and arrays containing only interface types.
        const INTERFACE = 0x8;

        /// Can be used for host-shareable structures.
        const HOST_SHARED = 0x10;

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
    #[error("The struct member[{index}] is not statically sized")]
    UnsizedMember { index: u32 },
    #[error("The type is not host-shareable")]
    NonHostShareable,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum TypeError {
    #[error("The {0:?} scalar width {1} is not supported")]
    InvalidWidth(crate::ScalarKind, crate::Bytes),
    #[error("The {0:?} scalar width {1} is not supported for an atomic")]
    InvalidAtomicWidth(crate::ScalarKind, crate::Bytes),
    #[error("The base handle {0:?} can not be resolved")]
    UnresolvedBase(Handle<crate::Type>),
    #[error("Invalid type for pointer target {0:?}")]
    InvalidPointerBase(Handle<crate::Type>),
    #[error("Unsized types like {base:?} must be in the `Storage` storage class, not `{class:?}`")]
    InvalidPointerToUnsized {
        base: Handle<crate::Type>,
        class: crate::StorageClass,
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
    #[error("Array stride {stride} is smaller than the base element size {base_size}")]
    InsufficientArrayStride { stride: u32, base_size: u32 },
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
        let alignment = Alignment::new(align);
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
                TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::COPY
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED
                        | TypeFlags::ARGUMENT,
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
                        | TypeFlags::COPY
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED
                        | TypeFlags::ARGUMENT,
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
                        | TypeFlags::INTERFACE
                        | TypeFlags::HOST_SHARED
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
                    TypeFlags::DATA | TypeFlags::SIZED | TypeFlags::HOST_SHARED,
                    width as u32,
                )
            }
            Ti::Pointer { base, class } => {
                use crate::StorageClass as Sc;

                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }

                let base_info = &self.types[base.index()];
                if !base_info.flags.contains(TypeFlags::DATA) {
                    return Err(TypeError::InvalidPointerBase(base));
                }

                // Runtime-sized values can only live in the `Storage` storage
                // class, so it's useless to have a pointer to such a type in
                // any other class.
                //
                // Detecting this problem here prevents the definition of
                // functions like:
                //
                //     fn f(p: ptr<workgroup, UnsizedType>) -> ... { ... }
                //
                // which would otherwise be permitted, but uncallable. (They
                // may also present difficulties in code generation).
                if !base_info.flags.contains(TypeFlags::SIZED) {
                    match class {
                        Sc::Storage { .. } => {}
                        _ => {
                            return Err(TypeError::InvalidPointerToUnsized { base, class });
                        }
                    }
                }

                // Pointers passed as arguments to user-defined functions must
                // be in the `Function`, `Private`, or `Workgroup` storage
                // class. We only mark pointers in those classes as `ARGUMENT`.
                //
                // `Validator::validate_function` actually checks the storage
                // class of pointer arguments explicitly before checking the
                // `ARGUMENT` flag, to give better error messages. But it seems
                // best to set `ARGUMENT` accurately anyway.
                let argument_flag = match class {
                    Sc::Function | Sc::Private | Sc::WorkGroup => TypeFlags::ARGUMENT,
                    Sc::Uniform | Sc::Storage { .. } | Sc::Handle | Sc::PushConstant => {
                        TypeFlags::empty()
                    }
                };

                // Pointers cannot be stored in variables, structure members, or
                // array elements, so we do not mark them as `DATA`.
                TypeInfo::new(argument_flag | TypeFlags::SIZED | TypeFlags::COPY, 0)
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
                TypeInfo::new(TypeFlags::DATA | TypeFlags::SIZED | TypeFlags::COPY, 0)
            }
            Ti::Array { base, size, stride } => {
                if base >= handle {
                    return Err(TypeError::UnresolvedBase(base));
                }
                let base_info = &self.types[base.index()];
                if !base_info.flags.contains(TypeFlags::DATA | TypeFlags::SIZED) {
                    return Err(TypeError::InvalidArrayBaseType(base));
                }

                let base_size = types[base].inner.span(constants);
                if stride < base_size {
                    return Err(TypeError::InsufficientArrayStride { stride, base_size });
                }

                let general_alignment = self.layouter[base].alignment;
                let uniform_layout = match base_info.uniform_layout {
                    Ok(base_alignment) => {
                        // combine the alignment requirements
                        let align = ((base_alignment.unwrap().get() - 1)
                            | (general_alignment.get() - 1))
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
                            Ok(Alignment::new(align))
                        }
                    }
                    Err(e) => Err(e),
                };
                let storage_layout = match base_info.storage_layout {
                    Ok(base_alignment) => {
                        let align = ((base_alignment.unwrap().get() - 1)
                            | (general_alignment.get() - 1))
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
                            Ok(Alignment::new(align))
                        }
                    }
                    Err(e) => Err(e),
                };

                let sized_flag = match size {
                    crate::ArraySize::Constant(const_handle) => {
                        let length_is_positive = match constants.try_get(const_handle) {
                            Some(&crate::Constant {
                                specialization: Some(_),
                                ..
                            }) => {
                                // Many of our back ends don't seem to support
                                // specializable array lengths. If you want to try to make
                                // this work, be sure to address all uses of
                                // `Constant::to_array_length`, which ignores
                                // specialization.
                                return Err(TypeError::UnsupportedSpecializedArrayLength(
                                    const_handle,
                                ));
                            }
                            Some(&crate::Constant {
                                inner:
                                    crate::ConstantInner::Scalar {
                                        width: _,
                                        value: crate::ScalarValue::Uint(length),
                                    },
                                ..
                            }) => length > 0,
                            // Accept a signed integer size to avoid
                            // requiring an explicit uint
                            // literal. Type inference should make
                            // this unnecessary.
                            Some(&crate::Constant {
                                inner:
                                    crate::ConstantInner::Scalar {
                                        width: _,
                                        value: crate::ScalarValue::Sint(length),
                                    },
                                ..
                            }) => length > 0,
                            other => {
                                log::warn!("Array size {:?}", other);
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

                let base_mask = TypeFlags::COPY | TypeFlags::HOST_SHARED | TypeFlags::INTERFACE;
                TypeInfo {
                    flags: TypeFlags::DATA | (base_info.flags & base_mask) | sized_flag,
                    uniform_layout,
                    storage_layout,
                }
            }
            Ti::Struct { ref members, span } => {
                let mut ti = TypeInfo::new(
                    TypeFlags::DATA
                        | TypeFlags::SIZED
                        | TypeFlags::COPY
                        | TypeFlags::HOST_SHARED
                        | TypeFlags::INTERFACE
                        | TypeFlags::ARGUMENT,
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
                    if !base_info.flags.contains(TypeFlags::HOST_SHARED) {
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
        })
    }
}
