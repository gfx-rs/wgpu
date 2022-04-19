use crate::arena::{Arena, BadHandle, Handle, UniqueArena};
use std::{num::NonZeroU32, ops};

pub type Alignment = NonZeroU32;

/// Size and alignment information for a type.
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct TypeLayout {
    pub size: u32,
    pub alignment: Alignment,
}

impl TypeLayout {
    /// Produce the stride as if this type is a base of an array.
    pub const fn to_stride(&self) -> u32 {
        Layouter::round_up(self.alignment, self.size)
    }
}

/// Helper processor that derives the sizes of all types.
///
/// `Layouter` uses the default layout algorithm/table, described in
/// [WGSL §4.3.7, "Memory Layout"]
///
/// A `Layouter` may be indexed by `Handle<Type>` values: `layouter[handle]` is the
/// layout of the type whose handle is `handle`.
///
/// [WGSL §4.3.7, "Memory Layout"](https://gpuweb.github.io/gpuweb/wgsl/#memory-layouts)
#[derive(Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct Layouter {
    /// Layouts for types in an arena, indexed by `Handle` index.
    layouts: Vec<TypeLayout>,
}

impl ops::Index<Handle<crate::Type>> for Layouter {
    type Output = TypeLayout;
    fn index(&self, handle: Handle<crate::Type>) -> &TypeLayout {
        &self.layouts[handle.index()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum LayoutErrorInner {
    #[error("Array element type {0:?} doesn't exist")]
    InvalidArrayElementType(Handle<crate::Type>),
    #[error("Struct member[{0}] type {1:?} doesn't exist")]
    InvalidStructMemberType(u32, Handle<crate::Type>),
    #[error("Zero width is not supported")]
    ZeroWidth,
    #[error("Array size is a bad handle")]
    BadHandle(#[from] BadHandle),
}

#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[error("Error laying out type {ty:?}: {inner}")]
pub struct LayoutError {
    pub ty: Handle<crate::Type>,
    pub inner: LayoutErrorInner,
}

impl LayoutErrorInner {
    const fn with(self, ty: Handle<crate::Type>) -> LayoutError {
        LayoutError { ty, inner: self }
    }
}

impl Layouter {
    /// Remove all entries from this `Layouter`, retaining storage.
    pub fn clear(&mut self) {
        self.layouts.clear();
    }

    /// Round `offset` up to the nearest `alignment` boundary.
    pub const fn round_up(alignment: Alignment, offset: u32) -> u32 {
        match offset & (alignment.get() - 1) {
            0 => offset,
            other => offset + alignment.get() - other,
        }
    }

    /// Return the offset and span of a struct member.
    ///
    /// The member must fall at or after `offset`. The member's alignment and
    /// size are `align` and `size` if given, defaulting to the values this
    /// `Layouter` has previously determined for `ty`.
    ///
    /// The return value is the range of offsets within the containing struct to
    /// reserve for this member, along with the alignment used. The containing
    /// struct must have sufficient space and alignment to accommodate these.
    pub fn member_placement(
        &self,
        offset: u32,
        ty: Handle<crate::Type>,
        align: Option<Alignment>,
        size: Option<NonZeroU32>,
    ) -> (ops::Range<u32>, Alignment) {
        let layout = self.layouts[ty.index()];
        let alignment = align.unwrap_or(layout.alignment);
        let start = Self::round_up(alignment, offset);
        let span = match size {
            Some(size) => size.get(),
            None => layout.size,
        };
        (start..start + span, alignment)
    }

    /// Extend this `Layouter` with layouts for any new entries in `types`.
    ///
    /// Ensure that every type in `types` has a corresponding [TypeLayout] in
    /// [`self.layouts`].
    ///
    /// Some front ends need to be able to compute layouts for existing types
    /// while module construction is still in progress and new types are still
    /// being added. This function assumes that the `TypeLayout` values already
    /// present in `self.layouts` cover their corresponding entries in `types`,
    /// and extends `self.layouts` as needed to cover the rest. Thus, a front
    /// end can call this function at any time, passing its current type and
    /// constant arenas, and then assume that layouts are available for all
    /// types.
    #[allow(clippy::or_fun_call)]
    pub fn update(
        &mut self,
        types: &UniqueArena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> Result<(), LayoutError> {
        use crate::TypeInner as Ti;

        for (ty_handle, ty) in types.iter().skip(self.layouts.len()) {
            let size = ty
                .inner
                .try_size(constants)
                .map_err(|error| LayoutErrorInner::BadHandle(error).with(ty_handle))?;
            let layout = match ty.inner {
                Ti::Scalar { width, .. } | Ti::Atomic { width, .. } => TypeLayout {
                    size,
                    alignment: Alignment::new(width as u32)
                        .ok_or(LayoutErrorInner::ZeroWidth.with(ty_handle))?,
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
                        Alignment::new(count * width as u32)
                            .ok_or(LayoutErrorInner::ZeroWidth.with(ty_handle))?
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
                        Alignment::new(count * width as u32)
                            .ok_or(LayoutErrorInner::ZeroWidth.with(ty_handle))?
                    },
                },
                Ti::Pointer { .. } | Ti::ValuePointer { .. } => TypeLayout {
                    size,
                    alignment: Alignment::new(1).unwrap(),
                },
                Ti::Array {
                    base,
                    stride: _,
                    size: _,
                } => TypeLayout {
                    size,
                    alignment: if base < ty_handle {
                        self[base].alignment
                    } else {
                        return Err(LayoutErrorInner::InvalidArrayElementType(base).with(ty_handle));
                    },
                },
                Ti::Struct { span, ref members } => {
                    let mut alignment = Alignment::new(1).unwrap();
                    for (index, member) in members.iter().enumerate() {
                        alignment = if member.ty < ty_handle {
                            alignment.max(self[member.ty].alignment)
                        } else {
                            return Err(LayoutErrorInner::InvalidStructMemberType(
                                index as u32,
                                member.ty,
                            )
                            .with(ty_handle));
                        };
                    }
                    TypeLayout {
                        size: span,
                        alignment,
                    }
                }
                Ti::Image { .. } | Ti::Sampler { .. } | Ti::BindingArray { .. } => TypeLayout {
                    size,
                    alignment: Alignment::new(1).unwrap(),
                },
            };
            debug_assert!(size <= layout.size);
            self.layouts.push(layout);
        }

        Ok(())
    }
}
