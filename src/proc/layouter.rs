use crate::arena::{Arena, BadHandle, Handle, UniqueArena};
use std::{fmt::Display, num::NonZeroU32, ops};

/// A newtype struct where its only valid values are powers of 2
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct Alignment(NonZeroU32);

impl Alignment {
    pub const ONE: Self = Self(unsafe { NonZeroU32::new_unchecked(1) });
    pub const TWO: Self = Self(unsafe { NonZeroU32::new_unchecked(2) });
    pub const FOUR: Self = Self(unsafe { NonZeroU32::new_unchecked(4) });
    pub const EIGHT: Self = Self(unsafe { NonZeroU32::new_unchecked(8) });
    pub const SIXTEEN: Self = Self(unsafe { NonZeroU32::new_unchecked(16) });

    pub const MIN_UNIFORM: Self = Self::SIXTEEN;

    pub const fn new(n: u32) -> Option<Self> {
        if n.is_power_of_two() {
            // SAFETY: value can't be 0 since we just checked if it's a power of 2
            Some(Self(unsafe { NonZeroU32::new_unchecked(n) }))
        } else {
            None
        }
    }

    /// # Panics
    /// If `width` is not a power of 2
    pub fn from_width(width: u8) -> Self {
        Self::new(width as u32).unwrap()
    }

    /// Returns whether or not `n` is a multiple of this alignment.
    pub const fn is_aligned(&self, n: u32) -> bool {
        // equivalent to: `n % self.0.get() == 0` but much faster
        n & (self.0.get() - 1) == 0
    }

    /// Round `n` up to the nearest alignment boundary.
    pub const fn round_up(&self, n: u32) -> u32 {
        // equivalent to:
        // match n % self.0.get() {
        //     0 => n,
        //     rem => n + (self.0.get() - rem),
        // }
        let mask = self.0.get() - 1;
        (n + mask) & !mask
    }
}

impl Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.get().fmt(f)
    }
}

impl ops::Mul<u32> for Alignment {
    type Output = u32;

    fn mul(self, rhs: u32) -> Self::Output {
        self.0.get() * rhs
    }
}

impl ops::Mul for Alignment {
    type Output = Alignment;

    fn mul(self, rhs: Alignment) -> Self::Output {
        // SAFETY: both lhs and rhs are powers of 2, the result will be a power of 2
        Self(unsafe { NonZeroU32::new_unchecked(self.0.get() * rhs.0.get()) })
    }
}

impl From<crate::VectorSize> for Alignment {
    fn from(size: crate::VectorSize) -> Self {
        match size {
            crate::VectorSize::Bi => Alignment::TWO,
            crate::VectorSize::Tri => Alignment::FOUR,
            crate::VectorSize::Quad => Alignment::FOUR,
        }
    }
}

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
        self.alignment.round_up(self.size)
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
    #[error("Type width must be a power of two")]
    NonPowerOfTwoWidth,
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
                Ti::Scalar { width, .. } | Ti::Atomic { width, .. } => {
                    let alignment = Alignment::new(width as u32)
                        .ok_or(LayoutErrorInner::NonPowerOfTwoWidth.with(ty_handle))?;
                    TypeLayout { size, alignment }
                }
                Ti::Vector {
                    size: vec_size,
                    width,
                    ..
                } => {
                    let alignment = Alignment::new(width as u32)
                        .ok_or(LayoutErrorInner::NonPowerOfTwoWidth.with(ty_handle))?;
                    TypeLayout {
                        size,
                        alignment: Alignment::from(vec_size) * alignment,
                    }
                }
                Ti::Matrix {
                    columns: _,
                    rows,
                    width,
                } => {
                    let alignment = Alignment::new(width as u32)
                        .ok_or(LayoutErrorInner::NonPowerOfTwoWidth.with(ty_handle))?;
                    TypeLayout {
                        size,
                        alignment: Alignment::from(rows) * alignment,
                    }
                }
                Ti::Pointer { .. } | Ti::ValuePointer { .. } => TypeLayout {
                    size,
                    alignment: Alignment::ONE,
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
                    let mut alignment = Alignment::ONE;
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
                    alignment: Alignment::ONE,
                },
            };
            debug_assert!(size <= layout.size);
            self.layouts.push(layout);
        }

        Ok(())
    }
}
