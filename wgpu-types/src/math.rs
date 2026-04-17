//! Utility math functions.

/// Trait for adjusting integers to the next multiple of an alignment.
pub trait AlignTo: Copy {
    /// Aligns `self` to `alignment`.
    ///
    /// Panics if doing so would overflow.
    fn align_to(self, alignment: Self) -> Self;
}

macro_rules! impl_align_to {
    ($ty:ty) => {
        impl AlignTo for $ty {
            fn align_to(self, alignment: Self) -> Self {
                self.checked_next_multiple_of(alignment).unwrap()
            }
        }
    };
}

impl_align_to!(u32);
impl_align_to!(u64);
impl_align_to!(usize);

/// Aligns a `value` to an `alignment`.
///
/// Returns the first number greater than or equal to `value` that is also a
/// multiple of `alignment`. If `value` is already a multiple of `alignment`,
/// `value` will be returned.
///
/// # Panics
///
/// If aligning `value` to `alignment` would overflow.
///
/// # Examples
///
/// ```
/// # use wgpu_types::math::align_to;
/// assert_eq!(align_to(253_u32, 16), 256);
/// assert_eq!(align_to(256_u32, 16), 256);
/// assert_eq!(align_to(0_u32, 16), 0);
/// ```
///
pub fn align_to<T: AlignTo>(value: T, alignment: T) -> T {
    value.align_to(alignment)
}
