//! Utilitary math functions.

use std::ops::{Add, Rem, Sub};

///
/// Aligns a `value` to an `alignment`.
///
/// Returns the first number greater than or equal to `value` that is also a
/// multiple of `alignment`. If `value` is already a multiple of `alignment`,
/// `value` will be returned.
///
/// # Examples
///
/// ```
/// # use wgpu_types::math::align_to;
/// assert_eq!(align_to(253, 16), 256);
/// assert_eq!(align_to(256, 16), 256);
/// assert_eq!(align_to(0, 16), 0);
/// ```
///
pub fn align_to<T>(value: T, alignment: T) -> T
where
    T: Add<Output = T> + Copy + Default + PartialEq<T> + Rem<Output = T> + Sub<Output = T>,
{
    let remainder = value % alignment;
    if remainder == T::default() {
        value
    } else {
        value + alignment - remainder
    }
}

/// Returns the next representable f32 value after `value`.
///
/// Copied from the unstable https://doc.rust-lang.org/src/core/num/f32.rs.html#710-730
pub fn f32_next(value: f32) -> f32 {
    // We must use strictly integer arithmetic to prevent denormals from
    // flushing to zero after an arithmetic operation on some platforms.
    const TINY_BITS: u32 = 0x1; // Smallest positive f32.
    const CLEAR_SIGN_MASK: u32 = 0x7fff_ffff;

    let bits = value.to_bits();
    if value.is_nan() || bits == f32::INFINITY.to_bits() {
        return value;
    }

    let abs = bits & CLEAR_SIGN_MASK;
    let next_bits = if abs == 0 {
        TINY_BITS
    } else if bits == abs {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next_bits)
}
