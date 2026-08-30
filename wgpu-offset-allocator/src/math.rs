//! Small, panic-free integer helpers.
//!
//! Every function here is total over `u64` (or documents its precondition and
//! upholds it internally). They deliberately avoid the overflow-prone
//! `offset + size <= end` idioms in the C++ sources in favour of subtraction-based
//! comparisons and checked arithmetic.

/// Returns whether `v` is a power of two (and non-zero).
#[inline]
pub(crate) const fn is_pow2(v: u64) -> bool {
    v != 0 && (v & (v - 1)) == 0
}

/// Aligns `val` up to the next multiple of `alignment`, saturating at [`u64::MAX`]
/// instead of overflowing.
///
/// `alignment` must be a power of two (callers validate this up front).
#[inline]
pub(crate) fn align_up(val: u64, alignment: u64) -> u64 {
    debug_assert!(is_pow2(alignment));
    // (val + alignment - 1) & !(alignment - 1), but overflow-safe: if val is close to
    // u64::MAX the naive add overflows, so saturate.
    match val.checked_add(alignment - 1) {
        Some(sum) => sum & !(alignment - 1),
        None => {
            // val + (alignment-1) overflowed. The aligned-up value would be >= 2^64
            // unless val is already aligned. Detect the already-aligned case exactly.
            if val & (alignment - 1) == 0 {
                val
            } else {
                // Genuinely does not fit in u64.
                u64::MAX
            }
        }
    }
}

/// Index of the most significant set bit of `v` (0-based). `v` must be non-zero.
///
/// This is `63 - v.leading_zeros()`, matching VMA's `VmaBitScanMSB` for non-zero
/// input.
#[inline]
pub(crate) const fn bit_scan_msb_u64(v: u64) -> u32 {
    debug_assert!(v != 0);
    63 - v.leading_zeros()
}

/// Index of the least significant set bit of `v` (0-based). `v` must be non-zero.
///
/// This is `v.trailing_zeros()`, matching VMA's `VmaBitScanLSB` for non-zero input.
#[inline]
pub(crate) const fn bit_scan_lsb_u32(v: u32) -> u32 {
    debug_assert!(v != 0);
    v.trailing_zeros()
}

/// Index of the least significant set bit of `v` (0-based). `v` must be non-zero.
#[inline]
pub(crate) const fn bit_scan_lsb_u64(v: u64) -> u32 {
    debug_assert!(v != 0);
    v.trailing_zeros()
}

/// Computes `!0u64 << shift`, treating any `shift >= 64` as producing `0`.
///
/// In C++ a shift by >= the width is undefined behaviour; VMA/D3D12MA rely on the
/// shift amount staying in range because memory classes stay small for realistic
/// block sizes. We make the "everything masked off" behaviour explicit and total so
/// the full `u64` size range is safe.
#[inline]
pub(crate) const fn shl_all_ones_u64(shift: u32) -> u64 {
    if shift >= 64 {
        0
    } else {
        u64::MAX << shift
    }
}

/// Computes `!0u32 << shift`, treating any `shift >= 32` as producing `0`.
#[inline]
pub(crate) const fn shl_all_ones_u32(shift: u32) -> u32 {
    if shift >= 32 {
        0
    } else {
        u32::MAX << shift
    }
}

/// Whether the ranges `[a_offset, a_offset + a_size)` and the page containing
/// `b_offset` share a granularity page.
///
/// Port of VMA's `VmaBlocksOnSamePage`, written overflow-safely. Precondition (as in
/// VMA): `a_offset + a_size <= b_offset`, `a_size > 0`, `page_size > 0`. `page_size`
/// must be a power of two.
///
/// Only used by the test-only reference model (the production TLSF path checks
/// granularity conflicts through the granularity module).
#[cfg(all(test, not(target_arch = "wasm32")))]
#[inline]
pub(crate) fn blocks_on_same_page(
    a_offset: u64,
    a_size: u64,
    b_offset: u64,
    page_size: u64,
) -> bool {
    debug_assert!(a_size > 0 && page_size > 0);
    // a_end = a_offset + a_size - 1. a_size > 0 guarantees no underflow; the sum
    // cannot overflow because the precondition bounds it below b_offset <= u64::MAX.
    let a_end = a_offset + (a_size - 1);
    let mask = !(page_size - 1);
    (a_end & mask) == (b_offset & mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(11, 8), 16);
        assert_eq!(align_up(16, 8), 16);
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 1), 1);
    }

    #[test]
    fn align_up_saturates() {
        // Already aligned near the top: returns itself.
        let aligned_max = !0xFu64;
        assert_eq!(align_up(aligned_max, 16), aligned_max);
        // Not aligned and cannot fit: saturates.
        assert_eq!(align_up(u64::MAX, 16), u64::MAX);
        assert_eq!(align_up(u64::MAX - 1, 16), u64::MAX);
    }

    #[test]
    fn shifts_are_total() {
        assert_eq!(shl_all_ones_u64(0), u64::MAX);
        assert_eq!(shl_all_ones_u64(63), 1u64 << 63);
        assert_eq!(shl_all_ones_u64(64), 0);
        assert_eq!(shl_all_ones_u64(100), 0);
        assert_eq!(shl_all_ones_u32(0), u32::MAX);
        assert_eq!(shl_all_ones_u32(31), 1u32 << 31);
        assert_eq!(shl_all_ones_u32(32), 0);
    }

    #[test]
    fn bitscan() {
        assert_eq!(bit_scan_msb_u64(1), 0);
        assert_eq!(bit_scan_msb_u64(256), 8);
        assert_eq!(bit_scan_msb_u64(u64::MAX), 63);
        assert_eq!(bit_scan_lsb_u32(1), 0);
        assert_eq!(bit_scan_lsb_u32(0x8000_0000), 31);
    }
}
