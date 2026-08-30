//! In-block buffer-image granularity handling.
//!
//! Port of `VmaBlockBufferImageGranularity` (vk_mem_alloc.h). Used by [`Tlsf`].
//!
//! Three regimes, keyed on the granularity value:
//!
//! - `granularity <= 1` or virtual block: disabled entirely.
//! - `1 < granularity <= 256` ([`MAX_LOW_BUFFER_IMAGE_GRANULARITY`]): no page
//!   tracking; [`roundup_alloc_request`](BufferImageGranularity::roundup_alloc_request)
//!   just bumps the alignment and size up to the granularity for the conservative
//!   allocation types.
//! - `granularity > 256`: per-page tracking. Each page records the type and count of
//!   allocations that touch it; only the first and last page an allocation touches
//!   are tracked (VMA does the same).
//!
//! [`Tlsf`]: crate::Tlsf

use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;

use crate::math::{align_up, bit_scan_msb_u64};
use crate::{AllocationType, CreateError};

/// Granularity values at or below this use the cheap "round up" path with no page
/// tracking (`VmaBlockBufferImageGranularity::MAX_LOW_BUFFER_IMAGE_GRANULARITY`).
pub(crate) const MAX_LOW_BUFFER_IMAGE_GRANULARITY: u64 = 256;

/// Upper bound on the number of page-tracking [`RegionInfo`] entries a single block
/// may require, i.e. the largest `ceil(size / granularity)` we will service.
///
/// Page tracking allocates one [`RegionInfo`] per granularity page up front. A
/// hostile `(size, granularity)` pair (e.g. `size == u64::MAX`, `granularity == 512`)
/// would otherwise ask for an astronomically large [`Vec`], which aborts the process
/// on a 64-bit target (allocation failure) or silently truncates on a 32-bit target
/// such as `wasm32` (the `as usize` cast wraps, disabling page tracking and letting
/// granularity conflicts go undetected). We reject such pairs at construction with
/// [`CreateError::GranularityTrackingTooLarge`].
///
/// The cap is `2^26` (~67M) pages. At [`RegionInfo`]'s size that is a bounded, sane
/// tracking allocation (well under the ~2 GiB `isize::MAX` byte ceiling on 32-bit) and
/// is far larger than any realistic GPU heap divided by any realistic granularity: a
/// 256 GiB heap at a 4 KiB granularity is only 2^26 pages, and real granularities
/// (typically ≤ 64 KiB) with real heap sizes stay orders of magnitude below this.
pub(crate) const MAX_TRACKED_PAGES: u64 = 1 << 26;

/// Per-page bookkeeping (`VmaBlockBufferImageGranularity::RegionInfo`).
///
/// `alloc_count` is a [`u32`] (VMA uses `uint16_t`): with a large granularity a single
/// page can host up to `granularity` unit-sized allocations, which can exceed
/// `u16::MAX`. The live-allocation count is bounded by the TLSF `u32` node arena, so a
/// `u32` per-page counter cannot overflow in practice; increments still guard
/// defensively (see [`alloc_page`](BufferImageGranularity::alloc_page)).
#[derive(Clone, Copy, Debug, Default)]
struct RegionInfo {
    alloc_type: AllocationType,
    alloc_count: u32,
}

/// Buffer-image granularity tracker for a single block.
#[derive(Debug)]
pub(crate) struct BufferImageGranularity {
    granularity: u64,
    /// Per-page info; empty unless [`is_page_tracking`](Self::is_page_tracking).
    regions: Vec<RegionInfo>,
    /// `log2(granularity)`, precomputed for offset -> page index. Meaningful only
    /// when page tracking is enabled.
    page_shift: u32,
}

impl BufferImageGranularity {
    /// Creates a tracker for a block of `size` units.
    ///
    /// `granularity` is assumed already validated (power of two, or `1` to disable).
    /// When `is_virtual`, page tracking is never enabled (VMA disables granularity
    /// for virtual blocks).
    ///
    /// # Errors
    ///
    /// Returns [`CreateError::GranularityTrackingTooLarge`] when page tracking is
    /// active and `ceil(size / granularity)` exceeds [`MAX_TRACKED_PAGES`] (or, more
    /// conservatively, would not fit in a [`usize`] on the target). This bounds the
    /// up-front tracking allocation so a hostile `(size, granularity)` pair cannot
    /// abort the process (64-bit) or silently truncate the page count (32-bit).
    pub(crate) fn new(granularity: u64, size: u64, is_virtual: bool) -> Result<Self, CreateError> {
        let page_tracking = !is_virtual && granularity > MAX_LOW_BUFFER_IMAGE_GRANULARITY;
        let (regions, page_shift) = if page_tracking {
            // Number of pages = ceil(size / granularity). Validate it against a sane
            // cap *and* the platform `usize`/`isize` limits before allocating, so a
            // hostile size/granularity pair fails cleanly instead of OOM-aborting or
            // (on 32-bit) truncating the page count to zero.
            let region_count = size.div_ceil(granularity);
            if region_count > MAX_TRACKED_PAGES {
                return Err(CreateError::GranularityTrackingTooLarge);
            }
            // Defense in depth for narrow-usize targets (wasm32): reject if the count
            // does not fit in usize or the byte size would exceed isize::MAX. The
            // MAX_TRACKED_PAGES cap already keeps us well under this on all targets,
            // but check explicitly rather than trust the cast.
            let region_count = usize::try_from(region_count)
                .map_err(|_| CreateError::GranularityTrackingTooLarge)?;
            let byte_size = region_count
                .checked_mul(size_of::<RegionInfo>())
                .filter(|&b| b <= isize::MAX as usize)
                .ok_or(CreateError::GranularityTrackingTooLarge)?;
            let _ = byte_size;
            let shift = bit_scan_msb_u64(granularity);
            (vec![RegionInfo::default(); region_count], shift)
        } else {
            (Vec::new(), 0)
        };
        Ok(BufferImageGranularity {
            granularity,
            regions,
            page_shift,
        })
    }

    /// Whether the cheap "round up size/alignment" behaviour applies (granularity in
    /// `(1, 256]`). Distinct from page tracking.
    #[inline]
    fn is_low(&self) -> bool {
        self.granularity > 1 && self.granularity <= MAX_LOW_BUFFER_IMAGE_GRANULARITY
    }

    /// Whether per-page tracking is active (`IsEnabled` in VMA).
    #[inline]
    fn is_page_tracking(&self) -> bool {
        !self.regions.is_empty()
    }

    /// For low granularities, bump the request's alignment and size up to the
    /// granularity for the conservative allocation types.
    ///
    /// Port of `RoundupAllocRequest`. Returns the (possibly increased) size and
    /// alignment. Size uses saturating align-up so huge requests do not overflow.
    pub(crate) fn roundup_alloc_request(
        &self,
        alloc_type: AllocationType,
        size: u64,
        alignment: u64,
    ) -> (u64, u64) {
        if self.is_low()
            && matches!(
                alloc_type,
                AllocationType::Unknown
                    | AllocationType::ImageUnknown
                    | AllocationType::ImageOptimal
            )
        {
            let alignment = alignment.max(self.granularity);
            let size = align_up(size, self.granularity);
            (size, alignment)
        } else {
            (size, alignment)
        }
    }

    #[inline]
    fn offset_to_page(&self, offset: u64) -> usize {
        (offset >> self.page_shift) as usize
    }

    #[inline]
    fn start_page(&self, offset: u64) -> usize {
        // GetStartPage: page of the offset aligned down to the granularity.
        self.offset_to_page(offset & !(self.granularity - 1))
    }

    #[inline]
    fn end_page(&self, offset: u64, size: u64) -> usize {
        // GetEndPage: page of the last touched unit, aligned down to the granularity.
        // size >= 1 for all real allocations, so offset + size - 1 does not underflow.
        let last = offset.saturating_add(size - 1);
        self.offset_to_page(last & !(self.granularity - 1))
    }

    /// Checks whether placing an allocation of `alloc_size`/`alloc_type` at
    /// `*offset` would conflict with an existing allocation on the start or end page,
    /// aligning `*offset` up to the next page if the start page conflicts.
    ///
    /// Port of `CheckConflictAndAlignUp`. Returns `true` if the placement conflicts
    /// or no longer fits after alignment (i.e. reject), `false` if valid (with
    /// `*offset` possibly increased).
    ///
    /// Written overflow-safely: the "does it still fit" test is subtraction based.
    pub(crate) fn check_conflict_and_align_up(
        &self,
        offset: &mut u64,
        alloc_size: u64,
        block_offset: u64,
        block_size: u64,
        alloc_type: AllocationType,
    ) -> bool {
        if !self.is_page_tracking() {
            return false;
        }

        let start = self.start_page(*offset);
        if self.regions[start].alloc_count > 0
            && self.regions[start].alloc_type.conflicts_with(alloc_type)
        {
            *offset = align_up(*offset, self.granularity);
            // Overflow-safe form of `block_size < alloc_size + (*offset - block_offset)`:
            //   available = block_size - (*offset - block_offset)
            // If *offset ran past the block end, or the remaining space is smaller
            // than alloc_size, reject.
            let padding = offset.wrapping_sub(block_offset);
            if *offset < block_offset || padding > block_size {
                return true;
            }
            let available = block_size - padding;
            if available < alloc_size {
                return true;
            }
            let start = self.start_page(*offset);
            if self.regions[start].alloc_count > 0
                && self.regions[start].alloc_type.conflicts_with(alloc_type)
            {
                return true;
            }
        }

        let end = self.end_page(*offset, alloc_size);
        let start = self.start_page(*offset);
        if end != start
            && self.regions[end].alloc_count > 0
            && self.regions[end].alloc_type.conflicts_with(alloc_type)
        {
            return true;
        }

        // Reject cleanly if committing this placement would overflow a per-page
        // `alloc_count`. `alloc_pages` bumps the start page and (if distinct) the end
        // page, so those are the only counters at risk. This turns the pathological
        // ">u32::MAX allocations on one page" case into an ordinary OutOfSpace
        // rather than an arithmetic panic / silent wrap.
        if self.regions[start].alloc_count == u32::MAX {
            return true;
        }
        if end != start && self.regions[end].alloc_count == u32::MAX {
            return true;
        }

        false
    }

    /// Records that an allocation of `alloc_type` now occupies `[offset, offset+size)`.
    /// Port of `AllocPages`.
    pub(crate) fn alloc_pages(&mut self, alloc_type: AllocationType, offset: u64, size: u64) {
        if !self.is_page_tracking() {
            return;
        }
        let start = self.start_page(offset);
        Self::alloc_page(&mut self.regions[start], alloc_type);

        let end = self.end_page(offset, size);
        if start != end {
            Self::alloc_page(&mut self.regions[end], alloc_type);
        }
    }

    fn alloc_page(page: &mut RegionInfo, alloc_type: AllocationType) {
        // A free page (count 0) or a page whose live allocations are all "free" takes
        // on the new type. Port of AllocPage.
        if page.alloc_count == 0 || page.alloc_type == AllocationType::Free {
            page.alloc_type = alloc_type;
        }
        // Defensive saturating increment. `check_conflict_and_align_up` already rejects
        // any request that would push a touched page to `u32::MAX + 1`, so the
        // saturation branch is unreachable through the public API; the debug_assert
        // catches any future caller that forgets the pre-check.
        debug_assert!(
            page.alloc_count < u32::MAX,
            "granularity page alloc_count overflow"
        );
        page.alloc_count = page.alloc_count.saturating_add(1);
    }

    /// Records that the allocation at `[offset, offset+size)` was freed. Port of
    /// `FreePages`.
    pub(crate) fn free_pages(&mut self, offset: u64, size: u64) {
        if !self.is_page_tracking() {
            return;
        }
        let start = self.start_page(offset);
        Self::free_page(&mut self.regions[start]);
        let end = self.end_page(offset, size);
        if start != end {
            Self::free_page(&mut self.regions[end]);
        }
    }

    fn free_page(page: &mut RegionInfo) {
        // Underflow here would mean freeing a page with no recorded allocation, which
        // is an internal invariant breach (a double free would have been rejected at
        // the handle layer before reaching here). Saturate defensively so a corrupted
        // count can never wrap to u32::MAX in release; debug_assert flags the bug.
        debug_assert!(
            page.alloc_count > 0,
            "granularity page alloc_count underflow"
        );
        page.alloc_count = page.alloc_count.saturating_sub(1);
        if page.alloc_count == 0 {
            page.alloc_type = AllocationType::Free;
        }
    }

    /// Resets all page tracking (port of `Clear`).
    pub(crate) fn clear(&mut self) {
        for r in &mut self.regions {
            *r = RegionInfo::default();
        }
    }

    /// Starts a validation pass, returning a per-page counter vector to be filled by
    /// [`validate`](Self::validate) and checked by
    /// [`finish_validation`](Self::finish_validation).
    pub(crate) fn start_validation(&self) -> Vec<u32> {
        if self.is_page_tracking() {
            vec![0u32; self.regions.len()]
        } else {
            Vec::new()
        }
    }

    /// Records that an allocation occupies `[offset, offset+size)` and checks the
    /// touched pages have non-zero counts. Port of `Validate`.
    pub(crate) fn validate(
        &self,
        page_allocs: &mut [u32],
        offset: u64,
        size: u64,
    ) -> Result<(), &'static str> {
        if !self.is_page_tracking() {
            return Ok(());
        }
        let start = self.start_page(offset);
        page_allocs[start] += 1;
        if self.regions[start].alloc_count == 0 {
            return Err("granularity: start page has zero alloc count for a live allocation");
        }
        let end = self.end_page(offset, size);
        if start != end {
            page_allocs[end] += 1;
            if self.regions[end].alloc_count == 0 {
                return Err("granularity: end page has zero alloc count for a live allocation");
            }
        }
        Ok(())
    }

    /// Checks the accumulated per-page counts match the tracked counts. Port of
    /// `FinishValidation`.
    pub(crate) fn finish_validation(&self, page_allocs: &[u32]) -> Result<(), &'static str> {
        if !self.is_page_tracking() {
            return Ok(());
        }
        for (page, &counted) in self.regions.iter().zip(page_allocs.iter()) {
            if counted != page.alloc_count {
                return Err("granularity: per-page alloc count mismatch");
            }
        }
        Ok(())
    }
}
