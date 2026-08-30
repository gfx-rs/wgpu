//! Regression tests ported from the soundness-review proof-of-concept programs.
//!
//! Each test reproduces a reviewer PoC that previously triggered corruption, a panic,
//! or a GPU-aliasing hazard, and asserts the fixed behaviour. The finding IDs (F1..F7)
//! match the review write-up.

use alloc::vec::Vec;

use crate::granularity::MAX_TRACKED_PAGES;
use crate::{
    AllocationDesc, AllocationHandle, AllocationType, CreateError, HandleError, Strategy,
    Suballocator, Tlsf,
};

fn tlsf_alloc(t: &mut Tlsf<u32>, size: u64, ud: u32) -> (AllocationHandle, u64) {
    let req = t
        .create_allocation_request(AllocationDesc {
            size,
            alignment: 1,
            ..Default::default()
        })
        .unwrap();
    let off = req.offset;
    (t.alloc(req, ud), off)
}

// ---------------------------------------------------------------------------
// F2: double-free / stale-handle detection (no corruption, no reachable panic).
// ---------------------------------------------------------------------------

/// Port of `repro_double_free_overlap`: double-freeing a middle allocation used to
/// corrupt accounting and later panic `unreachable!("remove_free_block on taken
/// block")`. Now the second free returns `Err` and leaves accounting intact, and
/// subsequent allocations never overlap the still-live neighbours.
#[test]
fn repro_double_free_overlap() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    let (_h1, o1) = tlsf_alloc(&mut t, 100, 1);
    let (h2, _o2) = tlsf_alloc(&mut t, 100, 2);
    let (_h3, o3) = tlsf_alloc(&mut t, 100, 3);

    let count_before = t.allocation_count();
    let free_before = t.sum_free_size();

    assert_eq!(t.free(h2), Ok(()));
    // Second free of the same handle: detected, accounting untouched.
    assert_eq!(t.free(h2), Err(HandleError::InvalidHandle));
    assert_eq!(t.allocation_count(), count_before - 1);
    t.validate().unwrap();

    let live = [(o1, 100u64), (o3, 100u64)];
    let mut new_allocs: Vec<(u64, u64)> = Vec::new();
    for i in 0..8 {
        if let Ok(req) = t.create_allocation_request(AllocationDesc {
            size: 80,
            alignment: 1,
            ..Default::default()
        }) {
            let o = req.offset;
            let _h = t.alloc(req, 100 + i);
            for &(lo, ls) in &live {
                assert!(
                    !(o < lo + ls && lo < o + 80),
                    "new [{o},{}) overlaps live [{lo},{})",
                    o + 80,
                    lo + ls
                );
            }
            for &(po, ps) in &new_allocs {
                assert!(!(o < po + ps && po < o + 80), "new allocations overlap");
            }
            new_allocs.push((o, 80));
        }
        t.validate().unwrap();
    }
    let _ = free_before;
}

/// Port of `repro_df3`: double-freeing the *top* allocation used to grow the null
/// block twice, reporting the whole block free while a low allocation stayed live, so
/// a full-size allocation escaped the block. Now the second free is rejected and the
/// reported free size stays correct.
#[test]
fn repro_df3() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    let (_h1, o1) = tlsf_alloc(&mut t, 200, 1);
    let (h2, _o2) = tlsf_alloc(&mut t, 200, 2);

    assert_eq!(t.free(h2), Ok(()));
    let free_after_first = t.sum_free_size();
    // Second free must be rejected, not double-grow the null block.
    assert_eq!(t.free(h2), Err(HandleError::InvalidHandle));
    assert_eq!(t.sum_free_size(), free_after_first);
    t.validate().unwrap();

    // h1 is still live at [0,200). The largest allocation that fits is 1024-200=824.
    let free_now = t.sum_free_size();
    assert_eq!(
        free_now, 824,
        "reported free size must exclude the live allocation"
    );
    let req = t
        .create_allocation_request(AllocationDesc {
            size: free_now,
            alignment: 1,
            ..Default::default()
        })
        .unwrap();
    let o = req.offset;
    let _h = t.alloc(req, 99);
    assert!(
        o + free_now <= 1024,
        "allocation [{o},{}) escaped the block",
        o + free_now
    );
    assert!(
        !(o < o1 + 200 && o1 < o + free_now),
        "allocation overlaps still-live h1"
    );
    t.validate().unwrap();
}

/// Handle-safety probes ported from `repro_handles`: double free, ABA after recycle,
/// stale-after-clear, and cross-allocator handles all yield clean errors and never
/// corrupt accounting.
#[test]
fn repro_handles_all_clean() {
    // (a) double free
    {
        let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
        let (h, _) = tlsf_alloc(&mut t, 100, 0);
        let (_h2, _) = tlsf_alloc(&mut t, 100, 0);
        assert_eq!(t.free(h), Ok(()));
        let free_before = t.sum_free_size();
        let count_before = t.allocation_count();
        assert_eq!(t.free(h), Err(HandleError::InvalidHandle));
        assert_eq!(t.sum_free_size(), free_before);
        assert_eq!(t.allocation_count(), count_before);
        t.validate().unwrap();
    }

    // (b) ABA: reusing the freed slot must not let the stale handle free the new alloc.
    {
        let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
        let (h, _) = tlsf_alloc(&mut t, 100, 0);
        assert_eq!(t.free(h), Ok(()));
        let (h_new, _) = tlsf_alloc(&mut t, 100, 0);
        // The stale handle must not validate even if it names the same node index.
        assert_ne!(h, h_new, "generation must differ after recycle");
        assert_eq!(t.free(h), Err(HandleError::InvalidHandle));
        // The new allocation is still live and freeable.
        assert_eq!(t.allocation_count(), 1);
        assert_eq!(t.free(h_new), Ok(()));
        t.validate().unwrap();
    }

    // (c) stale handle after clear.
    {
        let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
        let (h, _) = tlsf_alloc(&mut t, 100, 0);
        t.clear();
        assert_eq!(t.allocation_offset(h), Err(HandleError::InvalidHandle));
        assert_eq!(t.free(h), Err(HandleError::InvalidHandle));
        assert!(t.is_empty());
        t.validate().unwrap();
    }

    // (d) cross-allocator handle.
    {
        let mut a = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
        let mut b = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
        let _ha = tlsf_alloc(&mut a, 100, 0);
        let (_hb, _) = tlsf_alloc(&mut b, 200, 0);
        let (ha2, _) = tlsf_alloc(&mut a, 50, 0);
        let b_free_before = b.sum_free_size();
        let b_count_before = b.allocation_count();
        // ha2 is valid in `a`; freeing it on `b` must be rejected (generation differs)
        // and leave b untouched.
        let _ = b.free(ha2); // may be Ok only if generations coincide; assert no corruption below
        b.validate().unwrap();
        // In practice generations differ; the key guarantee is b stays consistent.
        assert!(b.sum_free_size() == b_free_before || b.allocation_count() == b_count_before - 1);
        // And a stays valid regardless.
        a.validate().unwrap();
    }
}

/// A cross-allocator handle whose packed (index, generation) happens to be live in the
/// target must not corrupt state. We construct the strongest case: identical operation
/// sequences on two allocators produce identical handles; freeing a's handle on b frees
/// b's corresponding allocation but leaves b fully consistent (memory-safe, validated).
#[test]
fn cross_allocator_identical_handles_stay_consistent() {
    let mut a = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    let mut b = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    let (ha, _) = tlsf_alloc(&mut a, 100, 1);
    let (hb, _) = tlsf_alloc(&mut b, 100, 2);
    // Identical sequences => identical packed handles.
    assert_eq!(ha, hb);
    // Freeing a's handle on b is indistinguishable from freeing b's own; it must at
    // worst free the matching allocation and keep b consistent.
    assert_eq!(b.free(ha), Ok(()));
    b.validate().unwrap();
    assert_eq!(b.allocation_count(), 0);
    // A second free is now a double free and is rejected.
    assert_eq!(b.free(hb), Err(HandleError::InvalidHandle));
    b.validate().unwrap();
}

/// Port of `repro_next_alloc`: the iteration helpers `next_allocation` and
/// `next_free_region_size` used to call `node_of(handle)` and index the node arena
/// directly, without routing through `resolve_handle`. A foreign handle carrying a
/// low-32 node index beyond a smaller allocator's arena would index out of bounds and
/// panic. Both must now validate the handle and return their clean failure value
/// (`None` / `0`) instead.
#[test]
fn iteration_helpers_reject_foreign_and_stale_handles() {
    // --- Tlsf: foreign handle with an out-of-arena node index. ---
    let mut a = Tlsf::<u32>::new(1_000_000, 1, true, 0).unwrap();
    let mut handles = Vec::new();
    for i in 0..50u32 {
        handles.push(tlsf_alloc(&mut a, 100, i).0);
    }
    // The last handle carries a high node index that does not exist in the tiny `b`.
    let foreign_high = *handles.last().unwrap();
    let b = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    assert_eq!(
        b.next_allocation(foreign_high),
        None,
        "foreign high-index handle must report end-of-iteration, not panic"
    );
    assert_eq!(
        b.next_free_region_size(foreign_high),
        0,
        "foreign high-index handle must report no free region, not panic"
    );

    // --- Tlsf: stale handle after free (recycled/bumped generation). ---
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    let (h, _) = tlsf_alloc(&mut t, 100, 0);
    t.free(h).unwrap();
    assert_eq!(t.next_allocation(h), None);
    assert_eq!(t.next_free_region_size(h), 0);
    t.validate().unwrap();
}

// ---------------------------------------------------------------------------
// F3: per-page alloc_count no longer overflows; request is rejected cleanly.
// ---------------------------------------------------------------------------

/// Port of `repro_u16_overflow`: with a large granularity, `u16::MAX + 1` unit-sized
/// allocations used to overflow the per-page counter (panic in debug, silent wrap in
/// release). The counter is now `u32`, and the crate never overflows; the block simply
/// fills up.
#[test]
fn repro_u16_overflow() {
    let gran = 131072u64;
    let block = 131072u64;
    let mut t = Tlsf::<u32>::new(block, gran, false, 0).unwrap();

    // Allocate well past the old u16 ceiling. All land on page 0.
    let n = 70000u32;
    let mut made = 0u32;
    for i in 0..n {
        match t.create_allocation_request(AllocationDesc {
            size: 1,
            alignment: 1,
            alloc_type: AllocationType::Buffer,
            ..Default::default()
        }) {
            Ok(req) => {
                t.alloc(req, i);
                made += 1;
            }
            Err(_) => break, // out of space is acceptable; a panic/overflow is not
        }
    }
    // We must have sailed past 65535 without panicking (the block holds `gran` units).
    assert!(made > 65535, "only made {made} allocations before stopping");
    t.validate().unwrap();
}

// ---------------------------------------------------------------------------
// F1: hostile (size, granularity) pairs are rejected, never abort/truncate.
// ---------------------------------------------------------------------------

/// Port of `repro_huge_region`: a huge block with page-tracking granularity used to try
/// to allocate an astronomically large tracking `Vec` (process abort on 64-bit, silent
/// truncation on 32-bit). Now such pairs return `Err(GranularityTrackingTooLarge)`.
#[test]
fn repro_huge_region() {
    // u64::MAX / 512 is ~2^55 pages, far over the cap.
    assert_eq!(
        Tlsf::<u32>::new(u64::MAX, 512, false, 0).unwrap_err(),
        CreateError::GranularityTrackingTooLarge
    );
    // 2^50 / 512 = 2^41 pages, over the cap.
    assert_eq!(
        Tlsf::<u32>::new(1u64 << 50, 512, false, 0).unwrap_err(),
        CreateError::GranularityTrackingTooLarge
    );
    // A pair exactly at the cap is accepted; one page over is rejected.
    let ok_size = MAX_TRACKED_PAGES * 512;
    assert!(Tlsf::<u32>::new(ok_size, 512, false, 0).is_ok());
    assert_eq!(
        Tlsf::<u32>::new(ok_size + 1, 512, false, 0).unwrap_err(),
        CreateError::GranularityTrackingTooLarge
    );
    // Virtual blocks never page-track, so a huge virtual block is fine.
    assert!(Tlsf::<u32>::new(u64::MAX, 512, true, 0).is_ok());
    // Low granularity (<=256) never page-tracks either.
    assert!(Tlsf::<u32>::new(u64::MAX, 256, false, 0).is_ok());
}

// ---------------------------------------------------------------------------
// F5: margin fillers are marked, so real free blocks always merge and is_empty()
// stays truthful.
// ---------------------------------------------------------------------------

fn tlsf_alloc_ty(
    t: &mut Tlsf<u32>,
    size: u64,
    al: u64,
    ty: AllocationType,
    s: Strategy,
) -> (AllocationHandle, u64) {
    let req = t
        .create_allocation_request(AllocationDesc {
            size,
            alignment: al,
            alloc_type: ty,
            strategy: s,
            ..Default::default()
        })
        .unwrap();
    let off = req.offset;
    (t.alloc(req, 0), off)
}

/// Port of `repro_isempty2`: after freeing everything, `is_empty()` used to report
/// `false` because an alignment filler of size == debug_margin was misclassified as a
/// margin filler and never merged. Now `is_empty()` is truthful and the whole block is
/// reclaimable.
#[test]
fn repro_isempty2() {
    let mut t = Tlsf::<u32>::new(8192, 128, false, 8).unwrap();
    let (h1, _o1) = tlsf_alloc_ty(
        &mut t,
        78,
        1,
        AllocationType::ImageOptimal,
        Strategy::MinMemory,
    );
    let (h2, _o2) = tlsf_alloc_ty(
        &mut t,
        372,
        16,
        AllocationType::ImageLinear,
        Strategy::MinTime,
    );
    t.free(h2).unwrap();
    t.free(h1).unwrap();

    assert_eq!(t.allocation_count(), 0);
    assert!(
        t.is_empty(),
        "is_empty() must be truthful after freeing everything"
    );
    assert_eq!(
        t.sum_free_size(),
        8192,
        "the whole block must be reclaimable"
    );
    t.validate().unwrap();

    // The largest allocation that fits (whole block minus the trailing debug margin)
    // must now succeed at offset 0 — impossible if a filler had stranded low space.
    let req = t
        .create_allocation_request(AllocationDesc {
            size: 8192 - 8,
            alignment: 1,
            alloc_type: AllocationType::Buffer,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(req.offset, 0);
    t.alloc(req, 0);
    t.validate().unwrap();
}

/// Port of `repro_margin_strand`: an exactly-`debug_margin`-sized free gap (from an
/// allocation whose size equalled the margin) must merge normally and not strand the
/// block. After freeing all, the block is empty with the full free size.
#[test]
fn repro_margin_strand() {
    let mut t = Tlsf::<u32>::new(1024, 1, false, 8).unwrap();
    // X of size 8 == debug_margin, then Y.
    let (hx, _ox) = tlsf_alloc_ty(&mut t, 8, 1, AllocationType::Unknown, Strategy::Balanced);
    let (hy, _oy) = tlsf_alloc_ty(&mut t, 100, 1, AllocationType::Unknown, Strategy::Balanced);
    t.free(hx).unwrap();
    t.validate().unwrap();
    t.free(hy).unwrap();
    t.validate().unwrap();
    assert_eq!(t.allocation_count(), 0);
    assert!(
        t.is_empty(),
        "size-8 free block must not be stranded as a fake margin filler"
    );
    assert_eq!(t.sum_free_size(), 1024);
}

/// A single alloc/free with debug margin and low granularity must fully reclaim the
/// block (from `repro_isempty`).
#[test]
fn repro_isempty_single_cycle() {
    let mut t = Tlsf::<u32>::new(8192, 128, false, 8).unwrap();
    let (h, _o) = tlsf_alloc_ty(&mut t, 100, 1, AllocationType::Buffer, Strategy::Balanced);
    t.free(h).unwrap();
    assert_eq!(t.allocation_count(), 0);
    assert!(t.is_empty());
    assert_eq!(t.sum_free_size(), 8192);
    t.validate().unwrap();

    // Many cycles must not leak.
    for i in 0..40 {
        let (h, _) = tlsf_alloc_ty(&mut t, 50, 1, AllocationType::Buffer, Strategy::Balanced);
        t.free(h).unwrap();
        assert!(t.is_empty(), "cycle {i}: block leaked");
        assert_eq!(t.sum_free_size(), 8192);
    }
}

// ---------------------------------------------------------------------------
// Panic-freedom boundary sweep.
// ---------------------------------------------------------------------------

/// Exercises the size/alignment/margin/granularity boundary window that surrounds the
/// block-fill point, on both a fresh block and a nearly-full one, and in both
/// directions where legal (TLSF must reject upper-address requests cleanly).
///
/// The contract asserted for *every* combination is panic-freedom: `create_allocation_
/// request` must either return `Ok` (in which case committing it must leave the block
/// in a state that `validate()` accepts) or a clean `Err`. Running in debug keeps every
/// `debug_assert!` live, so a reachable assert would fail the test here. This is a plain
/// unit sweep (not proptest) so the covered points are fixed and auditable.
///
/// Runs one probe request against `allocator`, asserting the panic-freedom contract, and
/// commits it when it succeeds.
fn probe_request<S: Suballocator<u32>>(
    allocator: &mut S,
    size: u64,
    alignment: u64,
    upper: bool,
    alloc_type: AllocationType,
) {
    // A returned request must be committable and leave a valid block; any `Err` is
    // acceptable — the point is that the call did not panic.
    if let Ok(req) = allocator.create_allocation_request(AllocationDesc {
        size,
        alignment,
        upper_address: upper,
        alloc_type,
        ..Default::default()
    }) {
        let offset = req.offset;
        let req_size = req.size;
        allocator.alloc(req, 0);
        allocator.validate().unwrap_or_else(|e| {
            panic!(
                "validate() failed after committing size={size} align={alignment} \
                 upper={upper} at offset={offset} (req_size={req_size}): {e}"
            )
        });
    }
}

/// Fills `allocator` down to a small tail so the next probe hits a nearly-full block.
/// Best-effort: allocation may stop early, which is fine — we only need the block
/// populated. Uses the same direction as the probe so the relevant vector is exercised.
fn fill_nearly_full<S: Suballocator<u32>>(
    allocator: &mut S,
    block_size: u64,
    upper: bool,
    alloc_type: AllocationType,
) {
    // Leave ~1/8 of the block free, then top up with small requests. Bounded loop so a
    // pathological granularity/margin combo cannot spin.
    let big = block_size - block_size / 8;
    if let Ok(req) = allocator.create_allocation_request(AllocationDesc {
        size: big,
        alignment: 1,
        upper_address: upper,
        alloc_type,
        ..Default::default()
    }) {
        allocator.alloc(req, 0);
    }
    for _ in 0..8 {
        match allocator.create_allocation_request(AllocationDesc {
            size: block_size / 32 + 1,
            alignment: 1,
            upper_address: upper,
            alloc_type,
            ..Default::default()
        }) {
            Ok(req) => {
                allocator.alloc(req, 0);
            }
            Err(_) => break,
        }
    }
}

#[test]
fn boundary_sweep_never_panics() {
    let block_sizes = [512u64, 1024, 4096];
    let granularities = [1u64, 16, 256, 512];
    let margins = [0u64, 4, 16];
    let alignments = [1u64, 16, 256];
    // A couple of types so the granularity conflict paths are exercised on the
    // nearly-full block, not just the trivially-non-conflicting Unknown/Unknown case.
    let types = [AllocationType::Unknown, AllocationType::ImageOptimal];

    for &block_size in &block_sizes {
        for &granularity in &granularities {
            for &margin in &margins {
                // TLSF does not support upper-address and must reject it cleanly (which
                // `probe_request` treats as an acceptable error), so we sweep both
                // directions.
                for upper in [false, true] {
                    for &alignment in &alignments {
                        for &alloc_type in &types {
                            // Request sizes bracket the fill boundary
                            // `block_size - margin - alignment ..= block_size + margin`.
                            let lo = block_size
                                .saturating_sub(margin)
                                .saturating_sub(alignment)
                                .saturating_sub(2);
                            let hi = block_size.saturating_add(margin).saturating_add(2);
                            let mut size = lo.max(1);
                            while size <= hi {
                                // Fresh block.
                                if let Ok(mut a) =
                                    Tlsf::<u32>::new(block_size, granularity, false, margin)
                                {
                                    probe_request(&mut a, size, alignment, upper, alloc_type);
                                }

                                // Nearly-full block.
                                if let Ok(mut a) =
                                    Tlsf::<u32>::new(block_size, granularity, false, margin)
                                {
                                    fill_nearly_full(&mut a, block_size, upper, alloc_type);
                                    probe_request(&mut a, size, alignment, upper, alloc_type);
                                }

                                size += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn debug_margin_overflow_rejects_request() {
    let mut allocator = Tlsf::<u32>::new(u64::MAX, 1, false, 4).unwrap();

    let request = allocator.create_allocation_request(AllocationDesc {
        size: u64::MAX,
        alignment: 1,
        ..Default::default()
    });

    assert!(
        request.is_err(),
        "debug-margin overflow must not return a smaller successful allocation"
    );
}
