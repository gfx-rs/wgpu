//! Differential property tests: drive a random operation sequence against both the
//! real allocators and a naive [reference model](crate::reference::ReferenceModel),
//! asserting the safety invariants after *every* operation.
//!
//! Invariants checked after each op:
//! - returned offsets are aligned;
//! - allocations fit within the block;
//! - no two live allocations overlap (checked against the model's interval set);
//! - granularity conflicts are never violated between neighbours on the same page;
//! - `validate()` passes;
//! - `sum_free_size` matches the model.

use alloc::format;
use alloc::vec::Vec;
use proptest::prelude::*;
// Disambiguate proptest's `Strategy` trait from our `Strategy` enum.
use proptest::strategy::Strategy as PropStrategy;

use crate::reference::ReferenceModel;
use crate::{
    Algorithm, AllocationDesc, AllocationHandle, AllocationType, HandleError, Strategy,
    Suballocator, Tlsf, VirtualBlock,
};

/// A single random operation.
#[derive(Clone, Debug)]
enum Op {
    Alloc {
        size: u64,
        align_shift: u32,
        alloc_type: AllocationType,
        strategy: Strategy,
    },
    /// Free the n-th live allocation (index taken mod live-count).
    Free {
        index: usize,
    },
    Clear,
}

fn alloc_type_strategy() -> impl PropStrategy<Value = AllocationType> {
    prop_oneof![
        Just(AllocationType::Unknown),
        Just(AllocationType::Buffer),
        Just(AllocationType::ImageUnknown),
        Just(AllocationType::ImageLinear),
        Just(AllocationType::ImageOptimal),
    ]
}

fn strategy_strategy() -> impl PropStrategy<Value = Strategy> {
    prop_oneof![
        Just(Strategy::Balanced),
        Just(Strategy::MinMemory),
        Just(Strategy::MinTime),
        Just(Strategy::MinOffset),
    ]
}

fn op_strategy(max_size: u64) -> impl PropStrategy<Value = Op> {
    prop_oneof![
        8 => (1..=max_size, 0u32..=8, alloc_type_strategy(), strategy_strategy()).prop_map(
            |(size, align_shift, alloc_type, strategy)| Op::Alloc {
                size,
                align_shift,
                alloc_type,
                strategy,
            }
        ),
        6 => (0usize..64).prop_map(|index| Op::Free { index }),
        1 => Just(Op::Clear),
    ]
}

/// Runs `ops` against a real [`Tlsf`] and the reference model, asserting invariants.
fn run_tlsf(size: u64, granularity: u64, debug_margin: u64, ops: &[Op]) {
    let is_virtual = granularity <= 1;
    let mut t = Tlsf::<u32>::new(size, granularity, is_virtual, debug_margin).unwrap();
    let mut model = ReferenceModel::new(size, if is_virtual { 1 } else { granularity });
    // Live allocations: (handle, offset, size, type).
    let mut live: Vec<(AllocationHandle, u64, u64, AllocationType)> = Vec::new();
    let mut counter: u32 = 0;

    for op in ops {
        match op {
            Op::Alloc {
                size: asize,
                align_shift,
                alloc_type,
                strategy,
            } => {
                let alignment = 1u64 << align_shift;
                let ty = if is_virtual {
                    AllocationType::Unknown
                } else {
                    *alloc_type
                };
                // Failure is always acceptable (the real allocator may be more
                // fragmented than the model), so we only act on success.
                if let Ok(req) = t.create_allocation_request(AllocationDesc {
                    size: *asize,
                    alignment,
                    alloc_type: ty,
                    strategy: *strategy,
                    ..Default::default()
                }) {
                    let offset = req.offset;
                    let committed_size = req.size;
                    // Verify the returned placement against the model's live set.
                    model
                        .check_placement(offset, committed_size, ty)
                        .expect("tlsf returned an invalid placement");
                    assert_eq!(offset % alignment, 0, "offset not aligned");
                    let handle = t.alloc(req, counter);
                    counter += 1;
                    model.insert(offset, committed_size, ty);
                    live.push((handle, offset, committed_size, ty));
                }
            }
            Op::Free { index } => {
                if !live.is_empty() {
                    let i = index % live.len();
                    let (handle, offset, _size, _ty) = live.swap_remove(i);
                    t.free(handle).expect("tlsf free of a live handle failed");
                    model.remove(offset);
                }
            }
            Op::Clear => {
                t.clear();
                model.clear();
                live.clear();
            }
        }

        // Invariants after every op.
        t.validate().expect("tlsf validate failed");
        assert_eq!(
            t.sum_free_size(),
            model.sum_free_size(),
            "sum_free_size mismatch with model"
        );
        assert_eq!(
            t.allocation_count(),
            model.allocation_count(),
            "allocation_count mismatch with model"
        );
        // is_empty() must be truthful: zero live allocations <=> empty, and an empty
        // block must report the whole block free. This catches a stranded margin/
        // alignment filler (F5) that would otherwise leave is_empty() lying.
        assert_eq!(
            t.is_empty(),
            t.allocation_count() == 0,
            "is_empty() disagrees with allocation_count()"
        );
        if t.allocation_count() == 0 {
            assert_eq!(
                t.sum_free_size(),
                size,
                "empty block does not report full free size"
            );
        }
    }
}

/// Runs `ops` against a [`VirtualBlock`] (either algorithm), asserting invariants that
/// hold for both (no granularity here; virtual blocks disable it).
fn run_virtual(size: u64, algorithm: Algorithm, ops: &[Op]) {
    let mut b = VirtualBlock::<u32>::new(size, algorithm).unwrap();
    let mut model = ReferenceModel::new(size, 1);
    let mut live: Vec<(AllocationHandle, u64)> = Vec::new();
    let mut counter: u32 = 0;

    for op in ops {
        match op {
            Op::Alloc {
                size: asize,
                align_shift,
                strategy,
                ..
            } => {
                let alignment = 1u64 << align_shift;
                // Allocation failure is always acceptable (real allocator may be more
                // fragmented than the model), so we only act on success.
                if let Ok((handle, offset)) = b.allocate(
                    AllocationDesc {
                        size: *asize,
                        alignment,
                        strategy: *strategy,
                        ..Default::default()
                    },
                    counter,
                ) {
                    model
                        .check_placement(offset, *asize, AllocationType::Unknown)
                        .expect("virtual block returned an invalid placement");
                    assert_eq!(offset % alignment, 0);
                    counter += 1;
                    model.insert(offset, *asize, AllocationType::Unknown);
                    live.push((handle, offset));
                }
            }
            Op::Free { index } => {
                if !live.is_empty() {
                    let i = index % live.len();
                    let (handle, offset) = live.swap_remove(i);
                    b.free(handle)
                        .expect("virtual free of a live handle failed");
                    model.remove(offset);
                }
            }
            Op::Clear => {
                b.clear();
                model.clear();
                live.clear();
            }
        }

        b.validate().expect("virtual validate failed");
        assert_eq!(b.allocation_count(), model.allocation_count());
    }
}

/// Drives a TLSF allocator and, interspersed, throws invalid handles at it (double
/// free, free-after-clear, cross-allocator), asserting each yields a clean `Err` and
/// leaves `validate()` passing with unchanged accounting (F2).
fn run_tlsf_invalid_handles(size: u64, granularity: u64, debug_margin: u64, ops: &[Op]) {
    let is_virtual = granularity <= 1;
    let mut t = Tlsf::<u32>::new(size, granularity, is_virtual, debug_margin).unwrap();
    // A sibling allocator with an independent operation history, to source foreign
    // handles that are valid in `other` but must be rejected by `t`.
    let mut other = Tlsf::<u32>::new(size, granularity, is_virtual, debug_margin).unwrap();
    let ty = if is_virtual {
        AllocationType::Unknown
    } else {
        AllocationType::Buffer
    };

    let mut live: Vec<AllocationHandle> = Vec::new();
    let mut freed: Vec<AllocationHandle> = Vec::new();
    let mut counter: u32 = 0;

    // Seed `other` with a couple of allocations for cross-allocator handles.
    for _ in 0..3 {
        if let Ok(req) = other.create_allocation_request(AllocationDesc {
            size: 64,
            alignment: 1,
            alloc_type: ty,
            ..Default::default()
        }) {
            other.alloc(req, 0);
        }
    }

    let assert_untouched = |t: &Tlsf<u32>, free_before: u64, count_before: usize| {
        assert_eq!(
            t.sum_free_size(),
            free_before,
            "invalid op changed sum_free_size"
        );
        assert_eq!(
            t.allocation_count(),
            count_before,
            "invalid op changed allocation_count"
        );
        t.validate().expect("validate failed after an invalid op");
    };

    for op in ops {
        match op {
            Op::Alloc {
                size: asize,
                align_shift,
                ..
            } => {
                let alignment = 1u64 << align_shift;
                if let Ok(req) = t.create_allocation_request(AllocationDesc {
                    size: *asize,
                    alignment,
                    alloc_type: ty,
                    ..Default::default()
                }) {
                    live.push(t.alloc(req, counter));
                    counter += 1;
                }
            }
            Op::Free { index } => {
                if !live.is_empty() {
                    let i = index % live.len();
                    let h = live.swap_remove(i);
                    t.free(h).expect("free of a live handle failed");
                    freed.push(h);
                }
            }
            Op::Clear => {
                t.clear();
                // Every previously-live handle is now stale.
                freed.append(&mut live);
            }
        }

        // Double-free / stale-handle probe: any freed handle must be rejected.
        if let Some(&stale) = freed.last() {
            let free_before = t.sum_free_size();
            let count_before = t.allocation_count();
            assert_eq!(t.free(stale), Err(HandleError::InvalidHandle));
            assert_eq!(t.allocation_offset(stale), Err(HandleError::InvalidHandle));
            assert_eq!(t.set_user_data(stale, 7), Err(HandleError::InvalidHandle));
            assert_untouched(&t, free_before, count_before);
        }

        // Cross-allocator probe: a handle valid in `other` must not free anything in `t`
        // unless it coincidentally names a live (index, generation) pair — in which case
        // t must still stay consistent. We only assert consistency, never a specific
        // outcome, to avoid depending on generation coincidence.
        if let Some(foreign) = other.allocation_list_begin() {
            let free_before = t.sum_free_size();
            let count_before = t.allocation_count();
            match t.free(foreign) {
                Err(HandleError::InvalidHandle) => assert_untouched(&t, free_before, count_before),
                Ok(()) => {
                    // Coincidental match: t freed a real allocation. Must remain valid.
                    t.validate()
                        .expect("validate failed after coincidental cross-free");
                    // Remove any of our tracked handles that named that node, to keep
                    // bookkeeping honest for later probes.
                    live.retain(|&h| t.allocation_offset(h).is_ok());
                }
            }
        }
    }
}

proptest! {
    #[test]
    fn tlsf_virtual_differential(ops in prop::collection::vec(op_strategy(4096), 0..300)) {
        run_tlsf(4096, 1, 0, &ops);
    }

    #[test]
    fn tlsf_granularity_low_differential(ops in prop::collection::vec(op_strategy(4096), 0..300)) {
        // Granularity 128 (<= 256): the "round up" path.
        run_tlsf(1 << 16, 128, 0, &ops);
    }

    #[test]
    fn tlsf_granularity_page_differential(ops in prop::collection::vec(op_strategy(4096), 0..300)) {
        // Granularity 1024 (> 256): the page-tracking path.
        run_tlsf(1 << 16, 1024, 0, &ops);
    }

    #[test]
    fn tlsf_granularity_large_page_differential(ops in prop::collection::vec(op_strategy(1 << 15), 0..300)) {
        // Granularity 512 (> 256) with a larger block: more pages per allocation.
        run_tlsf(1 << 18, 512, 0, &ops);
    }

    #[test]
    fn tlsf_debug_margin_differential(ops in prop::collection::vec(op_strategy(4096), 0..300)) {
        // Non-virtual with a debug margin: exercises the margin-filler marker (F5). The
        // reference model already accounts for margins correctly, because the real
        // allocator reports the margin space as free (it merges away on free) and never
        // overlaps usable ranges.
        run_tlsf(1 << 16, 1, 8, &ops);
    }

    #[test]
    fn tlsf_gran_and_margin_differential(ops in prop::collection::vec(op_strategy(4096), 0..300)) {
        // Page tracking + debug margin together.
        run_tlsf(1 << 16, 512, 16, &ops);
    }

    #[test]
    fn tlsf_invalid_handle_ops(
        ops in prop::collection::vec(op_strategy(4096), 0..200),
        margin in prop::sample::select(&[0u64, 8][..]),
        gran in prop::sample::select(&[1u64, 512][..]),
    ) {
        run_tlsf_invalid_handles(1 << 16, gran, margin, &ops);
    }

    #[test]
    fn virtualblock_tlsf_differential(ops in prop::collection::vec(op_strategy(4096), 0..300)) {
        run_virtual(4096, Algorithm::Tlsf, &ops);
    }

    #[test]
    fn tlsf_construction_never_aborts(
        size in any::<u64>(),
        gran_shift in 0u32..=40,
    ) {
        // A hostile (size, granularity) pair must return cleanly (Ok or a CreateError),
        // never abort or panic (F1). `gran = 1 << gran_shift` is always a valid power of
        // two (or 1). We only require that the call returns.
        let gran = 1u64 << gran_shift;
        let size = size.max(1);
        let _ = Tlsf::<u32>::new(size, gran, false, 0);
        // Virtual blocks never page-track, so they must always succeed for non-zero size.
        prop_assert!(Tlsf::<u32>::new(size, gran, true, 0).is_ok());
    }
}
