//! Differential property tests: drive a random operation sequence against a [`Pool`]
//! plus an oracle, asserting the safety invariants after *every* operation.
//!
//! Invariants checked after each op:
//! - `pool.validate()` passes (every block valid, sorted-ish, hysteresis holds);
//! - no two live allocations overlap within any single block;
//! - block count is within `[min_block_count, max_block_count]`;
//! - at most one empty block above `min_block_count`;
//! - aggregate statistics match a naive recount of the live set.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use proptest::prelude::*;
use proptest::strategy::Strategy as PropStrategy;

use crate::tests::mock::{MockBackend, MockBlock};
use crate::{
    Algorithm, Allocation, AllocationContext, AllocationDesc, AllocationType, BlockId, FreeContext,
    Pool, PoolConfig, Strategy,
};

#[derive(Clone, Debug)]
enum Op {
    Alloc {
        size: u64,
        align_shift: u32,
        alloc_type: AllocationType,
        strategy: Strategy,
        budget_free: Option<u64>,
        dedicated_fallback: bool,
        preferred_affinity: Option<bool>,
    },
    /// Free the n-th live allocation (index mod live-count).
    Free { index: usize, budget_exceeded: bool },
    /// Set the affinity of the n-th block (index mod block-count).
    SetAffinity { index: usize, affinity: bool },
}

fn alloc_type_strategy() -> impl PropStrategy<Value = AllocationType> {
    prop_oneof![
        Just(AllocationType::Unknown),
        Just(AllocationType::Buffer),
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
        8 => (
            1..=max_size,
            0u32..=6,
            alloc_type_strategy(),
            strategy_strategy(),
            prop::option::of(0u64..=(max_size * 4)),
            any::<bool>(),
            prop::option::of(any::<bool>()),
        )
            .prop_map(|(size, align_shift, alloc_type, strategy, budget_free, dedicated_fallback, preferred_affinity)| {
                Op::Alloc {
                    size,
                    align_shift,
                    alloc_type,
                    strategy,
                    budget_free,
                    dedicated_fallback,
                    preferred_affinity,
                }
            }),
        6 => (0usize..64, any::<bool>()).prop_map(|(index, budget_exceeded)| Op::Free {
            index,
            budget_exceeded,
        }),
        2 => (0usize..8, any::<bool>()).prop_map(|(index, affinity)| Op::SetAffinity { index, affinity }),
    ]
}

/// The oracle: tracks live allocations grouped by block, for overlap and stats checks.
struct Oracle {
    /// block_id -> list of (offset, size) live intervals. `BlockId` is an opaque `Copy`
    /// key with `Ord`, so it keys the map directly.
    by_block: BTreeMap<BlockId, Vec<(u64, u64)>>,
}

impl Oracle {
    fn new() -> Self {
        Oracle {
            by_block: BTreeMap::new(),
        }
    }

    fn insert(&mut self, a: &Allocation) {
        self.by_block
            .entry(a.block_id())
            .or_default()
            .push((a.offset(), a.size()));
    }

    fn remove(&mut self, a: &Allocation) {
        if let Some(v) = self.by_block.get_mut(&a.block_id()) {
            if let Some(pos) = v
                .iter()
                .position(|&(o, s)| o == a.offset() && s == a.size())
            {
                v.swap_remove(pos);
            }
        }
    }

    /// Assert no two live intervals overlap within a block, and none exceeds the block.
    fn check_no_overlap(&self) {
        for intervals in self.by_block.values() {
            let mut sorted = intervals.clone();
            sorted.sort_unstable();
            for w in sorted.windows(2) {
                let (o0, s0) = w[0];
                let (o1, _s1) = w[1];
                let end0 = o0.checked_add(s0).expect("interval overflow");
                assert!(end0 <= o1, "overlapping live allocations within a block");
            }
        }
    }

    fn total_allocations(&self) -> u64 {
        self.by_block.values().map(|v| v.len() as u64).sum()
    }

    fn total_bytes(&self) -> u64 {
        self.by_block
            .values()
            .flat_map(|v| v.iter().map(|&(_, s)| s))
            .sum()
    }
}

/// Runs `ops` against a `Pool` configured by the given parameters, plus the oracle.
#[allow(clippy::too_many_arguments)]
fn run(
    algorithm: Algorithm,
    preferred: u64,
    min_blocks: usize,
    max_blocks: usize,
    explicit: bool,
    granularity: u64,
    debug_margin: u64,
    affinity_clustering: bool,
    min_alloc_align: u64,
    byte_cap: Option<u64>,
    ops: &[Op],
) {
    let config = PoolConfig {
        algorithm,
        preferred_block_size: preferred,
        min_block_count: min_blocks,
        max_block_count: max_blocks,
        explicit_block_size: explicit,
        min_allocation_alignment: min_alloc_align,
        granularity,
        debug_margin,
        affinity_clustering,
        pool_salt: 0,
    };

    let mut backend = match byte_cap {
        Some(cap) => MockBackend::with_byte_cap(cap),
        None => MockBackend::new(),
    };
    let mut pool = match Pool::<MockBackend, u32>::new(config, &mut backend) {
        Ok(p) => p,
        // A byte cap can starve the eager min-block creation; that's a valid failure,
        // just skip this case.
        Err(_) => return,
    };

    let mut oracle = Oracle::new();
    let mut live: Vec<Allocation> = Vec::new();
    let mut counter: u32 = 0;
    // Total successful allocations over the whole run, for the churn oracle.
    let mut total_successful_allocs: u64 = 0;

    for op in ops {
        match op {
            Op::Alloc {
                size,
                align_shift,
                alloc_type,
                strategy,
                budget_free,
                dedicated_fallback,
                preferred_affinity,
            } => {
                let alignment = 1u64 << align_shift;
                let ty = if granularity <= 1 {
                    AllocationType::Unknown
                } else {
                    *alloc_type
                };
                let ctx = AllocationContext {
                    budget_free_bytes: *budget_free,
                    dedicated_fallback_allowed: *dedicated_fallback,
                    preferred_affinity: *preferred_affinity,
                };
                // Failure of any kind is acceptable; we only act on success.
                if let Ok(a) = pool.allocate(
                    AllocationDesc {
                        size: *size,
                        alignment,
                        alloc_type: ty,
                        strategy: *strategy,
                        upper_address: false,
                    },
                    ctx,
                    counter,
                    &mut backend,
                ) {
                    // Placement sanity.
                    let eff_align = alignment.max(min_alloc_align).max(1);
                    assert_eq!(a.offset() % eff_align, 0, "returned offset not aligned");
                    assert!(a.size() >= *size, "committed size below requested");
                    counter += 1;
                    total_successful_allocs += 1;
                    oracle.insert(&a);
                    live.push(a);
                }
            }
            Op::Free {
                index,
                budget_exceeded,
            } => {
                if !live.is_empty() {
                    let i = index % live.len();
                    let a = live.swap_remove(i);
                    let outcome = pool
                        .free(
                            a,
                            FreeContext {
                                budget_exceeded: *budget_exceeded,
                            },
                        )
                        .expect("free of a live allocation failed");
                    oracle.remove(&a);
                    // If a block was destroyed, drop any live allocations that belonged
                    // to it from the oracle bookkeeping. (There should be none — an
                    // empty block has no live allocations — but guard anyway.)
                    if let Some((block, id)) = outcome.destroyed_block {
                        release(&mut backend, block, id);
                        // Empty block: no live allocations should remain for it.
                        assert!(
                            !live.iter().any(|l| l.block_id() == id),
                            "destroyed a block that still had live allocations"
                        );
                        oracle.by_block.remove(&id);
                    }
                }
            }
            Op::SetAffinity { index, affinity } => {
                let report = pool.report();
                if !report.blocks.is_empty() {
                    let i = index % report.blocks.len();
                    let id = report.blocks[i].block_id;
                    pool.set_block_affinity(id, *affinity);
                }
            }
        }

        // Invariants after every op.
        pool.validate().expect("pool.validate() failed");
        oracle.check_no_overlap();

        assert!(pool.block_count() <= max_blocks, "block count exceeds max");
        assert!(
            pool.block_count() >= min_blocks,
            "block count fell below min_block_count"
        );

        // At most one empty block above min_block_count.
        if pool.block_count() > min_blocks {
            assert!(
                pool.empty_block_count() <= 1,
                "more than one empty block above min_block_count"
            );
        }

        // Aggregate statistics match the oracle recount.
        let stats = pool.statistics();
        assert_eq!(
            u64::from(stats.allocation_count),
            oracle.total_allocations(),
            "allocation_count mismatch with oracle"
        );
        assert_eq!(
            stats.allocation_bytes,
            oracle.total_bytes(),
            "allocation_bytes mismatch with oracle"
        );
        assert_eq!(
            usize::try_from(stats.block_count).unwrap(),
            pool.block_count(),
            "block_count stat mismatch"
        );

        // Fix 2 (no churn): every block the pool holds is a live backend block, and the
        // pool never creates-then-destroys a block inside `allocate`. The mock's live
        // block count (creates - destroys, where destroys are only the test-driven
        // FreeOutcome/drop releases) must therefore equal the pool's block count at all
        // times. A grow-rollback churn would momentarily destroy a block the pool never
        // told the test about, breaking this equality.
        assert_eq!(
            backend.live_blocks,
            pool.block_count(),
            "backend live blocks != pool block count (grow churn or leak)",
        );
    }

    // Churn oracle: over the whole run, backend creates minus destroys equals the blocks
    // still live (creates - destroys == live_blocks), and creates never exceed the number
    // of successful allocations plus the retained empties (at most one empty above
    // min_block_count, plus the eagerly created min blocks). This bounds block creation to
    // real demand — it cannot scale with failing calls (the old churn).
    let creates = backend.create_count() as u64;
    let destroys = backend.destroy_count() as u64;
    assert_eq!(
        creates - destroys,
        backend.live_blocks as u64,
        "creates - destroys != live blocks",
    );
    let retained_empties = 1 + min_blocks as u64;
    assert!(
        creates <= total_successful_allocs + retained_empties,
        "creates ({creates}) exceed successful allocs ({total_successful_allocs}) + retained empties ({retained_empties})",
    );

    // Drain remaining blocks (drop path) and confirm nothing leaks in the mock.
    for (block, id) in pool.into_blocks() {
        release(&mut backend, block, id);
    }
    assert_eq!(backend.live_blocks, 0, "device memory leaked on drop");
}

fn release(backend: &mut MockBackend, block: MockBlock, id: BlockId) {
    <MockBackend as crate::BlockBackend>::destroy_block(backend, block, id);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn tlsf_pool_invariants(ops in prop::collection::vec(op_strategy(4096), 0..200)) {
        run(Algorithm::Tlsf, 8192, 0, 4, false, 1, 0, false, 1, None, &ops);
    }

    #[test]
    fn tlsf_pool_with_margin(ops in prop::collection::vec(op_strategy(2048), 0..200)) {
        run(Algorithm::Tlsf, 8192, 0, 4, false, 1, 16, false, 1, None, &ops);
    }

    #[test]
    fn tlsf_pool_with_granularity(ops in prop::collection::vec(op_strategy(2048), 0..200)) {
        run(Algorithm::Tlsf, 16384, 0, 4, false, 512, 0, false, 1, None, &ops);
    }

    #[test]
    fn tlsf_pool_small_max_count(ops in prop::collection::vec(op_strategy(4096), 0..200)) {
        run(Algorithm::Tlsf, 8192, 1, 2, true, 1, 0, false, 256, None, &ops);
    }

    #[test]
    fn tlsf_pool_affinity_clustering(ops in prop::collection::vec(op_strategy(1024), 0..200)) {
        run(Algorithm::Tlsf, 8192, 0, 4, false, 1, 0, true, 1, None, &ops);
    }

    #[test]
    fn tlsf_pool_byte_cap_backend(ops in prop::collection::vec(op_strategy(2048), 0..200)) {
        // A byte-capped backend forces create failures and the halving retry.
        run(Algorithm::Tlsf, 8192, 0, 8, false, 1, 0, false, 1, Some(16384), &ops);
    }

    // Fix 2: preferred_block_size near/below granularity, ramp enabled. This is the
    // configuration that produced the grow-churn DoS: the ramp would floor a new block
    // below what granularity rounding needs. The churn oracle in `run` (creates bounded
    // by successful allocs + retained empties, and backend.live_blocks == block_count
    // after every op) proves the churn is gone. preferred=64 with granularity 64 means
    // the ramp cannot step below a full block that can still place a rounded request.
    #[test]
    fn tlsf_pool_churn_granularity_ge_preferred(ops in prop::collection::vec(op_strategy(48), 0..200)) {
        run(Algorithm::Tlsf, 64, 0, 4, false, 64, 0, false, 1, None, &ops);
    }

    // Fix 2, low granularity with a ramp and a margin, small preferred: exercises
    // align_up rounding + trailing margin in the footprint against the ramp floor.
    #[test]
    fn tlsf_pool_churn_low_granularity_margin(ops in prop::collection::vec(op_strategy(96), 0..200)) {
        run(Algorithm::Tlsf, 128, 0, 4, false, 32, 16, false, 1, None, &ops);
    }

    // Ramp + hysteresis interplay with min_block_count > 0 AND explicit_block_size =
    // false (previously untested combination): the eager min blocks are full-size
    // preferred, but subsequent growth uses the 1/8..1/2 ramp, and the hysteresis must
    // never drop below min_block_count while the ramp adds/removes blocks above it.
    #[test]
    fn tlsf_pool_min_blocks_with_ramp(ops in prop::collection::vec(op_strategy(4096), 0..200)) {
        run(Algorithm::Tlsf, 8192, 2, 5, false, 1, 0, false, 1, None, &ops);
    }
}
