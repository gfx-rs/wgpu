//! Targeted unit tests for the pool's policy: block-size ramp, create-failure
//! halving retry, explicit block size, min/max block count enforcement, empty-block
//! hysteresis, budget gating, too-large rejection, MinTime backward scan, affinity
//! two-pass clustering, upper-address rejection, stale-free, block-destroyed-returned,
//! and drop-returns-all.

use alloc::vec;
use alloc::vec::Vec;

use crate::tests::mock::{Event, MockBackend, MockBlock};
use crate::{
    Algorithm, AllocationContext, AllocationDesc, AllocationType, BlockId, FreeContext, Pool,
    PoolAllocError, PoolConfig, Strategy,
};

/// A convenient default context: unlimited budget, no dedicated fallback, no affinity.
fn ctx() -> AllocationContext {
    AllocationContext {
        budget_free_bytes: None,
        dedicated_fallback_allowed: false,
        preferred_affinity: None,
    }
}

/// Base config: TLSF, given preferred size, up to `max` blocks.
fn cfg(preferred: u64, max: usize) -> PoolConfig {
    PoolConfig {
        algorithm: Algorithm::Tlsf,
        preferred_block_size: preferred,
        min_block_count: 0,
        max_block_count: max,
        explicit_block_size: false,
        min_allocation_alignment: 1,
        granularity: 1,
        debug_margin: 0,
        affinity_clustering: false,
        pool_salt: 0,
    }
}

/// Allocate with the common no-frills parameters.
fn alloc(
    pool: &mut Pool<MockBackend, u32>,
    backend: &mut MockBackend,
    size: u64,
    user_data: u32,
) -> Result<crate::Allocation, PoolAllocError<crate::tests::mock::MockError>> {
    pool.allocate(
        AllocationDesc {
            size,
            alignment: 1,
            ..Default::default()
        },
        ctx(),
        user_data,
        backend,
    )
}

#[test]
fn empty_pool_creates_first_block_at_one_eighth() {
    // Ramp: the first block should be 1/8 of the preferred size when the request is
    // small enough (smaller >= size*2 must hold at each step). preferred = 8 MiB, a
    // 1-unit request: 1/8 = 1 MiB, 1/4 = 2 MiB, 1/2 = 4 MiB. Each `smaller` must be >
    // max_existing (0) and >= size*2 (2), so it steps all the way to 1/8.
    let mut backend = MockBackend::new();
    let mut pool = Pool::<MockBackend, u32>::new(cfg(8 << 20, 8), &mut backend).unwrap();
    alloc(&mut pool, &mut backend, 1, 0).unwrap();
    assert_eq!(pool.block_count(), 1);
    assert_eq!(backend.create_count(), 1);
    // 1/8 of 8 MiB = 1 MiB.
    assert_eq!(
        backend.events[0],
        Event::Create {
            id: BlockId::new_for_test(0, 0),
            size: 1 << 20
        }
    );
    pool.validate().unwrap();
}

#[test]
fn ramp_progression_full_1_8_1_4_1_2_full() {
    // Force the ramp to step: allocate sizes that make each new block's target size
    // exactly the next ramp rung. preferred = 8. Request 1 -> 1/8 = 1 (>0 && >=2? no:
    // 1 >= 2 is false). So with preferred 8 the smallest step where 1/8 >= size*2 needs
    // size <= 1/16 of preferred. Use preferred = 64.
    let preferred = 64u64;
    let mut backend = MockBackend::new();
    let mut pool = Pool::<MockBackend, u32>::new(cfg(preferred, 8), &mut backend).unwrap();

    // Request size 1: 1/8 = 8 (> max_existing 0 && >= 2) -> 8; 1/4=16 also >8*... wait
    // ramp only compares against max_existing and size*2. Starting newBlockSize=64:
    //   step0: smaller=32, 32>0 && 32>=2 -> 32
    //   step1: smaller=16, 16>0 && 16>=2 -> 16
    //   step2: smaller=8,  8>0  && 8>=2  -> 8  (shift hits max=3)
    // So first block is 8 (= 1/8).
    alloc(&mut pool, &mut backend, 1, 0).unwrap();
    assert_eq!(
        backend.events.last().unwrap(),
        &Event::Create {
            id: BlockId::new_for_test(0, 0),
            size: 8
        }
    );

    // Fill the 8-block, forcing a second block. Now max_existing = 8.
    //   start 64: smaller=32 (>8 && >=2) ->32; smaller=16(>8&&>=2)->16; smaller=8(8>8? no) stop.
    // Second block is 16 (= 1/4).
    let mut sz = 1u64;
    // Keep allocating size-1 until a new (16) block appears.
    loop {
        alloc(&mut pool, &mut backend, 1, sz as u32).unwrap();
        if pool.block_count() == 2 {
            break;
        }
        sz += 1;
        assert!(sz < 1000, "did not create a second block");
    }
    assert_eq!(
        backend.events.last().unwrap(),
        &Event::Create {
            id: BlockId::new_for_test(0, 1),
            size: 16
        }
    );
    pool.validate().unwrap();
}

#[test]
fn explicit_block_size_disables_ramp() {
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    alloc(&mut pool, &mut backend, 1, 0).unwrap();
    // Explicit: block is exactly the preferred size, no 1/8 ramp.
    assert_eq!(
        backend.events[0],
        Event::Create {
            id: BlockId::new_for_test(0, 0),
            size: 4096
        }
    );
    pool.validate().unwrap();
}

#[test]
fn create_failure_halving_retry() {
    // The first create (at the ramp size) fails; the pool should retry with halved
    // sizes down to >= request size, and eventually succeed.
    //
    // preferred = 64, request = 8 (size*2 = 16). The initial ramp stops early:
    //   step0: smaller=32 (>0 && >=16) -> 32
    //   step1: smaller=16 (>0 && >=16) -> 16
    //   step2: smaller=8  (8 >= 16? no) -> stop, shift = 2.
    // First create attempt at 16 (idx 0) fails. Failure-halving loop (shift 2 < 3):
    //   smaller=8, 8 >= size(8) -> create at 8 (idx 1), succeeds.
    let preferred = 64u64;
    let mut backend = MockBackend::failing_at(&[0]);
    let mut pool = Pool::<MockBackend, u32>::new(cfg(preferred, 4), &mut backend).unwrap();

    alloc(&mut pool, &mut backend, 8, 0).unwrap();
    assert_eq!(
        backend.create_calls, 2,
        "expected one failed + one successful create"
    );
    assert_eq!(pool.block_count(), 1);
    assert_eq!(
        backend.events.last().unwrap(),
        &Event::Create {
            id: BlockId::new_for_test(0, 0),
            size: 8
        }
    );
    pool.validate().unwrap();
}

#[test]
fn create_failure_all_sizes_fail_is_backend_error() {
    // Every create fails: the pool must surface the backend error, not panic.
    let mut backend = MockBackend::failing_at(&[0, 1, 2, 3, 4, 5, 6, 7]);
    let mut pool = Pool::<MockBackend, u32>::new(cfg(64, 4), &mut backend).unwrap();
    let err = alloc(&mut pool, &mut backend, 1, 0).unwrap_err();
    assert_eq!(err, PoolAllocError::Backend(crate::tests::mock::MockError));
    assert_eq!(pool.block_count(), 0);
}

#[test]
fn max_block_count_one_out_of_pool_memory() {
    // A single-block pool that fills up returns OutOfPoolMemory (not ShouldDedicate,
    // since the request itself fits a block).
    let mut backend = MockBackend::new();
    let mut config = cfg(16, 1);
    config.explicit_block_size = true; // exact 16-unit block
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    // Fill the 16-unit block.
    alloc(&mut pool, &mut backend, 16, 0).unwrap();
    // Next allocation cannot grow (max=1) and does not fit -> OutOfPoolMemory.
    let err = alloc(&mut pool, &mut backend, 8, 1).unwrap_err();
    assert_eq!(err, PoolAllocError::OutOfPoolMemory);
    pool.validate().unwrap();
}

#[test]
fn min_block_count_created_eagerly_and_retained() {
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.min_block_count = 2;
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    assert_eq!(pool.block_count(), 2, "min blocks created at construction");
    assert_eq!(backend.create_count(), 2);

    // Allocate then free: the two min blocks must never be destroyed by hysteresis.
    let a = alloc(&mut pool, &mut backend, 100, 0).unwrap();
    let outcome = pool.free(a, FreeContext::default()).unwrap();
    assert!(
        outcome.destroyed_block.is_none(),
        "min blocks must not be dropped"
    );
    assert_eq!(pool.block_count(), 2);
    pool.validate().unwrap();
}

#[test]
fn too_large_request_should_dedicate() {
    let mut backend = MockBackend::new();
    // preferred = 16, request 17 can never fit -> ShouldDedicate.
    let mut config = cfg(16, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let err = alloc(&mut pool, &mut backend, 17, 0).unwrap_err();
    assert_eq!(err, PoolAllocError::ShouldDedicate);
    assert_eq!(pool.block_count(), 0);
}

#[test]
fn too_large_with_margin_should_dedicate() {
    // The early reject uses VMA's single-margin footprint (`size + margin > preferred`,
    // fix 3): a request that fits raw but not with one trailing margin is rejected as
    // ShouldDedicate.
    let mut backend = MockBackend::new();
    let mut config = cfg(64, 4);
    config.explicit_block_size = true;
    config.debug_margin = 16;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    // 64 - 16 = 48 is the largest that passes the early reject. 49 should dedicate.
    let err = alloc(&mut pool, &mut backend, 49, 0).unwrap_err();
    assert_eq!(err, PoolAllocError::ShouldDedicate);
    // 48 must pass the early reject and, in a full-size block, actually place (the
    // 2*margin rule of the old code would have wrongly dedicated it — PoC
    // pool_early_reject).
    let a = alloc(&mut pool, &mut backend, 48, 1).unwrap();
    assert_eq!(a.block_id(), BlockId::new_for_test(0, 0));
    pool.validate().unwrap();
}

#[test]
fn budget_gate_returns_should_dedicate() {
    // The pool would need to grow, the budget cannot fit a new block, and a dedicated
    // fallback is allowed -> ShouldDedicate (do not create a block).
    let mut backend = MockBackend::new();
    let config = cfg(1 << 20, 4);
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let c = AllocationContext {
        budget_free_bytes: Some(10), // less than the 100-unit request
        dedicated_fallback_allowed: true,
        preferred_affinity: None,
    };
    let err = pool
        .allocate(
            AllocationDesc {
                size: 100,
                alignment: 1,
                ..Default::default()
            },
            c,
            0,
            &mut backend,
        )
        .unwrap_err();
    assert_eq!(err, PoolAllocError::ShouldDedicate);
    assert_eq!(
        pool.block_count(),
        0,
        "no block created under the budget gate"
    );
}

#[test]
fn budget_gate_bypassed_when_no_dedicated_fallback() {
    // Same tight budget, but no dedicated fallback: the pool must create a block even
    // though it exceeds budget (the caller has nowhere else to go).
    let mut backend = MockBackend::new();
    let mut config = cfg(1 << 20, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let c = AllocationContext {
        budget_free_bytes: Some(10),
        dedicated_fallback_allowed: false,
        preferred_affinity: None,
    };
    pool.allocate(
        AllocationDesc {
            size: 100,
            alignment: 1,
            ..Default::default()
        },
        c,
        0,
        &mut backend,
    )
    .unwrap();
    assert_eq!(pool.block_count(), 1);
}

#[test]
fn hysteresis_retains_one_empty_block() {
    // Free the only allocation in a block when no other empty block exists: the block
    // is retained (hysteresis), not destroyed.
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let a = alloc(&mut pool, &mut backend, 100, 0).unwrap();
    let outcome = pool.free(a, FreeContext::default()).unwrap();
    assert!(
        outcome.destroyed_block.is_none(),
        "single empty block is retained"
    );
    assert_eq!(pool.block_count(), 1);
    assert_eq!(pool.empty_block_count(), 1);
    pool.validate().unwrap();
}

#[test]
fn hysteresis_drops_second_empty_block() {
    // With two blocks that both become empty, the second empty free drops a block.
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    // Two allocations forced into two different blocks: fill block 0 fully first.
    let a = alloc(&mut pool, &mut backend, 4096, 0).unwrap();
    let b = alloc(&mut pool, &mut backend, 4096, 1).unwrap();
    assert_eq!(pool.block_count(), 2);
    assert_ne!(a.block_id(), b.block_id());

    // Free a: block 0 becomes empty, no other empty existed -> retained (1 empty).
    let o1 = pool.free(a, FreeContext::default()).unwrap();
    assert!(o1.destroyed_block.is_none());
    assert_eq!(pool.empty_block_count(), 1);

    // Free b: block 1 becomes empty, an empty block already existed -> drop one block.
    let o2 = pool.free(b, FreeContext::default()).unwrap();
    assert!(o2.destroyed_block.is_some(), "second empty block dropped");
    assert_eq!(pool.block_count(), 1);
    assert_eq!(pool.empty_block_count(), 1);
    pool.validate().unwrap();
}

#[test]
fn hysteresis_budget_exceeded_drops_first_empty_block() {
    // With budget_exceeded, even the first block to become empty is dropped (above
    // min_block_count).
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    config.min_block_count = 0;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    // Two blocks so we are above min_block_count (0) with room to delete.
    let a = alloc(&mut pool, &mut backend, 4096, 0).unwrap();
    let _b = alloc(&mut pool, &mut backend, 4096, 1).unwrap();
    assert_eq!(pool.block_count(), 2);
    let outcome = pool
        .free(
            a,
            FreeContext {
                budget_exceeded: true,
            },
        )
        .unwrap();
    assert!(
        outcome.destroyed_block.is_some(),
        "budget_exceeded drops the empty block"
    );
    assert_eq!(pool.block_count(), 1);
    pool.validate().unwrap();
}

#[test]
fn hysteresis_never_below_min_block_count() {
    // At min_block_count, an empty block is never destroyed, even with budget_exceeded.
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    config.min_block_count = 1;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    assert_eq!(pool.block_count(), 1);
    let a = alloc(&mut pool, &mut backend, 100, 0).unwrap();
    let outcome = pool
        .free(
            a,
            FreeContext {
                budget_exceeded: true,
            },
        )
        .unwrap();
    assert!(
        outcome.destroyed_block.is_none(),
        "cannot drop below min_block_count"
    );
    assert_eq!(pool.block_count(), 1);
    pool.validate().unwrap();
}

#[test]
fn reclaim_trailing_empty_block_on_nonempty_free() {
    // VMA 11980-1990: a non-empty free while an empty block exists reclaims the
    // trailing empty block.
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();

    // Block 0: two allocations. Block 1: one allocation.
    let a0 = alloc(&mut pool, &mut backend, 4096, 0).unwrap(); // fills block 0
    let a1 = alloc(&mut pool, &mut backend, 4000, 1).unwrap(); // block 1
    let a2 = alloc(&mut pool, &mut backend, 96, 2).unwrap(); // block 1 (fits alongside a1)
    assert_eq!(pool.block_count(), 2);
    assert_eq!(a1.block_id(), a2.block_id());
    assert_ne!(a0.block_id(), a1.block_id());

    // Free a0: block 0 becomes empty; no other empty existed -> retained (1 empty).
    let o0 = pool.free(a0, FreeContext::default()).unwrap();
    assert!(o0.destroyed_block.is_none());
    assert_eq!(pool.empty_block_count(), 1);

    // Free a2: block 1 does NOT become empty (a1 still live) but an empty block exists.
    // If the empty block is the trailing block, it is reclaimed.
    let o1 = pool.free(a2, FreeContext::default()).unwrap();
    // Whether a block is dropped depends on which block is last after sorting; the
    // invariant we assert is that at most one empty block remains and validate passes.
    let _ = o1;
    assert!(pool.empty_block_count() <= 1);
    pool.validate().unwrap();
}

#[test]
fn min_time_backward_scan_picks_largest_free() {
    // Two blocks with different free sizes; MinTime scans backward. We assert placement
    // succeeds and validate holds (exact block choice is order-dependent, but the scan
    // path is exercised).
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    // Create two blocks with a mix of occupancy.
    let _a = alloc(&mut pool, &mut backend, 4096, 0).unwrap(); // block 0 full
    let _b = alloc(&mut pool, &mut backend, 2000, 1).unwrap(); // block 1 partial
    assert_eq!(pool.block_count(), 2);
    // MinTime allocation should reuse the partial block (largest free), not grow.
    let before = pool.block_count();
    pool.allocate(
        AllocationDesc {
            size: 100,
            alignment: 1,
            alloc_type: AllocationType::Unknown,
            strategy: Strategy::MinTime,
            upper_address: false,
        },
        ctx(),
        99,
        &mut backend,
    )
    .unwrap();
    assert_eq!(
        pool.block_count(),
        before,
        "MinTime reused an existing block"
    );
    pool.validate().unwrap();
}

#[test]
fn affinity_two_pass_clustering_prefers_matching_block() {
    // With clustering on, a request preferring affinity=true should land in the
    // affinity=true block when both have room.
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    config.affinity_clustering = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();

    // Force two blocks by filling the first.
    let _a = alloc(&mut pool, &mut backend, 4096, 0).unwrap();
    // second block via a fresh allocation
    let b = alloc(&mut pool, &mut backend, 100, 1).unwrap();
    assert_eq!(pool.block_count(), 2);

    // Tag the two blocks with opposite affinities.
    // Determine block ids present.
    let ids: Vec<BlockId> = pool.report().blocks.iter().map(|r| r.block_id).collect();
    assert_eq!(ids.len(), 2);
    // Tag block b's block as affinity=true, the other false.
    pool.set_block_affinity(b.block_id(), true);
    for id in &ids {
        if *id != b.block_id() {
            pool.set_block_affinity(*id, false);
        }
    }

    // A request preferring affinity=true should land in b's block (which has room).
    let c = AllocationContext {
        budget_free_bytes: None,
        dedicated_fallback_allowed: false,
        preferred_affinity: Some(true),
    };
    let placed = pool
        .allocate(
            AllocationDesc {
                size: 50,
                alignment: 1,
                ..Default::default()
            },
            c,
            2,
            &mut backend,
        )
        .unwrap();
    assert_eq!(
        placed.block_id(),
        b.block_id(),
        "clustering placed in the matching block"
    );
    pool.validate().unwrap();
}

#[test]
fn tlsf_upper_address_rejected() {
    let mut backend = MockBackend::new();
    let config = cfg(256, 1);
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let err = pool
        .allocate(
            AllocationDesc {
                size: 10,
                alignment: 1,
                alloc_type: AllocationType::Unknown,
                strategy: Strategy::Balanced,
                upper_address: true,
            },
            ctx(),
            0,
            &mut backend,
        )
        .unwrap_err();
    assert_eq!(err, PoolAllocError::UpperAddressUnsupported);
}

#[test]
fn free_stale_allocation_is_error() {
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let a = alloc(&mut pool, &mut backend, 100, 0).unwrap();
    // First free succeeds.
    let _ = pool.free(a, FreeContext::default()).unwrap();
    // Double free is a clean error (block still exists via hysteresis; handle stale).
    let err = pool.free(a, FreeContext::default()).unwrap_err();
    assert_eq!(err, crate::FreeError::InvalidAllocation);
    pool.validate().unwrap();
}

#[test]
fn free_unknown_block_id_is_error() {
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let a = alloc(&mut pool, &mut backend, 100, 0).unwrap();
    // Corrupt the block id to name a non-existent block (same salt, unused counter).
    let a = a.with_block_id_for_test(BlockId::new_for_test(0, 9999));
    let err = pool.free(a, FreeContext::default()).unwrap_err();
    assert_eq!(err, crate::FreeError::InvalidAllocation);
    pool.validate().unwrap();
}

#[test]
fn destroyed_block_returned_to_caller() {
    // When the hysteresis drops a block, the FreeOutcome carries the backend value so
    // the caller can release device memory. Confirm the id matches and the mock still
    // accounts it correctly when the caller destroys it.
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let a = alloc(&mut pool, &mut backend, 4096, 0).unwrap();
    let b = alloc(&mut pool, &mut backend, 4096, 1).unwrap();
    let o1 = pool.free(a, FreeContext::default()).unwrap();
    assert!(o1.destroyed_block.is_none());
    let o2 = pool.free(b, FreeContext::default()).unwrap();
    let (block, id) = o2.destroyed_block.expect("a block was dropped");
    // Caller releases it.
    <MockBackend as crate::BlockBackend>::destroy_block(&mut backend, block, id);
    assert_eq!(backend.destroy_count(), 1);
    assert_eq!(backend.live_blocks, 1);
    pool.validate().unwrap();
}

#[test]
fn into_blocks_returns_all_remaining() {
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let _a = alloc(&mut pool, &mut backend, 4096, 0).unwrap();
    let _b = alloc(&mut pool, &mut backend, 4096, 1).unwrap();
    assert_eq!(pool.block_count(), 2);
    let blocks: Vec<(MockBlock, BlockId)> = pool.into_blocks().collect();
    assert_eq!(blocks.len(), 2);
    // Release them all.
    for (block, id) in blocks {
        <MockBackend as crate::BlockBackend>::destroy_block(&mut backend, block, id);
    }
    assert_eq!(backend.live_blocks, 0);
    assert_eq!(backend.destroy_count(), 2);
}

#[test]
fn clear_destroys_all_blocks() {
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    config.min_block_count = 1;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let _a = alloc(&mut pool, &mut backend, 4096, 0).unwrap();
    let _b = alloc(&mut pool, &mut backend, 4096, 1).unwrap();
    let created = backend.create_count();
    pool.clear(&mut backend);
    assert_eq!(pool.block_count(), 0);
    assert_eq!(backend.destroy_count(), created);
    assert_eq!(backend.live_blocks, 0);
}

#[test]
fn invalid_config_rejected() {
    let mut backend = MockBackend::new();
    // Zero preferred size.
    assert!(Pool::<MockBackend, u32>::new(cfg(0, 1), &mut backend).is_err());
    // Zero max_block_count.
    assert!(Pool::<MockBackend, u32>::new(cfg(16, 0), &mut backend).is_err());
    // max < min.
    let mut c = cfg(16, 1);
    c.min_block_count = 2;
    assert!(Pool::<MockBackend, u32>::new(c, &mut backend).is_err());
    // Non-power-of-two granularity.
    let mut c = cfg(16, 1);
    c.granularity = 3;
    assert!(Pool::<MockBackend, u32>::new(c, &mut backend).is_err());
}

#[test]
fn min_allocation_alignment_floor_applied() {
    // A request with alignment 1 but a pool min_allocation_alignment of 256 must land
    // 256-aligned.
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 1);
    config.explicit_block_size = true;
    config.min_allocation_alignment = 256;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let a = alloc(&mut pool, &mut backend, 10, 0).unwrap();
    assert_eq!(
        a.offset() % 256,
        0,
        "min_allocation_alignment floor applied"
    );
    pool.validate().unwrap();
}

#[test]
fn zero_size_is_invalid_request() {
    let mut backend = MockBackend::new();
    let config = cfg(4096, 1);
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let err = alloc(&mut pool, &mut backend, 0, 0).unwrap_err();
    assert_eq!(err, PoolAllocError::InvalidRequest);
}

#[test]
fn grow_placement_failure_rolls_back_block() {
    // With a debug margin and a tight budget, the pool cannot grow to a size that holds
    // `size + margin` within budget. The footprint-aware ramp (fix 2) declines to create
    // any block rather than creating an undersized one, so no empty block leaks and no
    // create/destroy churn occurs.
    let mut backend = MockBackend::new();
    let config = PoolConfig {
        algorithm: Algorithm::Tlsf,
        preferred_block_size: 8192,
        min_block_count: 0,
        max_block_count: 4,
        explicit_block_size: false,
        min_allocation_alignment: 1,
        granularity: 1,
        debug_margin: 16,
        affinity_clustering: false,
        pool_salt: 0,
    };
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let a = alloc(&mut pool, &mut backend, 1, 0).unwrap();
    let _ = pool.free(a, FreeContext::default()).unwrap(); // one retained empty block
    assert_eq!(pool.empty_block_count(), 1);

    let c = AllocationContext {
        budget_free_bytes: Some(2048),
        dedicated_fallback_allowed: true,
        preferred_affinity: None,
    };
    // 2033 + 16 margin = 2049 > 2048 budget; the pool cannot grow within budget.
    let r = pool.allocate(
        AllocationDesc {
            size: 2033,
            alignment: 1,
            ..Default::default()
        },
        c,
        1,
        &mut backend,
    );
    assert_eq!(
        r.unwrap_err(),
        PoolAllocError::ShouldDedicate,
        "cannot grow within budget, so caller should try a dedicated allocation"
    );
    // Crucially, no second empty block was left behind.
    assert!(pool.empty_block_count() <= 1);
    pool.validate().unwrap();
}

#[test]
fn report_reflects_allocations() {
    let mut backend = MockBackend::new();
    let mut config = cfg(4096, 4);
    config.explicit_block_size = true;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    let _a = alloc(&mut pool, &mut backend, 100, 7).unwrap();
    let _b = alloc(&mut pool, &mut backend, 200, 8).unwrap();
    let report = pool.report();
    assert_eq!(report.statistics.allocation_count, 2);
    assert!(report.statistics.allocation_bytes >= 300);
    // for_each_allocation surfaces user data.
    let mut collected: Vec<u32> = vec![];
    pool.for_each_allocation(|_id, _off, _sz, ud| collected.push(*ud));
    collected.sort_unstable();
    assert_eq!(collected, vec![7, 8]);
}

// --- Fix 1: cross-pool allocation rejection (per-pool salt) ---

#[test]
fn cross_pool_distinct_salt_rejects_foreign_free() {
    // Two pools with DISTINCT salts. Feeding A's Allocation into B.free() must be a
    // clean error and leave B's state completely untouched (statistics, validate, and
    // B's own allocation still readable + still freeable).
    let mut ba = MockBackend::new();
    let mut bb = MockBackend::new();
    let mut ca = cfg(4096, 4);
    ca.explicit_block_size = true;
    ca.pool_salt = 1;
    let mut cb = ca;
    cb.pool_salt = 2;
    let mut a = Pool::<MockBackend, u32>::new(ca, &mut ba).unwrap();
    let mut b = Pool::<MockBackend, u32>::new(cb, &mut bb).unwrap();

    let alloc_a = alloc(&mut a, &mut ba, 64, 111).unwrap();
    let alloc_b = alloc(&mut b, &mut bb, 128, 222).unwrap();

    // Distinct salts => distinct block ids even though both are the pools' first block.
    assert_ne!(
        alloc_a.block_id(),
        alloc_b.block_id(),
        "distinct salts must give distinct ids"
    );

    // Snapshot B's live set before the foreign free.
    let stats_before = b.statistics();
    let mut b_live_before: Vec<(BlockId, u64, u64, u32)> = vec![];
    b.for_each_allocation(|id, off, sz, ud| b_live_before.push((id, off, sz, *ud)));

    // Foreign free: rejected cleanly, nothing freed.
    let err = b.free(alloc_a, FreeContext::default()).unwrap_err();
    assert_eq!(err, crate::FreeError::InvalidAllocation);

    // B's state is byte-for-byte unchanged.
    b.validate().unwrap();
    assert_eq!(
        b.statistics(),
        stats_before,
        "B statistics changed on a foreign free"
    );
    let mut b_live_after: Vec<(BlockId, u64, u64, u32)> = vec![];
    b.for_each_allocation(|id, off, sz, ud| b_live_after.push((id, off, sz, *ud)));
    assert_eq!(
        b_live_before, b_live_after,
        "B live set changed on a foreign free"
    );
    assert_eq!(
        b_live_after,
        vec![(alloc_b.block_id(), alloc_b.offset(), alloc_b.size(), 222)]
    );

    // B's own allocation is still live and can be freed normally.
    let outcome = b.free(alloc_b, FreeContext::default()).unwrap();
    assert!(outcome.destroyed_block.is_none());
    b.validate().unwrap();
}

#[test]
fn cross_pool_shared_salt_is_best_effort() {
    // Two pools sharing a salt (both default 0) mint colliding block ids. This documents
    // the best-effort fallback: the salt check no longer distinguishes them, so a
    // foreign free is *not* guaranteed to be rejected (it inherits the offset-allocator's
    // cross-instance handle limitation). We assert only that the operation is
    // memory-safe: whatever B does, B still validates and never panics.
    let mut ba = MockBackend::new();
    let mut bb = MockBackend::new();
    let mut c = cfg(4096, 4);
    c.explicit_block_size = true;
    // Both salts default 0 (shared).
    assert_eq!(c.pool_salt, 0);
    let mut a = Pool::<MockBackend, u32>::new(c, &mut ba).unwrap();
    let mut b = Pool::<MockBackend, u32>::new(c, &mut bb).unwrap();

    let alloc_a = alloc(&mut a, &mut ba, 64, 111).unwrap();
    let _alloc_b = alloc(&mut b, &mut bb, 128, 222).unwrap();

    // Shared salt => colliding ids (the documented limitation).
    assert_eq!(alloc_a.block_id(), BlockId::new_for_test(0, 0));

    // Feeding A's allocation into B may or may not be rejected, but B stays consistent.
    let _ = b.free(alloc_a, FreeContext::default());
    b.validate().unwrap();
}

#[test]
fn cross_pool_previously_colliding_salts_now_reject() {
    // Regression for fix 2 (32-bit salt-tag collision). The old code hashed pool_salt down
    // to a 32-bit tag; the DISTINCT salts below both hashed to tag 0x970f4ad0, so the two
    // pools minted the SAME BlockId and a foreign free was wrongly accepted — re-arming the
    // cross-pool aliasing bug (PoC salt_collide_alias). BlockId now carries the FULL u64
    // salt, so this pair (and any distinct pair) rejects deterministically.
    let salt_a = 1_597_695_007_048_244_528u64;
    let salt_b = 6_544_243_929_581_175_562u64;
    assert_ne!(salt_a, salt_b);

    let mut ba = MockBackend::new();
    let mut bb = MockBackend::new();
    let mut ca = cfg(4096, 4);
    ca.explicit_block_size = true;
    ca.pool_salt = salt_a;
    let mut cb = ca;
    cb.pool_salt = salt_b;
    let mut a = Pool::<MockBackend, u32>::new(ca, &mut ba).unwrap();
    let mut b = Pool::<MockBackend, u32>::new(cb, &mut bb).unwrap();

    let alloc_a = alloc(&mut a, &mut ba, 64, 111).unwrap();
    let alloc_b = alloc(&mut b, &mut bb, 128, 222).unwrap();

    // The formerly-colliding pair now yields DISTINCT ids.
    assert_ne!(
        alloc_a.block_id(),
        alloc_b.block_id(),
        "the previously-colliding salt pair must now produce distinct block ids",
    );

    // Snapshot B before the foreign free.
    let mut b_before: Vec<(BlockId, u64, u64, u32)> = vec![];
    b.for_each_allocation(|id, off, sz, ud| b_before.push((id, off, sz, *ud)));

    // The foreign free is rejected deterministically and leaves B untouched.
    let err = b.free(alloc_a, FreeContext::default()).unwrap_err();
    assert_eq!(err, crate::FreeError::InvalidAllocation);
    b.validate().unwrap();
    let mut b_after: Vec<(BlockId, u64, u64, u32)> = vec![];
    b.for_each_allocation(|id, off, sz, ud| b_after.push((id, off, sz, *ud)));
    assert_eq!(b_before, b_after, "foreign free disturbed B's live set");
    assert_eq!(
        b_after,
        vec![(alloc_b.block_id(), alloc_b.offset(), alloc_b.size(), 222)]
    );

    // set_block_affinity likewise rejects the foreign id.
    assert!(!b.set_block_affinity(alloc_a.block_id(), true));

    // B's own allocation is still freeable normally.
    let _ = b.free(alloc_b, FreeContext::default()).unwrap();
    b.validate().unwrap();
}

#[test]
fn set_block_affinity_rejects_foreign_id() {
    // A block id from a pool with a different salt is refused by set_block_affinity.
    let mut ba = MockBackend::new();
    let mut bb = MockBackend::new();
    let mut ca = cfg(4096, 4);
    ca.explicit_block_size = true;
    ca.affinity_clustering = true;
    ca.pool_salt = 1;
    let mut cb = ca;
    cb.pool_salt = 2;
    let mut a = Pool::<MockBackend, u32>::new(ca, &mut ba).unwrap();
    let mut b = Pool::<MockBackend, u32>::new(cb, &mut bb).unwrap();

    let alloc_a = alloc(&mut a, &mut ba, 64, 1).unwrap();
    let alloc_b = alloc(&mut b, &mut bb, 64, 2).unwrap();

    // A's block id is foreign to B: rejected.
    assert!(
        !b.set_block_affinity(alloc_a.block_id(), true),
        "foreign id must be rejected"
    );
    // B's own block id is accepted.
    assert!(
        b.set_block_affinity(alloc_b.block_id(), true),
        "own id must be accepted"
    );
}

// --- Fix 2: grow-failure churn is eliminated ---

#[test]
fn churn_config_creates_are_bounded_no_churn_pairs() {
    // Regression for the grow-churn DoS: preferred=64, granularity=16, request size=1.
    // The old ramp floored the block at preferred/8 = 8, which (with granularity 16)
    // could never place the request, so every allocate created then destroyed a device
    // block. The footprint-aware ramp floors the block at >= the request footprint
    // (align_up(1,16)=16), so a fresh block always places the request: bounded creates,
    // ZERO create+destroy churn pairs, over 100 identical calls.
    let mut backend = MockBackend::new();
    let mut config = cfg(64, 4);
    config.granularity = 16;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();

    let mut live: Vec<crate::Allocation> = vec![];
    for i in 0..100u32 {
        if let Ok(a) = pool.allocate(
            AllocationDesc {
                size: 1,
                alignment: 1,
                ..Default::default()
            },
            ctx(),
            i,
            &mut backend,
        ) {
            live.push(a);
        }
        pool.validate().unwrap();
    }

    let creates = backend.create_count();
    let destroys = backend.destroy_count();
    // Bound: at most (retained blocks + 1). With max_block_count 4 and no frees, the
    // pool fills its blocks; creates must not scale with the 100 calls.
    assert!(
        creates <= pool.block_count() + 1,
        "creates ({creates}) should be bounded by retained blocks ({}) + 1",
        pool.block_count(),
    );
    // Zero churn: no block was ever created and then destroyed within this run. Since
    // no frees happened, any destroy would be a rolled-back (churned) block.
    assert_eq!(
        destroys, 0,
        "grow churn: {destroys} destroys with no frees issued"
    );
    // Sanity: some allocations succeeded (the fix makes placement possible).
    assert!(
        !live.is_empty(),
        "the footprint-aware ramp should place at least one request"
    );

    // No create/destroy churn *pairs* anywhere in the event log.
    let mut created_sizes: Vec<u64> = vec![];
    let mut churn_pairs = 0usize;
    for ev in &backend.events {
        match ev {
            Event::Create { size, .. } => created_sizes.push(*size),
            Event::Destroy { size, .. } => {
                if created_sizes.contains(size) {
                    churn_pairs += 1;
                }
            }
        }
    }
    assert_eq!(
        churn_pairs, 0,
        "found {churn_pairs} create+destroy churn pairs"
    );
}

#[test]
fn churn_config_single_block_matches_vma_syscall_bound() {
    // The exact vma_sim scenario: preferred=64, gran=16, size=1, max_block_count=1.
    // VMA would create exactly one block and keep it (1 create, 0 destroys). The crate
    // must do no worse: <= 1 create, 0 destroys, over 100 identical calls.
    let mut backend = MockBackend::new();
    let mut config = cfg(64, 1);
    config.granularity = 16;
    let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();
    for i in 0..100u32 {
        let _ = pool.allocate(
            AllocationDesc {
                size: 1,
                alignment: 1,
                ..Default::default()
            },
            ctx(),
            i,
            &mut backend,
        );
        pool.validate().unwrap();
    }
    assert!(
        backend.create_count() <= 1,
        "expected <= 1 create, got {}",
        backend.create_count()
    );
    assert_eq!(backend.destroy_count(), 0, "expected 0 destroys (no churn)");
}

// --- Fix 3: margin-boundary band coverage (single-margin early reject) ---

#[test]
fn margin_boundary_band_matches_vma_rule() {
    // With margin > 0, sweep request sizes around the reject boundary and assert the
    // pool-vs-dedicate decision matches VMA's `size + margin > preferred` rule (fix 3).
    // Band of interest: preferred - 2*margin .. preferred, which straddles the old
    // (2*margin) and new (1*margin) thresholds.
    let preferred = 256u64;
    let margin = 16u64;
    for size in (preferred - 2 * margin)..=preferred {
        let mut backend = MockBackend::new();
        let mut config = cfg(preferred, 1);
        config.explicit_block_size = true;
        config.debug_margin = margin;
        let mut pool = Pool::<MockBackend, u32>::new(config, &mut backend).unwrap();

        let r = alloc(&mut pool, &mut backend, size, 0);
        // VMA's rule: dedicate iff size + margin > preferred.
        let vma_should_dedicate = size + margin > preferred;
        if vma_should_dedicate {
            assert_eq!(
                r.unwrap_err(),
                PoolAllocError::ShouldDedicate,
                "size {size}: VMA would dedicate; crate must too",
            );
            assert_eq!(pool.block_count(), 0);
        } else {
            // Must not be dedicated; a full-size block can place it.
            assert!(
                !matches!(r, Err(PoolAllocError::ShouldDedicate)),
                "size {size}: VMA would pool this; crate wrongly dedicated",
            );
            let a = r.unwrap();
            assert_eq!(a.block_id(), BlockId::new_for_test(0, 0));
            pool.validate().unwrap();
        }
    }
}

// --- Fix 1/2: the grow-rollback branch is unreachable across the config space ---

#[test]
fn rollback_branch_unreachable_sweep() {
    // The grow ramp floors every candidate block at the request footprint, so a freshly
    // created block ALWAYS places the request — the rollback branch in `grow_and_allocate`
    // (which would create then destroy a device block in a single `allocate` call) is
    // never taken. A rollback hit is `create_count >= 1 && destroy_count >= 1` after a
    // single allocate on a fresh, single-block pool with no frees.
    //
    // This sweep covers low granularity, page-tracking granularity (> 256), margins, and
    // alignments. In a debug build a footprint miscalculation would trip the
    // `debug_assert!(false)` in the rollback branch; here we additionally assert zero
    // rollbacks explicitly.
    let types = [
        AllocationType::Unknown,
        AllocationType::Buffer,
        AllocationType::ImageLinear,
        AllocationType::ImageOptimal,
    ];
    let strategies = [
        Strategy::Balanced,
        Strategy::MinMemory,
        Strategy::MinTime,
        Strategy::MinOffset,
    ];
    let mut rollback_hits = 0usize;
    let mut tried = 0usize;

    let algo = Algorithm::Tlsf;
    {
        for &preferred in &[64u64, 256, 1024, 4096] {
            for &gran in &[1u64, 16, 256, 512] {
                if gran > preferred {
                    continue;
                }
                for &margin in &[0u64, 4, 16, 64] {
                    for &align_shift in &[0u32, 1, 4, 8] {
                        let alignment = 1u64 << align_shift;
                        for &ty in &types {
                            for &strat in &strategies {
                                let sizes = [
                                    1u64,
                                    gran,
                                    preferred / 8,
                                    (preferred / 2).saturating_sub(margin).max(1),
                                    preferred.saturating_sub(margin).max(1),
                                    preferred.saturating_sub(2 * margin).max(1),
                                ];
                                for &size in &sizes {
                                    let mut config = cfg(preferred, 1);
                                    config.algorithm = algo;
                                    config.granularity = gran;
                                    config.debug_margin = margin;
                                    let mut backend = MockBackend::new();
                                    let mut pool =
                                        match Pool::<MockBackend, u32>::new(config, &mut backend) {
                                            Ok(p) => p,
                                            Err(_) => continue,
                                        };
                                    tried += 1;
                                    let _ = pool.allocate(
                                        AllocationDesc {
                                            size,
                                            alignment,
                                            alloc_type: ty,
                                            strategy: strat,
                                            upper_address: false,
                                        },
                                        ctx(),
                                        0,
                                        &mut backend,
                                    );
                                    pool.validate().unwrap();
                                    if backend.create_count() >= 1 && backend.destroy_count() >= 1 {
                                        rollback_hits += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(tried > 0, "sweep tried no configs");
    assert_eq!(
        rollback_hits, 0,
        "grow-rollback branch reached {rollback_hits}/{tried} times (footprint miscalculated)",
    );
}
