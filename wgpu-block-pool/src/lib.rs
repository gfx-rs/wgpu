//! A generic "block vector / pool policy" layer over [`wgpu_offset_allocator`].
//!
//! [`wgpu_offset_allocator`] answers a purely arithmetic question about a *single*
//! fixed-size block. This crate answers the layer above it: given a growing and
//! shrinking set of memory blocks, each backed by one suballocator, *which* block
//! serves an allocation, and *when* should a block be created or destroyed?
//!
//! It is the Rust equivalent of AMD's `VmaBlockVector` / D3D12MA's `BlockVector`, the
//! policy layer VMA and D3D12MA wrap around their per-block metadata.
//!
//! # Mental model
//!
//! The pool owns *placement policy only*. Like [`wgpu_offset_allocator`], it never
//! touches memory, dereferences a pointer, performs I/O, or names any GPU API type:
//!
//! - Real memory lives behind the caller-implemented [`BlockBackend`]. The pool asks
//!   it to create or destroy blocks and stores its opaque
//!   [`Block`](BlockBackend::Block) values (e.g. a `vk::DeviceMemory` wrapper or an
//!   `ID3D12Heap` wrapper); the pool never allocates or frees device memory itself.
//! - Environment the pool deliberately does not own — budget headroom and the
//!   dedicated-allocation fallback — is passed per call as [`AllocationContext`] /
//!   [`FreeContext`].
//! - The pool is not internally synchronized. The caller owns locking, exactly as
//!   VMA / D3D12MA expect their block vectors to be called under a per-vector mutex.
//!
//! Start at [`Pool`]: create one with [`Pool::new`] and a [`PoolConfig`], then call
//! [`allocate`](Pool::allocate) / [`free`](Pool::free).
//!
//! # Contracts
//!
//! Because the pool does not own the memory it places, three obligations fall on the
//! caller:
//!
//! 1. **Route every destroyed block to the backend.** A successful [`Pool::free`]
//!    returns a `#[must_use]` [`FreeOutcome`]. If it carries a destroyed block, the
//!    pool has already forgotten that block; the caller must release it (normally via
//!    [`BlockBackend::destroy_block`]) or its memory leaks.
//! 2. **Drain the pool instead of dropping it.** Dropping a [`Pool`] drops its stored
//!    [`Block`](BlockBackend::Block) values without any backend call. Call
//!    [`Pool::clear`], or consume the pool with [`Pool::into_blocks`] and destroy
//!    every returned block.
//! 3. **Keep the user-data type `T` cheap to clone.** `T` is cloned on each
//!    candidate-block attempt inside [`Pool::allocate`] and read by value when
//!    reporting; use an index, an id, a refcounted pointer, or `()`.
//!
//! Misuse in the other direction is *detected* rather than trusted: an [`Allocation`]
//! handed to the wrong pool, already freed, or stale is rejected with [`FreeError`]
//! instead of corrupting pool state. Give every pool a distinct
//! [`PoolConfig::pool_salt`] to make the cross-pool case deterministic.
//!
//! # Example
//!
//! ```
//! use wgpu_block_pool::{
//!     AllocationContext, AllocationDesc, AllocationType, BlockBackend, BlockId, FreeContext,
//!     Pool, PoolConfig,
//! };
//!
//! // A backend that just counts blocks; a real one would create and destroy device
//! // memory (`vkAllocateMemory`, `CreateHeap`, ...), keyed by `BlockId` if useful.
//! struct Backend {
//!     live_blocks: usize,
//! }
//!
//! impl BlockBackend for Backend {
//!     type Block = u64; // stand-in for a real device-memory handle
//!     type Error = core::convert::Infallible;
//!
//!     fn create_block(&mut self, size: u64, _id: BlockId) -> Result<u64, Self::Error> {
//!         self.live_blocks += 1;
//!         Ok(size)
//!     }
//!     fn destroy_block(&mut self, _block: u64, _id: BlockId) {
//!         self.live_blocks -= 1;
//!     }
//! }
//!
//! let mut backend = Backend { live_blocks: 0 };
//! let mut pool = Pool::<Backend, ()>::new(
//!     PoolConfig {
//!         preferred_block_size: 64 * 1024,
//!         max_block_count: 8,
//!         ..PoolConfig::default()
//!     },
//!     &mut backend,
//! )
//! .unwrap();
//!
//! // The pool grows itself through the backend as needed.
//! let alloc = pool
//!     .allocate(
//!         AllocationDesc {
//!             size: 256,
//!             alignment: 16,
//!             alloc_type: AllocationType::Buffer,
//!             ..Default::default()
//!         },
//!         AllocationContext::default(),
//!         (),                           // per-allocation user data
//!         &mut backend,
//!     )
//!     .unwrap();
//! assert_eq!(alloc.offset() % 16, 0);
//!
//! // Contract 1: route any block the free destroyed back to the backend.
//! let outcome = pool.free(alloc, FreeContext::default()).unwrap();
//! if let Some((block, id)) = outcome.destroyed_block {
//!     backend.destroy_block(block, id);
//! }
//!
//! // Contract 2: drain the pool before dropping it.
//! for (block, id) in pool.into_blocks() {
//!     backend.destroy_block(block, id);
//! }
//! assert_eq!(backend.live_blocks, 0);
//! ```
//!
//! # Algorithms and provenance
//!
//! The policy is ported from AMD's [VulkanMemoryAllocator][vma] and
//! [D3D12MemoryAllocator][d3d12ma], both MIT licensed:
//!
//! - [`Pool::allocate`] ports `VmaBlockVector::AllocatePage` (vk_mem_alloc.h) and
//!   `D3D12MA::BlockVector::AllocatePage` (D3D12MemAlloc.cpp): early size reject, scan
//!   existing blocks (smallest-free-first, or largest-free-first for
//!   [`Strategy::MinTime`]), then grow the pool with a `1/8 -> 1/4 -> 1/2 -> full`
//!   block-size ramp that halves further on backend failure.
//! - [`Pool::free`] ports `VmaBlockVector::Free` / `D3D12MA::BlockVector::Free`: the
//!   empty-block hysteresis (retain exactly one empty block unless the budget is
//!   exceeded), plus reclaiming a trailing empty block on a non-empty free.
//! - Blocks are kept incrementally sorted by free size (one bubble-sort step per
//!   mutation), a port of `IncrementallySortBlocks`.
//!
//! [vma]: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
//! [d3d12ma]: https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator
//!
//! # Fidelity notes
//!
//! This is a faithful port of VMA / D3D12MA block-vector policy, with a few
//! deliberate, documented divergences:
//!
//! - **Backend abstraction.** VMA / D3D12MA create and destroy device memory
//!   directly. This crate delegates both to the caller's [`BlockBackend`], and never
//!   destroys device memory itself: destroyed blocks are returned to the caller.
//! - **Explicit per-call context.** VMA / D3D12MA read the heap budget and the
//!   "can fall back to a dedicated/committed allocation" flag from global allocator
//!   state. This crate takes them per call as [`AllocationContext`] /
//!   [`FreeContext`], so the pool owns no global state.
//! - **Generic affinity tag.** VMA's host-visible mapped/unmapped two-pass
//!   clustering (which prefers already-mapped blocks for mappable requests) is
//!   generalized to a caller-set boolean "affinity tag" per block plus an optional
//!   preferred tag per request. D3D12MA has no such pass; this crate makes it opt-in.
//! - **Checked arithmetic.** All size math uses checked/saturating operations;
//!   no public input can panic or overflow (the C++ uses raw `size + margin`,
//!   `size * 2`, `size / 2`).
//! - **Detected, memory-safe stale handles.** A stale or foreign [`Allocation`]
//!   passed to [`free`](Pool::free) is reported as [`FreeError`] (via the
//!   suballocator's generation-tagged handle plus a block-id check) rather than
//!   corrupting pool state.
//! - **Per-pool salt.** Each pool stamps its full [`PoolConfig::pool_salt`] into
//!   every [`BlockId`] it mints; [`free`](Pool::free) /
//!   [`set_block_affinity`](Pool::set_block_affinity) reject an [`Allocation`] whose salt
//!   names a *different* pool. Because the whole 64-bit salt is compared (not a hashed-
//!   down tag), pools with distinct salts refuse each other's allocations
//!   deterministically, with no possibility of collision. Pools *sharing* a salt fall
//!   back to the generation-tagged handle check (the offset-allocator's documented
//!   cross-instance limitation). See [`PoolConfig::pool_salt`].
//! - **Request-aware grow ramp.** The C++ can create a block that granularity
//!   rounding then makes too small for the request, either leaving a useless block
//!   behind or (with a rollback) creating and destroying a device block on every call.
//!   This crate floors every candidate block size at the request's footprint, so a
//!   freshly created block always places the request and that churn cannot occur; if
//!   the footprint exceeds the pool's largest block, [`allocate`](Pool::allocate)
//!   returns [`PoolAllocError::ShouldDedicate`] without creating anything.
//!
//! # Panic freedom
//!
//! No panic is reachable from any input; see the per-method docs. All size arithmetic
//! is checked or saturating, and the internal edge case that the C++ handles by leaving
//! an unusable block in the vector — a freshly created block that cannot place the
//! request because of granularity rounding or a debug margin — is eliminated at the
//! root: the grow ramp floors every candidate block size at the request's footprint, so
//! a fresh block always places the request (see [`Pool::allocate`]).
//!
//! # `no_std`
//!
//! This crate is `#![no_std]`; it uses [`alloc`]. There are no Cargo features and no
//! mandatory runtime dependencies beyond [`wgpu_offset_allocator`].

#![no_std]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod pool;

#[cfg(test)]
mod tests;

pub use pool::{
    Allocation, AllocationContext, BlockBackend, BlockId, BlockReport, FreeContext, FreeError,
    FreeOutcome, Pool, PoolAllocError, PoolConfig, PoolReport,
};

// Re-export the pieces of the offset-allocator surface that appear in this crate's
// public API, so callers need not name both crates for the common path.
pub use wgpu_offset_allocator::{
    Algorithm, AllocationDesc, AllocationType, DetailedStatistics, Statistics, Strategy,
};
