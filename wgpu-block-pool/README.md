# wgpu-block-pool

A generic, `no_std`, `#![forbid(unsafe_code)]` "block vector / pool policy" layer
that sits on top of [`wgpu-offset-allocator`].

Where `wgpu-offset-allocator` answers "given _one_ fixed-size block, where does the
next allocation go?", this crate answers the layer above: it manages a growing and
shrinking set of memory blocks, each backed by one suballocator, and decides _which_
block serves an allocation and _when_ to create or destroy blocks. It is entirely
GPU-API-free: the device memory backing each block is abstracted behind the
caller-implemented [`BlockBackend`] trait, so this crate never names a Vulkan, D3D12,
Metal, or GL type and never touches memory.

[`wgpu-offset-allocator`]: https://crates.io/crates/wgpu-offset-allocator

```rust
// The caller implements BlockBackend (create/destroy real memory blocks)...
let mut pool = Pool::<MyBackend, ()>::new(config, &mut backend)?;

// ...and the pool decides which block serves each allocation.
let desc = AllocationDesc { size, alignment, alloc_type, ..Default::default() };
let alloc = pool.allocate(desc, ctx, (), &mut backend)?;
let outcome = pool.free(alloc, FreeContext::default())?;
if let Some((block, id)) = outcome.destroyed_block {
    backend.destroy_block(block, id); // the caller releases destroyed blocks
}
```

See the crate-level rustdoc for a complete runnable example and the caller
contracts (routing `FreeOutcome`, draining the pool before drop).

## What it does

- Owns a `Vec` of blocks, each a TLSF `Suballocator` plus the caller's opaque
  `Block` handle and a stable, caller-visible block id.
- Implements VMA's / D3D12MA's block-search and block-creation policy: keep blocks
  incrementally sorted by free size, scan smallest-free-first for best packing (or
  largest-free-first for `MinTime`), and grow the pool with a `1/8 -> 1/4 -> 1/2 ->
full` block-size ramp, halving further on backend allocation failure.
- Implements the empty-block hysteresis on free: retain exactly one empty block
  unless the budget is exceeded (or another empty block already exists), and never
  drop below `min_block_count`.
- Never allocates or frees device memory itself: block creation and destruction are
  delegated to the [`BlockBackend`], and destroyed blocks are handed back to the
  caller to release.

## Provenance

The policy in this crate is ported from AMD's
[VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
(`VmaBlockVector`: `AllocatePage`, `Free`, `IncrementallySortBlocks`,
`CalcMaxBlockSize`, `HasEmptyBlock`) and
[D3D12MemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator)
(`BlockVector`: `AllocatePage`, `Free`, `IncrementallySortBlocks`). Both are MIT
licensed.

## Fidelity notes

The port is a faithful reproduction of VMA / D3D12MA block-vector policy, with a few
deliberate, documented divergences (see the crate-level rustdoc for the full list and
rationale):

- **Backend abstraction instead of a GPU allocator.** Block creation and destruction
  go through the caller's [`BlockBackend`]; the pool never names a device API.
- **Explicit per-call `AllocationContext`.** VMA / D3D12MA read the heap budget and
  the "can fall back to committed/dedicated" decision from global allocator state.
  This crate takes them as an explicit per-call context so the pool owns no global
  state.
- **Generic affinity tag instead of Vulkan `IsMapped`.** VMA's mapped/unmapped
  two-pass clustering is generalized to a caller-set boolean "affinity tag" per block
  and an optional preferred tag per request.
- **Checked arithmetic everywhere.** All size math (`size + margin`, `size * 2`, the
  halving ramp) uses checked/saturating operations; no public input can panic or
  overflow.
- **Request-aware grow ramp.** VMA / D3D12MA can create a block that granularity
  rounding then makes too small to place the request, returning out-of-memory but
  leaving the useless block behind (VMA) — or, if the block is rolled back, creating
  and destroying a device block on _every_ allocate call. This crate floors every
  candidate block size at the request's footprint (its granularity-rounded size plus
  one debug margin), so a freshly created block always places the request and the
  create/destroy churn cannot occur; if the footprint exceeds the largest block the
  pool can make, `allocate` returns `ShouldDedicate` without creating anything.
- **Per-pool salt for cross-pool rejection.** Each pool stamps its full
  `PoolConfig::pool_salt` (all 64 bits) into every `BlockId`. `free` and
  `set_block_affinity` reject an [`Allocation`] whose salt names a _different_ pool, so
  pools constructed with distinct salts refuse each other's allocations
  _deterministically_ — the whole `u64` salt is compared, so distinct salts can never
  collide. Pools sharing a salt (e.g. both default `0`) fall back to the suballocator's
  generation-tagged handle check, which inherits the offset-allocator's cross-instance
  limitation; the intended usage is to give each pool a unique salt.
- **Detected, memory-safe stale handles.** A stale or foreign [`Allocation`] passed
  to `free` is reported as an error (via the salt tag, the block-id lookup, and the
  underlying suballocator's generation-tagged handle) rather than corrupting pool
  state.

## License and attribution

This crate is a derivative work of AMD's
[VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
and
[D3D12MemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator),
both of which are MIT licensed and copyright Advanced Micro Devices, Inc. (see
the "Provenance" section above for the specific block-vector policy ported).

Because of this, and unlike the rest of the wgpu workspace (which is
`MIT OR Apache-2.0`), this crate is licensed **MIT only**, to remain compatible
with its upstream sources. The upstream AMD copyright notices are preserved
alongside the gfx-rs developers' copyright in the crate's [`LICENSE`](LICENSE)
file.
