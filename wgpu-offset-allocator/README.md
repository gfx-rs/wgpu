# wgpu-offset-allocator

A pure, `no_std`, `#![forbid(unsafe_code)]` index/offset suballocation library.

This crate operates entirely on integers. It answers the question "given a block
of `size` units, where should the next allocation of `size`/`alignment` go, and
which regions are free?" — it never touches memory, pointers, or any GPU API type.
It is the moral equivalent of VMA's `VmaVirtualBlock` / D3D12MA's `VirtualBlock`,
exposed both as a standalone [`VirtualBlock`] facade and as a lower-level
[`Suballocator`] trait so a higher-level block-pool layer — such as the
[`wgpu-block-pool`] crate — can drive many blocks.

```rust
use wgpu_offset_allocator::{Algorithm, AllocationDesc, VirtualBlock};

let mut block = VirtualBlock::<()>::new(1 << 20, Algorithm::Tlsf).unwrap();
let (handle, offset) = block
    .allocate(AllocationDesc { size: 4096, alignment: 256, ..Default::default() }, ())
    .unwrap();
block.free(handle).unwrap();
```

[`wgpu-block-pool`]: https://crates.io/crates/wgpu-block-pool

## Algorithm

The [`Suballocator`] trait is implemented by:

- **[`Tlsf`]** — a Two-Level Segregated Fit allocator. General-purpose,
  low-fragmentation, O(1) amortized allocation and free. This is the default and
  the recommended algorithm.

It supports optional _buffer-image granularity_ handling (for placing linear and
optimal-tiling resources in the same block without aliasing hazards) and optional
_debug margins_ (reserved padding after each allocation for later corruption
checking by a memory-touching layer).

## Provenance

The algorithms in this crate are derived from AMD's
[VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
(`VmaBlockMetadata_TLSF`, `VmaBlockBufferImageGranularity`, `VmaVirtualBlock_T`) and
[D3D12MemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator)
(`BlockMetadata_TLSF`, `VirtualBlock`). Both are MIT licensed.

The C++ originals use intrusive linked lists of heap-allocated nodes addressed by
raw pointers. To satisfy `#![forbid(unsafe_code)]`, this port replaces those with a
slab/arena of nodes addressed by `u32` indices (`NodeIndex`), with a free-node
recycling list. This is a faithful algorithmic port with equivalent asymptotic
performance.

## Fidelity notes

The port is faithful to VMA / D3D12MA, with a few deliberate, documented divergences
(see the crate-level rustdoc for the full list and rationale):

- **Widened top-level TLSF bitmap** from `u32` to `u64`, with all shifts made total,
  to support the full `u64` block-size range with no undefined shifts.
- **Bounded granularity page tracking:** a `(size, granularity)` pair requiring too
  many per-page records is rejected (`CreateError::GranularityTrackingTooLarge`)
  instead of aborting (64-bit) or silently truncating (32-bit / `wasm32`).
- **`u32` per-page allocation counts** (VMA uses `u16`), rejecting cleanly on overflow
  rather than wrapping.
- **Explicit debug-margin-filler marker** instead of VMA's `size == debug_margin`
  heuristic, so genuine free blocks always coalesce and `is_empty()` stays truthful.
- **Detected, memory-safe handle errors:** stale / double-freed / foreign handles
  return a `HandleError` (identically in debug and release) rather than corrupting
  allocator state.
- **Zero-alignment coercion** to `1` at the `VirtualBlock` facade, matching VMA's
  `vmaVirtualAllocate`.

## License and attribution

This crate is a derivative work of AMD's
[VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
and
[D3D12MemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator),
both of which are MIT licensed and copyright Advanced Micro Devices, Inc. (see
the "Provenance" section above for the specific algorithms ported).

Because of this, and unlike the rest of the wgpu workspace (which is
`MIT OR Apache-2.0`), this crate is licensed **MIT only**, to remain compatible
with its upstream sources. The upstream AMD copyright notices are preserved
alongside the gfx-rs developers' copyright in the crate's [`LICENSE`](LICENSE)
file.
