//! Pure integer offset/size suballocation.
//!
//! This crate answers a purely arithmetic question: given a block of `size`
//! "units" (bytes, texels, whatever the caller decides), where should the next
//! allocation of a given size and alignment be placed, and which sub-ranges are
//! free? It never dereferences a pointer, touches memory, performs I/O, or names
//! any GPU API type. Everything is [`u64`] offsets and sizes.
//!
//! It is the Rust equivalent of AMD's `VmaVirtualBlock` / D3D12MA's `VirtualBlock`,
//! plus the underlying metadata algorithms exposed through the [`Suballocator`]
//! trait so that a higher-level block-pool layer (such as the `wgpu-block-pool`
//! crate) can drive many blocks and implement defragmentation.
//!
//! # Entry points
//!
//! - [`VirtualBlock`] — the standalone, batteries-included facade. Create one with
//!   [`VirtualBlock::new`], then [`allocate`](VirtualBlock::allocate) /
//!   [`free`](VirtualBlock::free). Start here if you just need to carve up a
//!   single address range.
//! - [`Tlsf`] — the [`Suballocator`] implementation, if you
//!   need direct control (e.g. the split-phase [`create_allocation_request`] /
//!   [`alloc`] flow to probe several blocks before committing to one).
//!
//! [`create_allocation_request`]: Suballocator::create_allocation_request
//! [`alloc`]: Suballocator::alloc
//!
//! # Quick start
//!
//! ```
//! use wgpu_offset_allocator::{Algorithm, AllocationDesc, VirtualBlock};
//!
//! // Carve up 1 MiB of anything addressable: a GPU heap, a file, an index space.
//! let mut block = VirtualBlock::<()>::new(1 << 20, Algorithm::Tlsf).unwrap();
//!
//! let (handle, offset) = block
//!     .allocate(AllocationDesc { size: 4096, alignment: 256, ..Default::default() }, ())
//!     .unwrap();
//! assert_eq!(offset % 256, 0);
//!
//! // Handles are validated: a double free is reported as an error instead of
//! // corrupting allocator state.
//! block.free(handle).unwrap();
//! assert!(block.free(handle).is_err());
//! ```
//!
//! # Algorithms and provenance
//!
//! The algorithms are ported from AMD's
//! [VulkanMemoryAllocator][vma] and [D3D12MemoryAllocator][d3d12ma], both MIT
//! licensed:
//!
//! - [`Tlsf`] ports `VmaBlockMetadata_TLSF` / D3D12MA's `BlockMetadata_TLSF`: a
//!   Two-Level Segregated Fit allocator with O(1) amortized alloc/free and low
//!   fragmentation. This is the recommended general-purpose algorithm.
//! - Buffer-image granularity handling ports `VmaBlockBufferImageGranularity`.
//! - [`VirtualBlock`] ports `VmaVirtualBlock_T`.
//!
//! [vma]: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
//! [d3d12ma]: https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator
//!
//! # Architecture: arena of indices instead of intrusive pointers
//!
//! The C++ originals use intrusive doubly linked lists of heap-allocated nodes
//! addressed by raw pointers, plus a union that overlays `nextFree` and `userData`
//! and uses `prevFree == self` as the "this block is taken" marker. To satisfy
//! [`forbid(unsafe_code)`](https://doc.rust-lang.org/reference/attributes/diagnostics.html),
//! this crate replaces all of that with:
//!
//! - a [`Vec`](alloc::vec::Vec)-backed slab/arena of block nodes, each addressed by
//!   a [`u32`] index rather than a pointer, with a free-node recycling list so that
//!   freed node slots are reused (mirroring VMA's `VmaPoolAllocator`);
//! - an explicit enum (per-node "taken vs free" state) instead of the pointer-union
//!   trick, so the taken/free distinction and the user data / free-list links are
//!   separate, type-checked fields.
//!
//! This keeps the code 100% safe with equivalent asymptotic performance.
//!
//! # Panic freedom
//!
//! No panic is reachable from any *public* input. In particular:
//!
//! - a zero-size allocation request returns [`AllocationError::InvalidSize`];
//! - a non-power-of-two alignment returns [`AllocationError::InvalidAlignment`];
//! - astronomically large sizes/alignments (up to [`u64::MAX`]) never overflow — all
//!   placement arithmetic uses checked/saturating operations and is written to avoid
//!   the overflow-prone `offset + size <= end` forms found in the C++ source.
//!
//! Internal invariant violations that would indicate a bug *in this crate* use
//! [`debug_assert!`], and a small number of documented `unreachable!`/`panic!`
//! guards. [`Suballocator::validate`] performs a full invariant check and is used
//! heavily by the test suite.
//!
//! # Handle validity
//!
//! Every method that takes an [`AllocationHandle`] ([`free`](Suballocator::free),
//! [`allocation_offset`](Suballocator::allocation_offset),
//! [`allocation_info`](Suballocator::allocation_info),
//! [`set_user_data`](Suballocator::set_user_data), and the [`VirtualBlock`]
//! equivalents) returns a [`Result`]: a stale, already-freed (double-free), or foreign
//! handle yields [`HandleError`] instead of corrupting allocator state, and this holds
//! **identically in debug and release builds** — there is no debug-only assertion that
//! turns into undefined behaviour when disabled. See [`HandleError`] for the precise,
//! per-algorithm detection guarantees (generation-counter based for [`Tlsf`]).
//!
//! # Fidelity notes
//!
//! This crate is a faithful algorithmic port of VMA / D3D12MA, with a few deliberate,
//! documented divergences:
//!
//! - **Widened top-level bitmap.** VMA/D3D12MA store the TLSF top-level bitmap in a
//!   `uint32_t`, which only covers block sizes below ~512 GiB. This crate widens it to
//!   [`u64`] and makes all shifts total, supporting the full [`u64`] size range with no
//!   undefined shifts.
//! - **Bounded granularity page tracking.** VMA allocates one page-tracking record
//!   per granularity page unconditionally. This crate rejects a `(size, granularity)`
//!   pair whose page count exceeds a bounded cap with
//!   [`CreateError::GranularityTrackingTooLarge`], so a hostile pair cannot abort the
//!   process (64-bit) or silently truncate the page count (32-bit / `wasm32`).
//! - **`u32` per-page allocation counts.** VMA uses a `uint16_t` per-page counter,
//!   which overflows once more than `65535` allocations share one page (possible with a
//!   large granularity). This crate uses a `u32` counter and rejects the offending
//!   request cleanly rather than overflowing.
//! - **Explicit margin-filler marker.** VMA distinguishes a debug-margin filler
//!   free block from a genuine free block by its size (`size == debug_margin`). That
//!   strands real free blocks that happen to equal the margin size and makes
//!   [`is_empty`](Suballocator::is_empty) lie. This crate marks margin fillers
//!   explicitly, so genuine free blocks always coalesce and `is_empty` stays truthful.
//! - **Zero alignment coercion.** [`VirtualBlock::allocate`] coerces a zero
//!   alignment to `1` (as VMA's `vmaVirtualAllocate` does); the lower-level
//!   [`Suballocator`] trait still rejects zero alignment with
//!   [`AllocationError::InvalidAlignment`].
//!
//! # `no_std`
//!
//! This crate is `#![no_std]`; it uses [`alloc`]. Enable nothing — there are no
//! Cargo features and no mandatory runtime dependencies.

#![no_std]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod granularity;
mod math;
mod statistics;
mod tlsf;
mod virtual_block;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod reference;
#[cfg(test)]
mod tests;

pub use statistics::{DetailedStatistics, Statistics};
pub use tlsf::Tlsf;
pub use virtual_block::{Algorithm, VirtualBlock};

/// The 32-bit "magic" value VMA and D3D12MA write into debug-margin padding to
/// detect buffer overruns (`VMA_CORRUPTION_DETECTION_MAGIC_VALUE`).
///
/// This crate never writes it — it only reserves and accounts for the padding
/// space (see [debug margins](Suballocator::new)). A higher layer that has real
/// access to the underlying memory can use this constant, together with
/// [`debug_margin_offset`], to write and later verify the pattern.
pub const CORRUPTION_DETECTION_MAGIC_VALUE: u32 = 0x7F84_E666;

/// Returns the offset at which the debug-margin padding for an allocation begins,
/// i.e. the offset just past the usable end of the allocation.
///
/// A memory-touching layer would write [`CORRUPTION_DETECTION_MAGIC_VALUE`] into the
/// `debug_margin` units starting here (as `debug_margin / 4` little-endian `u32`s),
/// then re-check them on free. This crate guarantees `debug_margin` is a multiple of
/// 4 (validated at construction), so the region is always a whole number of `u32`s.
///
/// This is pure arithmetic; it does not consult any allocator state.
#[inline]
#[must_use]
pub const fn debug_margin_offset(allocation_offset: u64, allocation_size: u64) -> u64 {
    // The margin sits immediately after the usable allocation. Uses saturating add
    // so this helper is panic-free for any inputs.
    allocation_offset.saturating_add(allocation_size)
}

/// The kind of resource an allocation holds, used to enforce buffer-image
/// granularity (`VmaSuballocationType`).
///
/// Two allocations that *conflict* (see [`Self::conflicts_with`]) must not share a
/// granularity page. When granularity is disabled (`granularity <= 1`, or the block
/// is [virtual](Suballocator::is_virtual)) the type is irrelevant and may be left as
/// [`AllocationType::Unknown`].
///
/// The discriminant values match VMA's `VmaSuballocationType` so the conflict table
/// ports directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AllocationType {
    /// A free region. Never conflicts with anything. Used internally; not a valid
    /// input to [`allocate`](VirtualBlock::allocate).
    Free = 0,
    /// Unknown resource kind. Conservatively conflicts with *everything*.
    #[default]
    Unknown = 1,
    /// A buffer.
    Buffer = 2,
    /// An image of unknown tiling. Conservatively conflicts like a buffer plus
    /// images.
    ImageUnknown = 3,
    /// A linear-tiled image.
    ImageLinear = 4,
    /// An optimal-tiled image.
    ImageOptimal = 5,
}

impl AllocationType {
    /// Returns whether two allocation types conflict for buffer-image granularity
    /// purposes, i.e. must not be placed on the same granularity page.
    ///
    /// This is a direct port of VMA's `VmaIsBufferImageGranularityConflict`:
    ///
    /// - [`Free`](Self::Free) never conflicts.
    /// - [`Unknown`](Self::Unknown) conflicts with everything.
    /// - [`Buffer`](Self::Buffer) conflicts with [`ImageUnknown`](Self::ImageUnknown)
    ///   and [`ImageOptimal`](Self::ImageOptimal).
    /// - [`ImageUnknown`](Self::ImageUnknown) conflicts with itself,
    ///   [`ImageLinear`](Self::ImageLinear), and [`ImageOptimal`](Self::ImageOptimal).
    /// - [`ImageLinear`](Self::ImageLinear) conflicts with
    ///   [`ImageOptimal`](Self::ImageOptimal).
    /// - [`ImageOptimal`](Self::ImageOptimal) does *not* conflict with itself.
    ///
    /// The relation is symmetric.
    #[must_use]
    pub fn conflicts_with(self, other: Self) -> bool {
        // Port of VmaIsBufferImageGranularityConflict (vk_mem_alloc.h): sort the pair
        // by discriminant, then match on the smaller.
        let (a, b) = if (self as u8) <= (other as u8) {
            (self, other)
        } else {
            (other, self)
        };
        use AllocationType::*;
        match a {
            Free => false,
            Unknown => true,
            Buffer => matches!(b, ImageUnknown | ImageOptimal),
            ImageUnknown => matches!(b, ImageUnknown | ImageLinear | ImageOptimal),
            ImageLinear => matches!(b, ImageOptimal),
            ImageOptimal => false,
        }
    }
}

/// Placement strategy (`VMA_ALLOCATION_CREATE_STRATEGY_*`).
///
/// Controls the order in which free regions are probed. All strategies find *a*
/// valid placement if one exists; they differ only in which one they prefer and how
/// much work they do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Strategy {
    /// Balance speed and fragmentation. The default. For [`Tlsf`], probes the
    /// next-larger size bucket, then the null block, then best-fit.
    #[default]
    Balanced,
    /// Minimize wasted memory / fragmentation, preferring the tightest fit
    /// (`VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT`).
    MinMemory,
    /// Minimize allocation time (`VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT`).
    MinTime,
    /// Prefer the lowest offset that fits
    /// (`VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT`). Slower; primarily useful
    /// for defragmentation.
    MinOffset,
}

/// A fully-described request for a single allocation.
///
/// This is the *what to allocate* half of an allocation call: the size, alignment, and
/// the placement knobs. It is shared verbatim across every allocation entry point —
/// [`Suballocator::create_allocation_request`], [`VirtualBlock::allocate`], and (re-exported)
/// `wgpu_block_pool::Pool::allocate` — so those APIs cannot disagree on parameter order.
///
/// # Mental model
///
/// Two fields are always meaningful; the rest have sensible defaults, so the terse form
///
/// ```
/// # use wgpu_offset_allocator::AllocationDesc;
/// let desc = AllocationDesc { size: 4096, alignment: 256, ..Default::default() };
/// ```
///
/// requests a balanced, lower-address allocation of an [`Unknown`](AllocationType) resource.
///
/// The [`Default`] is `size = 0`, `alignment = 0`, [`Strategy::Balanced`], `upper_address =
/// false`, [`AllocationType::Unknown`]. A default `size` of `0` is *not* a valid request —
/// it exists only so `..Default::default()` can fill in the placement knobs; an allocation
/// with `size == 0` is rejected with [`AllocationError::InvalidSize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AllocationDesc {
    /// The number of units to allocate. Must be non-zero
    /// ([`AllocationError::InvalidSize`] otherwise).
    pub size: u64,
    /// The required alignment of the allocation's offset. Must be a power of two
    /// ([`AllocationError::InvalidAlignment`] otherwise); [`VirtualBlock::allocate`]
    /// additionally coerces `0` to `1`.
    pub alignment: u64,
    /// The resource kind, for buffer-image granularity conflict checking. Ignored when
    /// granularity is disabled (the default is [`AllocationType::Unknown`]).
    pub alloc_type: AllocationType,
    /// Allocate top-down. [`Tlsf`] does not support this and returns
    /// [`AllocationError::UpperAddressUnsupported`]. Defaults to `false`.
    pub upper_address: bool,
    /// The placement strategy (see [`Strategy`]). Defaults to
    /// [`Strategy::Balanced`].
    pub strategy: Strategy,
}

/// An opaque handle to a live allocation within a [`Suballocator`] or
/// [`VirtualBlock`].
///
/// Returned by [`allocate`](VirtualBlock::allocate) / [`alloc`](Suballocator::alloc)
/// and passed back to [`free`](VirtualBlock::free) and the info accessors.
///
/// Handles are only meaningful for the allocator that produced them. Passing a handle
/// to a different allocator, or reusing one after it has been freed (or after
/// [`clear`](Suballocator::clear)), is a caller error — but it is a *detected*,
/// *memory-safe* one: the handle-taking methods return [`HandleError`] rather than
/// corrupting allocator state, both in debug and release builds. See [`HandleError`]
/// for the exact detection guarantees.
///
/// # Encoding
///
/// The `u64` is algorithm-private. For [`Tlsf`] it packs a `u32` node index in the low
/// 32 bits and a `u32` per-node generation in the high 32 bits, so a stale handle is
/// caught by a generation mismatch. Callers must treat the value as opaque.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationHandle(pub(crate) u64);

/// Information about a live allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationInfo<T> {
    /// The offset of the allocation within the block.
    pub offset: u64,
    /// The usable size of the allocation (not counting any debug margin).
    pub size: u64,
    /// The user data associated with the allocation, if any.
    pub user_data: T,
}

/// Error returned when an allocation request is invalid or cannot be satisfied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationError {
    /// The requested size was zero. Zero-size allocations are not permitted.
    InvalidSize,
    /// The requested alignment was zero or not a power of two.
    InvalidAlignment,
    /// Upper-address (top-down) allocation was requested from an algorithm that does
    /// not support it. [`Tlsf`] does not support upper-address allocation.
    UpperAddressUnsupported,
    /// The block does not have a free region that satisfies the request. This is the
    /// ordinary "out of space" case, not a programming error.
    OutOfSpace,
}

impl core::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            AllocationError::InvalidSize => "allocation size must be non-zero",
            AllocationError::InvalidAlignment => "alignment must be a non-zero power of two",
            AllocationError::UpperAddressUnsupported => {
                "upper-address allocation is not supported by this allocator"
            }
            AllocationError::OutOfSpace => "no free region satisfies the allocation request",
        };
        f.write_str(msg)
    }
}

impl core::error::Error for AllocationError {}

/// Error returned when [`Suballocator::new`] / [`VirtualBlock::new`] arguments are
/// invalid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreateError {
    /// The block size was zero.
    ZeroSize,
    /// The granularity was zero. Pass `1` to disable granularity handling.
    ZeroGranularity,
    /// The granularity was greater than 1 but not a power of two.
    GranularityNotPowerOfTwo,
    /// The debug margin was greater than zero but not a multiple of four. Debug
    /// margins hold [`u32`] magic values, so they must be a whole number of `u32`s.
    DebugMarginNotMultipleOfFour,
    /// Buffer-image granularity page tracking would require an unreasonably large
    /// number of per-page records for this `(size, granularity)` pair.
    ///
    /// Page tracking (granularity greater than 256 on a non-virtual block) allocates
    /// one record per granularity page, i.e. `ceil(size / granularity)` records. This
    /// is capped so a hostile pair (e.g. a near-`u64::MAX` size with a small
    /// granularity) cannot exhaust memory (aborting the process) or, on a 32-bit
    /// target, silently truncate the page count. See the granularity module's
    /// `MAX_TRACKED_PAGES`. Use a coarser granularity or a smaller block.
    GranularityTrackingTooLarge,
}

impl core::fmt::Display for CreateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            CreateError::ZeroSize => "block size must be non-zero",
            CreateError::ZeroGranularity => "granularity must be non-zero (pass 1 to disable)",
            CreateError::GranularityNotPowerOfTwo => {
                "granularity greater than 1 must be a power of two"
            }
            CreateError::DebugMarginNotMultipleOfFour => {
                "debug margin must be a multiple of four"
            }
            CreateError::GranularityTrackingTooLarge => {
                "buffer-image granularity page tracking would require too many per-page records; use a coarser granularity or smaller block"
            }
        };
        f.write_str(msg)
    }
}

impl core::error::Error for CreateError {}

/// Error returned when an [`AllocationHandle`] passed to [`free`](Suballocator::free),
/// [`allocation_offset`](Suballocator::allocation_offset),
/// [`allocation_info`](Suballocator::allocation_info), or
/// [`set_user_data`](Suballocator::set_user_data) is not a valid, live handle for the
/// allocator it was given to.
///
/// A handle is *stale* once it has been freed (or the allocator was
/// [`clear`](Suballocator::clear)ed), and it is *foreign* if it came from a different
/// allocator instance. Both cases — including a double free — are reported here rather
/// than corrupting allocator state or panicking.
///
/// # Detection guarantees
///
/// Detection holds identically in debug and release builds (there is no debug-only
/// assertion path). It is **reliable for stale handles within a single allocator** but
/// only **best-effort across allocator instances**: a handle must be used with the
/// allocator that issued it. Passing a handle to a *different* allocator is memory-safe
/// and leaves both allocators internally consistent, but the operation may silently act
/// on the wrong allocation (see below) rather than being reported as an error.
///
/// - [`Tlsf`] packs a per-node generation counter into the handle (see
///   [`AllocationHandle`]). A freed, recycled, or cleared node bumps its generation, so
///   a stale handle fails the generation check. The one guaranteed miss for stale
///   handles is generation *wrap* (ABA): a specific node slot would have to be freed and
///   reused exactly `2^32` times for a stale handle to a prior occupant to alias a
///   current one. Cross-instance detection is only probabilistic: two allocators both
///   start with node 0 at generation 0, so one allocator will accept the other's first
///   handle and act on *its own* allocation at that node index. This is memory-safe but
///   operates on the wrong allocation; do not mix handles between allocators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandleError {
    /// The handle does not identify a live allocation in this allocator. It has been
    /// freed, the allocator was cleared, or the handle belongs to a different
    /// allocator (double frees and stale handles both surface here).
    InvalidHandle,
}

impl core::fmt::Display for HandleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("allocation handle is stale, already freed, or from another allocator")
    }
}

impl core::error::Error for HandleError {}

/// An outcome of [`Suballocator::create_allocation_request`]: an opaque, committed
/// plan for a single allocation that [`Suballocator::alloc`] can turn into a real
/// allocation.
///
/// This is the query half of the split-phase allocation API (VMA's
/// `VmaAllocationRequest`). It carries the algorithm-private information needed to
/// commit the allocation without redoing the search — e.g. for [`Tlsf`], the block
/// node that was found and the aligned offset within it. Callers should treat it as
/// opaque and pass it straight to [`alloc`](Suballocator::alloc).
///
/// A request is only valid for the allocator that produced it and only until that
/// allocator is next mutated. The split exists so a pool layer can probe several
/// blocks and only commit to one.
#[derive(Clone, Copy, Debug)]
pub struct AllocationRequest {
    /// The final offset the allocation will occupy (already aligned and
    /// granularity-adjusted). Exposed so a pool layer can compare candidate blocks.
    pub offset: u64,
    /// The usable size of the allocation (not counting debug margin).
    pub size: u64,
    /// Algorithm-private payload. Opaque to callers.
    pub(crate) payload: RequestPayload,
}

/// Algorithm-private data carried inside an [`AllocationRequest`].
///
/// This replaces the C++ `void* customData` / `uint64_t algorithmData` opaque pair
/// with a real, type-checked enum.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RequestPayload {
    /// [`Tlsf`] stashes the free block node it found and the type of the allocation.
    Tlsf {
        block: u32,
        alloc_type: AllocationType,
    },
}

/// The core suballocation interface, mirroring VMA's `VmaBlockMetadata` /
/// D3D12MA's `BlockMetadata` virtual interface.
///
/// A `Suballocator` manages the free/used regions of a single fixed-size block. It
/// is generic over the per-allocation user data `T` (which replaces the C++
/// `void*`). Implementations: [`Tlsf`].
///
/// # Split-phase allocation
///
/// Allocation is two-phase so a higher layer can probe several blocks before
/// committing to one:
///
/// 1. [`create_allocation_request`](Self::create_allocation_request) *queries* for a
///    placement. It performs no lasting mutation (with the sole exception, for
///    [`Tlsf`], of an internal free-list reordering that is purely a performance
///    optimization and preserves all invariants).
/// 2. [`alloc`](Self::alloc) *commits* a request produced in step 1, returning a
///    handle.
///
/// The convenience path is to call both back to back; [`VirtualBlock::allocate`]
/// does exactly that.
pub trait Suballocator<T>: Sized {
    /// Creates a new suballocator managing a block of `size` units.
    ///
    /// - `granularity`: the buffer-image granularity. Pass `1` to disable
    ///   granularity handling entirely. Values greater than 1 must be a power of
    ///   two.
    /// - `is_virtual`: `true` for a pure virtual block (disables granularity and
    ///   forces `debug_margin` to 0, matching VMA's `GetDebugMargin`), and — for
    ///   [`Tlsf`] — selects the finer 32-way small-size bucketing.
    /// - `debug_margin`: units of reserved padding placed after every allocation, for
    ///   later corruption detection by a memory-touching layer. Must be 0 or a
    ///   multiple of four. Forced to 0 when `is_virtual`.
    ///
    /// # Errors
    ///
    /// Returns [`CreateError`] if the arguments are invalid (zero size, bad
    /// granularity, or a debug margin that is not a multiple of four).
    fn new(
        size: u64,
        granularity: u64,
        is_virtual: bool,
        debug_margin: u64,
    ) -> Result<Self, CreateError>;

    /// The total size of the managed block.
    fn size(&self) -> u64;

    /// Whether this is a virtual block (granularity disabled, debug margin forced
    /// off, finer small-size bucketing for [`Tlsf`]).
    fn is_virtual(&self) -> bool;

    /// Validates every internal invariant.
    ///
    /// Returns `Ok(())` if the structure is consistent, or `Err` with a short
    /// description of the first violated invariant. Used heavily by tests; a correct
    /// program never observes an error here.
    fn validate(&self) -> Result<(), &'static str>;

    /// The number of live allocations.
    fn allocation_count(&self) -> usize;

    /// The number of free regions (including the trailing free region).
    fn free_regions_count(&self) -> usize;

    /// The total number of free units (the sum of the sizes of all free regions).
    fn sum_free_size(&self) -> u64;

    /// Whether the block is empty (has no live allocations).
    fn is_empty(&self) -> bool {
        self.allocation_count() == 0
    }

    /// The offset of the allocation identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] if `handle` is stale, already freed, or from another
    /// allocator. See [`HandleError`] for detection guarantees.
    fn allocation_offset(&self, handle: AllocationHandle) -> Result<u64, HandleError>;

    /// Full information about the allocation identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] if `handle` is stale, already freed, or from another
    /// allocator.
    fn allocation_info(&self, handle: AllocationHandle) -> Result<AllocationInfo<T>, HandleError>
    where
        T: Clone;

    /// Replaces the user data of the allocation identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] if `handle` is stale, already freed, or from another
    /// allocator. On error the allocator is left unchanged.
    fn set_user_data(&mut self, handle: AllocationHandle, user_data: T) -> Result<(), HandleError>;

    /// Returns a handle to the first allocation for iteration, or `None` if empty.
    ///
    /// Iteration order is by *descending* offset (matching VMA, which walks
    /// `prevPhysical` from the trailing block). Used by defragmentation.
    fn allocation_list_begin(&self) -> Option<AllocationHandle>;

    /// Returns the next allocation after `prev` in iteration order, or `None` at the
    /// end. See [`allocation_list_begin`](Self::allocation_list_begin).
    ///
    /// A stale, freed, or foreign `prev` handle is treated as the end of iteration and
    /// returns `None` (never panics); see [`HandleError`] for the detection guarantees.
    fn next_allocation(&self, prev: AllocationHandle) -> Option<AllocationHandle>;

    /// Returns the size of the free region immediately preceding `alloc` (lower
    /// offset), or 0 if that neighbour is not free. Used by defragmentation.
    ///
    /// A stale, freed, or foreign `alloc` handle returns 0 (never panics); see
    /// [`HandleError`] for the detection guarantees.
    fn next_free_region_size(&self, alloc: AllocationHandle) -> u64;

    /// Accumulates basic statistics into `stats`.
    fn add_statistics(&self, stats: &mut Statistics);

    /// Accumulates detailed statistics into `stats`.
    fn add_detailed_statistics(&self, stats: &mut DetailedStatistics);

    /// Query phase: tries to find a placement for the allocation described by `desc`.
    ///
    /// See the [trait docs](Self#split-phase-allocation) for the two-phase contract, and
    /// [`AllocationDesc`] for the request fields (size, alignment, resource type,
    /// upper-address, strategy).
    ///
    /// # Errors
    ///
    /// - [`AllocationError::InvalidSize`] if [`desc.size`](AllocationDesc::size) is `0`.
    /// - [`AllocationError::InvalidAlignment`] if [`desc.alignment`](AllocationDesc::alignment)
    ///   is not a power of two.
    /// - [`AllocationError::UpperAddressUnsupported`] if
    ///   [`desc.upper_address`](AllocationDesc::upper_address) is set on an algorithm that
    ///   does not support it ([`Tlsf`] does not).
    /// - [`AllocationError::OutOfSpace`] if no free region fits.
    fn create_allocation_request(
        &mut self,
        desc: AllocationDesc,
    ) -> Result<AllocationRequest, AllocationError>;

    /// Commit phase: turns a [`AllocationRequest`] from
    /// [`create_allocation_request`](Self::create_allocation_request) into a real
    /// allocation, associating `user_data` with it, and returns its handle.
    ///
    /// The request must have been produced by *this* allocator and the allocator must
    /// not have been mutated in between.
    fn alloc(&mut self, request: AllocationRequest, user_data: T) -> AllocationHandle;

    /// Frees the allocation identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] if `handle` is stale, already freed (double free), or
    /// from another allocator. On error nothing is freed and all accounting is left
    /// unchanged. See [`HandleError`] for detection guarantees.
    fn free(&mut self, handle: AllocationHandle) -> Result<(), HandleError>;

    /// Frees all allocations, returning the block to its initial empty state.
    ///
    /// Handles obtained before the call become stale; passing one to a handle-taking
    /// method afterwards returns [`HandleError`].
    fn clear(&mut self);
}
