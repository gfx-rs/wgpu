//! The [`Pool`] block vector and its supporting types.
//!
//! Port of `VmaBlockVector` (vk_mem_alloc.h) and `D3D12MA::BlockVector`
//! (D3D12MemAlloc.cpp). See the crate-level docs for provenance and fidelity notes.

use alloc::vec::Vec;

use wgpu_offset_allocator::{
    Algorithm, AllocationDesc, AllocationError, AllocationHandle, AllocationType, CreateError,
    DetailedStatistics, HandleError, Statistics, Strategy, Suballocator, Tlsf,
};

/// `NEW_BLOCK_SIZE_SHIFT_MAX` in VMA / D3D12MA: the ramp halves the preferred block
/// size at most this many times (giving first blocks of 1/8, 1/4, 1/2, then full).
const NEW_BLOCK_SIZE_SHIFT_MAX: u32 = 3;

/// `align_up(value, alignment)` for a power-of-two `alignment`, saturating to `u64::MAX`
/// instead of overflowing. `align_up(v, 1) == v` and `align_up(0, a) == 0`.
#[inline]
fn align_up_saturating(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    match value.checked_add(alignment - 1) {
        Some(sum) => sum & !(alignment - 1),
        None if value & (alignment - 1) == 0 => value,
        None => u64::MAX,
    }
}

/// A stable, caller-visible identifier for a block within a [`Pool`].
///
/// Unlike a block's position in the pool's internal vector — which is reordered as
/// blocks are incrementally sorted by free size — a `BlockId` never changes for the
/// life of the block and is never reused, so a caller can key resource bindings on it.
///
/// # Structure
///
/// A `BlockId` is a structured, opaque key (not an opaque monotonic counter as VMA's
/// `m_NextBlockId` was): it carries the owning pool's **full** [`PoolConfig::pool_salt`]
/// (all 64 bits) plus a per-pool monotonic counter. The salt lets a pool reject a
/// `BlockId` (inside an [`Allocation`]) that was minted by a *different* pool constructed
/// with a distinct salt: because the *whole* `u64` salt is compared, two pools with
/// distinct salts reject each other's ids **deterministically**, with no possibility of a
/// tag collision — see [`PoolConfig::pool_salt`] and [`Pool::free`].
///
/// The fields are private: a `BlockId` is only meaningful as an opaque, `Copy` map key.
/// Callers must not synthesize one (there is no public constructor); use the value handed
/// back by [`Pool::allocate`] / [`Pool::report`] / [`BlockBackend::create_block`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId {
    /// The owning pool's full [`PoolConfig::pool_salt`]. Compared in full by
    /// [`Pool::owns`] so distinct salts reject deterministically.
    salt: u64,
    /// The per-pool monotonic counter (`m_NextBlockId`), unique within a salt and never
    /// reused for the life of the pool.
    counter: u32,
}

impl BlockId {
    /// Builds an id carrying the owning pool's full `salt` and its per-pool `counter`.
    #[inline]
    fn new(salt: u64, counter: u32) -> Self {
        BlockId { salt, counter }
    }

    /// Test-only: the id a pool built with `pool_salt` would mint for `counter`.
    #[cfg(test)]
    pub(crate) fn new_for_test(pool_salt: u64, counter: u32) -> Self {
        BlockId::new(pool_salt, counter)
    }
}

/// The caller-implemented abstraction over device memory backing a [`Pool`]'s blocks.
///
/// The pool asks the backend to [create](Self::create_block) a block of a given size
/// (returning an opaque [`Block`](Self::Block) the caller can bind resources against)
/// and to [destroy](Self::destroy_block) one when the pool releases it. The pool never
/// allocates or frees device memory itself.
///
/// Implementations are single-threaded from the pool's perspective: every call happens
/// while the caller holds whatever lock guards the pool.
pub trait BlockBackend {
    /// The opaque per-block value, e.g. a `vk::DeviceMemory` wrapper or an
    /// `ID3D12Heap` wrapper. The pool stores it and hands it back on destruction.
    type Block;

    /// The error type returned by [`create_block`](Self::create_block) on failure
    /// (e.g. out of device memory). It is surfaced to the caller as
    /// [`PoolAllocError::Backend`] only when *no* smaller block could be created
    /// either; an intermediate failure just drives the halving retry.
    type Error;

    /// Creates a new block of exactly `size` units, tagged with the stable
    /// `block_id` the pool has assigned it.
    ///
    /// Returning `Err` drives the pool's block-size halving retry (VMA
    /// `AllocatePage`): the pool will try progressively smaller sizes (down to the
    /// request size) before giving up. The `block_id` is provided so a backend may key
    /// bookkeeping on it; it is unique and never reused.
    fn create_block(&mut self, size: u64, block_id: BlockId) -> Result<Self::Block, Self::Error>;

    /// Destroys a block previously returned by [`create_block`](Self::create_block),
    /// releasing its device memory. `block_id` is the id that block was created with.
    ///
    /// Called when the pool drops an empty block (hysteresis) and for every remaining
    /// block when the pool is consumed by [`Pool::into_blocks`]. The pool guarantees
    /// the block has no live allocations when this is called.
    fn destroy_block(&mut self, block: Self::Block, block_id: BlockId);
}

/// Per-pool policy, fixed at [`Pool::new`].
///
/// Mirrors the constructor arguments of `VmaBlockVector` / `D3D12MA::BlockVector`. The
/// caller is responsible for computing `preferred_block_size` — VMA uses
/// `heap <= 1 GiB ? heap / 8 : 256 MiB`, D3D12MA uses 64 MiB — and this crate imposes
/// no default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolConfig {
    /// Which suballocation algorithm each block uses (`m_Algorithm`). Currently only
    /// [`Algorithm::Tlsf`] is supported.
    pub algorithm: Algorithm,
    /// The preferred (and, with [`explicit_block_size`](Self::explicit_block_size),
    /// exact) size of a newly created block (`m_PreferredBlockSize`).
    pub preferred_block_size: u64,
    /// The minimum number of blocks the pool keeps alive (`m_MinBlockCount`). These
    /// are created eagerly at [`Pool::new`] and never dropped by the hysteresis.
    pub min_block_count: usize,
    /// The maximum number of blocks the pool may create (`m_MaxBlockCount`). Must be
    /// `>= min_block_count` and `>= 1`.
    pub max_block_count: usize,
    /// When `true`, every block is created at exactly `preferred_block_size` — the
    /// `1/8 -> 1/4 -> 1/2` ramp and the failure-halving retry are both disabled
    /// (`m_ExplicitBlockSize`).
    pub explicit_block_size: bool,
    /// A floor applied to the alignment of every allocation as
    /// `max(alignment, min_allocation_alignment)` (`m_MinAllocationAlignment`). Pass
    /// `1` (or `0`, treated as `1`) to disable.
    pub min_allocation_alignment: u64,
    /// Buffer-image granularity passed through to each block's suballocator. Pass `1`
    /// to disable granularity handling. Values greater than 1 must be a power of two.
    pub granularity: u64,
    /// Debug-margin units reserved after every allocation, passed through to each
    /// block's suballocator. Must be `0` or a multiple of four.
    pub debug_margin: u64,
    /// When `true`, [`allocate`](Pool::allocate) performs VMA's two-pass affinity
    /// clustering: it first scans only blocks whose affinity tag matches the request's
    /// preferred tag, then all blocks. When `false`, the request's preferred tag is
    /// ignored and a single pass runs.
    pub affinity_clustering: bool,
    /// A caller-supplied value that scopes this pool's [`BlockId`]s to this pool.
    ///
    /// Every [`BlockId`] this pool mints carries this full `pool_salt` (all 64 bits; see
    /// [`BlockId`]). [`free`](Pool::free) and
    /// [`set_block_affinity`](Pool::set_block_affinity) reject any id whose salt does not
    /// match this pool's, so an [`Allocation`] handed to the *wrong* pool is refused
    /// rather than acted on.
    ///
    /// # Guarantee
    ///
    /// - **Distinct salts (recommended).** Two pools constructed with *different*
    ///   `pool_salt` values reject each other's allocations **deterministically**: a
    ///   foreign [`Allocation`]'s `block_id` carries the other pool's salt, so the
    ///   full-width salt comparison fails and [`free`](Pool::free) returns
    ///   [`FreeError::InvalidAllocation`] without touching any live allocation. Because
    ///   the *entire* `u64` salt is compared (not a hashed-down tag), no pair of distinct
    ///   salts can ever collide. This is the intended usage: give every pool in the
    ///   process a distinct `pool_salt` (e.g. from a global counter), so cross-pool
    ///   mix-ups are always caught.
    /// - **Shared salt (e.g. both default `0`).** Two pools sharing a `pool_salt` mint
    ///   colliding [`BlockId`]s, so cross-pool detection falls back to *best effort*:
    ///   the underlying suballocator's generation-tagged handle usually still rejects a
    ///   foreign allocation, but this inherits the offset-allocator's documented
    ///   cross-instance limitation — two fresh pools both start at counter `0`,
    ///   generation `0`, so a foreign allocation whose handle happens to match a live
    ///   local one can be accepted. Give each pool a distinct salt to avoid this.
    ///
    /// The [default](Self::default) is `0`.
    pub pool_salt: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            algorithm: Algorithm::Tlsf,
            preferred_block_size: 0,
            min_block_count: 0,
            max_block_count: 1,
            explicit_block_size: false,
            min_allocation_alignment: 1,
            granularity: 1,
            debug_margin: 0,
            affinity_clustering: false,
            pool_salt: 0,
        }
    }
}

/// A live allocation handed out by [`Pool::allocate`].
///
/// It is fully opaque: it carries everything the caller needs to bind a resource — which
/// block ([`block_id`](Self::block_id)), and where in it ([`offset`](Self::offset) /
/// [`size`](Self::size)) — plus a private suballocator handle that identifies the
/// allocation for [`Pool::free`]. Read the placement through the accessor methods; to free,
/// pass the whole [`Allocation`] back to [`free`](Pool::free).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Allocation {
    /// The stable id of the block this allocation lives in.
    block_id: BlockId,
    /// The offset of the allocation within its block.
    offset: u64,
    /// The usable size of the allocation (not counting any debug margin). May exceed
    /// the requested size, as the underlying suballocator may round up.
    size: u64,
    /// The suballocator handle that identifies this allocation for [`Pool::free`]. Kept
    /// private (there is no accessor): it is only meaningful to the pool that issued it.
    handle: AllocationHandle,
}

impl Allocation {
    /// The stable [`BlockId`] of the block this allocation lives in.
    #[must_use]
    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    /// The offset of the allocation within its block.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The usable size of the allocation, in units (not counting any debug margin).
    ///
    /// May exceed the requested size, as the underlying suballocator may round up.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Test-only: a copy of this allocation with its block id replaced, for exercising
    /// the unknown-block / wrong-pool rejection paths in [`Pool::free`].
    #[cfg(test)]
    pub(crate) fn with_block_id_for_test(self, block_id: BlockId) -> Self {
        Allocation { block_id, ..self }
    }
}

/// Per-call environment for [`Pool::allocate`] that the pool deliberately does not
/// own (VMA / D3D12MA read the equivalents from global allocator state).
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocationContext {
    /// Remaining budget, in units, for the memory type/segment this pool draws from.
    /// `None` means unlimited or unknown (D3D12MA's `UINT64_MAX` for non-standard heaps
    /// / VMA when no budget is configured).
    ///
    /// Used only to gate *new block creation*: it never prevents reusing space in an
    /// existing block.
    pub budget_free_bytes: Option<u64>,
    /// Whether the caller can fall back to a dedicated/committed allocation if the
    /// pool declines to grow (VMA `canFallbackToDedicated` / D3D12MA `committedAllowed`).
    ///
    /// When `true` and the budget cannot fit a new block, the pool returns
    /// [`PoolAllocError::ShouldDedicate`] instead of creating a block. When `false`,
    /// the pool may exceed the budget to create a block (the caller has nowhere else
    /// to go).
    pub dedicated_fallback_allowed: bool,
    /// The request's preferred block affinity tag, used only when
    /// [`PoolConfig::affinity_clustering`] is enabled. `None` disables clustering for
    /// this call (single pass).
    pub preferred_affinity: Option<bool>,
}

/// Per-call environment for [`Pool::free`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeContext {
    /// Whether the memory type/segment's budget is currently exceeded (VMA /
    /// D3D12MA `budgetExceeded`). When `true`, a block that becomes empty is destroyed
    /// eagerly (subject to `min_block_count`) rather than retained by the hysteresis.
    pub budget_exceeded: bool,
}

/// Error from [`Pool::allocate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolAllocError<E> {
    /// The request can never be served from this pool's blocks and the caller should
    /// make a dedicated/committed allocation instead. Two cases collapse here, matching
    /// VMA / D3D12MA:
    ///
    /// - the request's footprint (its granularity-rounded size plus one debug margin) is
    ///   larger than the biggest block this pool can create
    ///   (`footprint > effective_max_block_size`); or
    /// - the pool would need to grow, the budget cannot fit a new block, and the caller
    ///   allowed a dedicated fallback ([`AllocationContext::dedicated_fallback_allowed`]).
    ShouldDedicate,
    /// The request is malformed: zero size, or a non-power-of-two alignment (after the
    /// `min_allocation_alignment` floor).
    InvalidRequest,
    /// Upper-address (top-down) allocation was requested, which this pool does not
    /// support (it is a linear-only, double-stack feature).
    UpperAddressUnsupported,
    /// Every existing block is full and the pool cannot create a new one — it is at
    /// `max_block_count`, or all block-creation sizes failed. The caller decides
    /// whether to fall back to a dedicated allocation or the next memory type.
    OutOfPoolMemory,
    /// The backend failed to create a block and no smaller block could be created
    /// either. Carries the backend's error.
    Backend(E),
}

impl<E: core::fmt::Display> core::fmt::Display for PoolAllocError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PoolAllocError::ShouldDedicate => {
                f.write_str("request cannot fit a pool block; use a dedicated allocation")
            }
            PoolAllocError::InvalidRequest => {
                f.write_str("allocation size must be non-zero and alignment a power of two")
            }
            PoolAllocError::UpperAddressUnsupported => {
                f.write_str("upper-address allocation is not supported")
            }
            PoolAllocError::OutOfPoolMemory => {
                f.write_str("all blocks full and the pool cannot grow")
            }
            PoolAllocError::Backend(e) => write!(f, "backend failed to create a block: {e}"),
        }
    }
}

impl<E: core::fmt::Display + core::fmt::Debug> core::error::Error for PoolAllocError<E> {}

/// Error from [`Pool::free`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeError {
    /// The [`Allocation`] did not identify a live allocation in this pool, so nothing
    /// was freed and pool state is unchanged. Any of: the `block_id`'s salt names a
    /// *different* pool (see [`PoolConfig::pool_salt`]; detected deterministically when
    /// pools use distinct salts); the `block_id` names no current block; or the `handle`
    /// is stale/already-freed. See [`Pool::free`] for the full breakdown.
    InvalidAllocation,
}

impl core::fmt::Display for FreeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("allocation is stale, already freed, or from another pool")
    }
}

impl core::error::Error for FreeError {}

/// The outcome of a successful [`Pool::free`].
///
/// If the free emptied a block that the hysteresis decided to drop, the block's
/// backend value is returned so the caller can release its device memory. The caller
/// *must* pass it to [`BlockBackend::destroy_block`] (or otherwise release it) — the
/// pool has already forgotten it.
#[derive(Debug)]
#[must_use = "a destroyed block's memory must be released by the caller"]
pub struct FreeOutcome<Block> {
    /// The block dropped by this free, if any: its backend value and stable id. `None`
    /// if no block was destroyed.
    pub destroyed_block: Option<(Block, BlockId)>,
}

/// A per-block entry in a [`PoolReport`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockReport {
    /// The block's stable id.
    pub block_id: BlockId,
    /// The block's total size in units.
    pub size: u64,
    /// The number of live allocations in the block.
    pub allocation_count: usize,
    /// The total free units in the block.
    pub free_bytes: u64,
    /// The caller-set affinity tag.
    pub affinity: bool,
}

/// A snapshot report of a [`Pool`], sufficient to build a higher-level allocator
/// report (e.g. wgpu's `wgt::AllocatorReport`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolReport {
    /// One entry per block, in the pool's current internal order.
    pub blocks: Vec<BlockReport>,
    /// Aggregate statistics across all blocks.
    pub statistics: Statistics,
}

/// One suballocator. This is the pool's own equivalent of the offset-allocator's
/// `VirtualBlock` inner wrapper, but for *non* virtual blocks (granularity and debug
/// margin enabled).
#[derive(Debug)]
enum BlockSuballocator<T> {
    Tlsf(Tlsf<T>),
}

impl<T> BlockSuballocator<T> {
    fn new(
        algorithm: Algorithm,
        size: u64,
        granularity: u64,
        debug_margin: u64,
    ) -> Result<Self, CreateError> {
        // is_virtual = false: real blocks track granularity and debug margins.
        let Algorithm::Tlsf = algorithm;
        Ok(BlockSuballocator::Tlsf(Tlsf::new(
            size,
            granularity,
            false,
            debug_margin,
        )?))
    }

    fn size(&self) -> u64 {
        match self {
            BlockSuballocator::Tlsf(t) => t.size(),
        }
    }

    fn sum_free_size(&self) -> u64 {
        match self {
            BlockSuballocator::Tlsf(t) => t.sum_free_size(),
        }
    }

    fn allocation_count(&self) -> usize {
        match self {
            BlockSuballocator::Tlsf(t) => t.allocation_count(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            BlockSuballocator::Tlsf(t) => t.is_empty(),
        }
    }

    /// Convenience over the split-phase API: query + commit in one shot, returning the
    /// handle and its offset and committed size. Mirrors `VirtualBlock::allocate`.
    fn allocate(
        &mut self,
        desc: AllocationDesc,
        user_data: T,
    ) -> Result<(AllocationHandle, u64, u64), AllocationError> {
        match self {
            BlockSuballocator::Tlsf(t) => {
                let req = t.create_allocation_request(desc)?;
                let (offset, committed) = (req.offset, req.size);
                Ok((t.alloc(req, user_data), offset, committed))
            }
        }
    }

    fn free(&mut self, handle: AllocationHandle) -> Result<(), HandleError> {
        match self {
            BlockSuballocator::Tlsf(t) => t.free(handle),
        }
    }

    fn add_statistics(&self, stats: &mut Statistics) {
        match self {
            BlockSuballocator::Tlsf(t) => t.add_statistics(stats),
        }
    }

    fn add_detailed_statistics(&self, stats: &mut DetailedStatistics) {
        match self {
            BlockSuballocator::Tlsf(t) => t.add_detailed_statistics(stats),
        }
    }

    fn validate(&self) -> Result<(), &'static str> {
        match self {
            BlockSuballocator::Tlsf(t) => t.validate(),
        }
    }

    /// Iterates live allocations, invoking `f(offset, size, &user_data)` for each.
    fn for_each_allocation<F: FnMut(u64, u64, &T)>(&self, mut f: F)
    where
        T: Clone,
    {
        match self {
            BlockSuballocator::Tlsf(t) => Self::iterate(t, &mut f),
        }
    }

    fn iterate<S: Suballocator<T>, F: FnMut(u64, u64, &T)>(s: &S, f: &mut F)
    where
        T: Clone,
    {
        let mut cur = s.allocation_list_begin();
        while let Some(h) = cur {
            if let Ok(info) = s.allocation_info(h) {
                f(info.offset, info.size, &info.user_data);
            }
            cur = s.next_allocation(h);
        }
    }
}

/// A block in the pool: the caller's backend value, its suballocator, and its metadata.
struct BlockEntry<B: BlockBackend, T> {
    /// The caller's opaque backend value. `Option` so it can be moved out on
    /// destruction without leaving the entry in an invalid state; it is always `Some`
    /// while the entry is in `blocks`.
    block: Option<B::Block>,
    suballocator: BlockSuballocator<T>,
    id: BlockId,
    /// The caller-set affinity tag (VMA's `IsMapped`, generalized). Defaults to
    /// `false`.
    affinity: bool,
}

// Manual `Debug` so the pool is `Debug` without requiring `B::Block: Debug` (the
// caller's opaque device-memory value need not be printable). The block value is
// elided; its size and id are visible via the suballocator and `id`.
impl<B: BlockBackend, T: core::fmt::Debug> core::fmt::Debug for BlockEntry<B, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlockEntry")
            .field("id", &self.id)
            .field("suballocator", &self.suballocator)
            .field("affinity", &self.affinity)
            .field("has_block", &self.block.is_some())
            .finish()
    }
}

/// A growing/shrinking set of memory blocks, each backed by one suballocator.
///
/// See the [crate docs](crate) for the model. Generic over the caller's
/// [`BlockBackend`] `B` and the per-allocation user-data type `T` (which the
/// suballocator stores alongside each allocation and the [report](Pool::report)
/// surfaces back).
///
/// `T` must be [`Clone`] for allocation and reporting because the suballocator stores
/// user data by value and the report reads it by value. Use a cheap-to-clone type such
/// as an index, id, or `()` if you do not need per-allocation data.
pub struct Pool<B: BlockBackend, T> {
    config: PoolConfig,
    blocks: Vec<BlockEntry<B, T>>,
    /// Monotonic per-pool counter for the next [`BlockId`] (`m_NextBlockId`). Never
    /// reused. (The salt each id carries is [`PoolConfig::pool_salt`], read from
    /// `config`.)
    next_counter: u32,
    /// The largest block size the pool will ever create, precomputed at construction:
    /// `preferred_block_size` (the ramp only ever produces *smaller* blocks). Used for
    /// the early size reject.
    effective_max_block_size: u64,
}

// Manual `Debug` so the pool is `Debug` without requiring `B::Block: Debug`.
impl<B: BlockBackend, T: core::fmt::Debug> core::fmt::Debug for Pool<B, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pool")
            .field("config", &self.config)
            .field("blocks", &self.blocks)
            .field("next_counter", &self.next_counter)
            .field("effective_max_block_size", &self.effective_max_block_size)
            .finish()
    }
}

impl<B: BlockBackend, T> Pool<B, T> {
    /// Creates a pool with the given `config`, eagerly creating `min_block_count`
    /// blocks at `preferred_block_size` via `backend` (VMA `CreateMinBlocks`).
    ///
    /// # Errors
    ///
    /// - [`PoolAllocError::InvalidRequest`] if the config is inconsistent:
    ///   `preferred_block_size == 0`, `max_block_count == 0`,
    ///   `max_block_count < min_block_count`, or a suballocator-rejected
    ///   `granularity` / `debug_margin`.
    /// - [`PoolAllocError::Backend`] if creating one of the minimum blocks fails; any
    ///   blocks already created are handed back to `backend` before returning.
    pub fn new(config: PoolConfig, backend: &mut B) -> Result<Self, PoolAllocError<B::Error>> {
        if config.preferred_block_size == 0
            || config.max_block_count == 0
            || config.max_block_count < config.min_block_count
        {
            return Err(PoolAllocError::InvalidRequest);
        }
        // Validate the granularity / debug-margin by constructing (and discarding) one
        // suballocator up front, so a bad config fails cleanly here rather than on the
        // first allocation. Mirrors the C++ constructor asserting its arguments.
        BlockSuballocator::<T>::new(
            config.algorithm,
            config.preferred_block_size,
            config.granularity,
            config.debug_margin,
        )
        .map_err(|_| PoolAllocError::InvalidRequest)?;

        let mut pool = Pool {
            effective_max_block_size: config.preferred_block_size,
            config,
            blocks: Vec::new(),
            next_counter: 0,
        };

        for _ in 0..pool.config.min_block_count {
            let size = pool.config.preferred_block_size;
            if let Err(e) = pool.create_block(size, backend) {
                // Unwind: hand back every block we already created.
                pool.destroy_all(backend);
                return Err(e);
            }
        }

        Ok(pool)
    }

    /// The pool's configuration.
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// The number of blocks currently in the pool.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// The number of empty blocks currently in the pool (VMA `HasEmptyBlock`, counted).
    pub fn empty_block_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| b.suballocator.is_empty())
            .count()
    }

    /// Whether `block_id` was minted by *this* pool, i.e. it carries this pool's full
    /// [`PoolConfig::pool_salt`]. A foreign id (from a pool with a different salt) fails
    /// this check regardless of its counter, so it is never looked up. The salt is
    /// compared in full (all 64 bits), so distinct salts reject deterministically — there
    /// is no tag to collide. Pools *sharing* a salt pass this check and fall back to the
    /// block-vector/suballocator identity checks (see [`PoolConfig::pool_salt`]).
    #[inline]
    fn owns(&self, block_id: BlockId) -> bool {
        block_id.salt == self.config.pool_salt
    }

    /// Sets the affinity tag of the block with the given id (VMA marks a block mapped
    /// or unmapped). Returns `false` if no such block exists, or if `block_id` was
    /// minted by a different pool (its salt does not match this pool's).
    ///
    /// Only consulted when [`PoolConfig::affinity_clustering`] is enabled.
    pub fn set_block_affinity(&mut self, block_id: BlockId, affinity: bool) -> bool {
        if !self.owns(block_id) {
            return false;
        }
        match self.blocks.iter_mut().find(|b| b.id == block_id) {
            Some(entry) => {
                entry.affinity = affinity;
                true
            }
            None => false,
        }
    }

    /// Creates a new block of exactly `size` via the backend, pushing it onto the
    /// block vector with a fresh stable id. Port of VMA `CreateBlock`.
    fn create_block(&mut self, size: u64, backend: &mut B) -> Result<(), PoolAllocError<B::Error>> {
        // The per-pool counter is 32 bits; if it were ever exhausted (2^32 blocks minted
        // over the pool's life, never plausible for a GPU allocator) treat it as the
        // pool being unable to grow rather than wrapping and reusing an id.
        let counter = self.next_counter;
        let next = counter
            .checked_add(1)
            .ok_or(PoolAllocError::OutOfPoolMemory)?;
        let id = BlockId::new(self.config.pool_salt, counter);
        // Build the suballocator first; if it rejects the size (should not happen for a
        // size we chose), surface as InvalidRequest without asking the backend.
        let suballocator = BlockSuballocator::<T>::new(
            self.config.algorithm,
            size,
            self.config.granularity,
            self.config.debug_margin,
        )
        .map_err(|_| PoolAllocError::InvalidRequest)?;
        let block = backend
            .create_block(size, id)
            .map_err(PoolAllocError::Backend)?;
        self.next_counter = next;
        self.blocks.push(BlockEntry {
            block: Some(block),
            suballocator,
            id,
            affinity: false,
        });
        Ok(())
    }

    /// Hands every block back to the backend and clears the vector. Used on
    /// construction failure and by [`clear`](Self::clear).
    fn destroy_all(&mut self, backend: &mut B) {
        for mut entry in self.blocks.drain(..) {
            if let Some(block) = entry.block.take() {
                backend.destroy_block(block, entry.id);
            }
        }
    }

    /// Allocates the allocation described by `desc` from the pool, creating a new block
    /// if necessary and permitted.
    ///
    /// This is the port of `VmaBlockVector::AllocatePage` / `D3D12MA::BlockVector::
    /// AllocatePage`. See [`AllocationDesc`] for the request fields (size, alignment,
    /// resource type, upper-address, strategy) and [`AllocationContext`] for the meaning
    /// of `ctx`. `user_data` is stored with the allocation and surfaced by
    /// [`report`](Self::report); `backend` creates a block if the pool must grow.
    ///
    /// The pool applies its [`min_allocation_alignment`](PoolConfig::min_allocation_alignment)
    /// floor to [`desc.alignment`](AllocationDesc::alignment) before allocating.
    ///
    /// # Errors
    ///
    /// See [`PoolAllocError`] for the full taxonomy: `ShouldDedicate` (too large, or
    /// budget-gated), `OutOfPoolMemory` (full and cannot grow), `InvalidRequest`,
    /// `UpperAddressUnsupported`, and `Backend`.
    ///
    /// # Panics
    ///
    /// Never, for any caller input. The grow ramp floors every candidate block size at
    /// the request's footprint (granularity-rounded size plus one debug margin), so a
    /// freshly created block always places the request; if a footprint miscalculation
    /// ever broke that invariant, a debug build would trip a `debug_assert` and a release
    /// build would roll the block back and return an error rather than panic.
    pub fn allocate(
        &mut self,
        desc: AllocationDesc,
        ctx: AllocationContext,
        user_data: T,
        backend: &mut B,
    ) -> Result<Allocation, PoolAllocError<B::Error>>
    where
        T: Clone,
    {
        if desc.size == 0 {
            return Err(PoolAllocError::InvalidRequest);
        }
        // Apply the min-allocation-alignment floor (VMA 11694 / D3D12MA 8799), coercing
        // 0 to 1 first. The suballocator rejects non-power-of-two alignment.
        let desc = AllocationDesc {
            alignment: desc
                .alignment
                .max(self.config.min_allocation_alignment)
                .max(1),
            ..desc
        };
        if !desc.alignment.is_power_of_two() {
            return Err(PoolAllocError::InvalidRequest);
        }

        // Upper-address allocation is a linear-only, double-stack feature (VMA
        // 11756-11760); the TLSF pool never supports it.
        if desc.upper_address {
            return Err(PoolAllocError::UpperAddressUnsupported);
        }

        // Early reject: an allocation that can never fit a block, even the largest this
        // pool can create (VMA 11762-11766 / D3D12MA 8660-8664). VMA uses
        // `size + VMA_DEBUG_MARGIN > preferred`; we use the exact space the suballocator
        // charges for this request as the sole allocation of a fresh block (see
        // `request_footprint`): its granularity-rounded size plus one trailing debug
        // margin. This is the *same* footprint the grow ramp floors against, so the two
        // never disagree. Saturating so a near-u64::MAX size cannot overflow.
        let footprint = self.request_footprint(desc);
        if footprint > self.effective_max_block_size {
            return Err(PoolAllocError::ShouldDedicate);
        }

        // 1. Search existing blocks.
        if let Some(alloc) = self.try_existing_blocks(desc, &ctx, &user_data) {
            return Ok(alloc);
        }

        // 2. Try to create a new block. VMA 11747-11752 / D3D12MA 8674-8686.
        let can_fall_back = ctx.dedicated_fallback_allowed;
        let budget = ctx.budget_free_bytes;
        let can_create_new_block = self.blocks.len() < self.config.max_block_count;
        if can_create_new_block {
            // Budget gate: even if we don't strictly have to stay within budget, when
            // the budget cannot fit *this allocation* and the caller can fall back to a
            // dedicated allocation, decline to grow (D3D12MA 8683-8686 collapses to
            // E_OUTOFMEMORY, which the caller turns into a committed allocation).
            let budget_too_small = matches!(budget, Some(free) if free < footprint);
            if budget_too_small && can_fall_back {
                return Err(PoolAllocError::ShouldDedicate);
            }
            return self.grow_and_allocate(desc, budget, can_fall_back, user_data, backend);
        }

        // 3. All existing blocks full and cannot grow.
        Err(PoolAllocError::OutOfPoolMemory)
    }

    /// Scans existing blocks for a placement, respecting scan order and affinity
    /// clustering. Returns the committed [`Allocation`] on success. Does not create
    /// blocks.
    fn try_existing_blocks(
        &mut self,
        desc: AllocationDesc,
        ctx: &AllocationContext,
        user_data: &T,
    ) -> Option<Allocation>
    where
        T: Clone,
    {
        // Two-pass affinity clustering (VMA 11790-11822), generalized to the affinity
        // tag (see the crate-level fidelity notes): first only
        // blocks whose affinity matches the request's preferred tag, then the rest. When
        // clustering is off or no preference is given, a single pass over all blocks.
        let clustering = self.config.affinity_clustering && ctx.preferred_affinity.is_some();
        let passes = if clustering { 2u8 } else { 1u8 };

        // MinTime scans backward (largest free first, VMA 11842-11857); otherwise
        // forward (smallest free first, best packing, VMA 11804-11840).
        let min_time = matches!(desc.strategy, Strategy::MinTime);

        for pass in 0..passes {
            let count = self.blocks.len();
            for step in 0..count {
                let i = if min_time { count - 1 - step } else { step };
                if clustering {
                    // pass 0: matching affinity; pass 1: the rest.
                    let matches = Some(self.blocks[i].affinity) == ctx.preferred_affinity;
                    if (pass == 0) != matches {
                        continue;
                    }
                }
                if let Some(alloc) = self.alloc_from_block_index(i, desc, user_data) {
                    self.incrementally_sort_blocks();
                    return Some(alloc);
                }
            }
        }
        None
    }

    /// Attempts to allocate from the block at internal index `i`, returning the
    /// committed [`Allocation`] on success.
    fn alloc_from_block_index(
        &mut self,
        i: usize,
        desc: AllocationDesc,
        user_data: &T,
    ) -> Option<Allocation>
    where
        T: Clone,
    {
        // Skip a block too small to hold this request's footprint as its *sole*
        // allocation: it can never place the request regardless of current occupancy
        // (the footprint is the minimum viable block size; see `request_footprint`).
        // This mirrors the suballocator's own rejection.
        // `effective_max_block_size` (= preferred) can exceed a ramp-created block's actual
        // size, so the `allocate`-level early reject alone does not cover this per-block case.
        let footprint = self.request_footprint(desc);
        let entry = &mut self.blocks[i];
        if footprint > entry.suballocator.size() {
            return None;
        }
        let id = entry.id;
        match entry.suballocator.allocate(desc, user_data.clone()) {
            Ok((handle, offset, committed_size)) => Some(Allocation {
                block_id: id,
                offset,
                size: committed_size,
                handle,
            }),
            Err(_) => None,
        }
    }

    /// The **request footprint**: the smallest block size in which this request can be
    /// placed as the *sole* allocation of a freshly created (empty) block. Any block at
    /// least this large can always place the request; any smaller block never can.
    ///
    /// This is the single, coherent footprint used by both the early size reject and the
    /// grow-ramp floor, so the two can never disagree. It mirrors exactly
    /// what the suballocator charges for the first allocation of an empty block, derived
    /// end-to-end from the offset-allocator source (both placement paths):
    ///
    /// - **Granularity rounding.** For a "low" granularity in `(1, 256]` and a
    ///   conservative allocation type (the suballocator's `roundup_alloc_request` set:
    ///   `Unknown` / `ImageUnknown` / `ImageOptimal`; `granularity.rs:150-160`), the
    ///   usable size is rounded up to a multiple of the granularity. This rounding is
    ///   applied by TLSF's `create_allocation_request_impl` (`tlsf.rs:855-857`). For
    ///   `granularity > 256` the suballocator uses page tracking, which adds *no* footprint
    ///   for the first allocation of an empty block (all pages start free, so there is no
    ///   conflict bump); the alignment adds none either, since `align_up(0, alignment) == 0`.
    /// - **Lower-address debug margin (one margin).** The lower-address suballocator
    ///   reserves exactly one *trailing* `debug_margin` after the allocation
    ///   (`alloc_size += debug_margin` in TLSF `tlsf.rs:864`). A fresh block's first
    ///   allocation sits at offset 0 with no leading filler, so the footprint is
    ///   `rounded_size + debug_margin`.
    ///
    /// All arithmetic saturates, so a near-`u64::MAX` `size` yields `u64::MAX` (which the
    /// callers then reject) rather than overflowing.
    fn request_footprint(&self, desc: AllocationDesc) -> u64 {
        let margin = self.config.debug_margin;
        let size = desc.size;

        // Lower-address footprint: (granularity-rounded size) + one trailing margin.
        let granularity = self.config.granularity;
        let low_granularity = granularity > 1 && granularity <= 256;
        let conservative = matches!(
            desc.alloc_type,
            AllocationType::Unknown | AllocationType::ImageUnknown | AllocationType::ImageOptimal
        );
        let rounded = if low_granularity && conservative {
            align_up_saturating(size, granularity)
        } else {
            size
        };
        rounded.saturating_add(margin)
    }

    /// Computes the new block size (the `1/8 -> 1/4 -> 1/2 -> full` ramp), creates the
    /// block via the backend with failure-halving retry, then allocates from it. Port
    /// of the "2. Try to create new block" section of VMA / D3D12MA `AllocatePage`.
    ///
    /// Every candidate block size the ramp considers is floored at the request's
    /// [`request_footprint`], so a freshly created block can *never* fail to place the
    /// request: the churn where the ramp created a nominally-large-enough block
    /// that granularity rounding then made too small — creating and destroying a device
    /// block on every call — is eliminated at the root. The rollback of a
    /// created-but-unusable block is kept only as a defensive fallback (with a
    /// `debug_assert` that it is unreachable), preserving the invariant "the caller only
    /// ever holds blocks the pool told it about, and at most one empty block survives".
    fn grow_and_allocate(
        &mut self,
        desc: AllocationDesc,
        budget: Option<u64>,
        can_fall_back: bool,
        user_data: T,
        backend: &mut B,
    ) -> Result<Allocation, PoolAllocError<B::Error>>
    where
        T: Clone,
    {
        // The minimum viable block size: the exact space the suballocator charges for
        // this request as the sole allocation of a fresh block (a lower-address request's
        // granularity-rounded size + one trailing margin, or an upper-address request's
        // raw size + a margin at both ends; see `request_footprint`). A block at
        // least this large is *guaranteed* to place the request; a smaller one is
        // guaranteed to fail. This is the same footprint the early reject uses,
        // so the caller has already returned `ShouldDedicate` if `footprint >
        // effective_max_block_size` — every size we consider below is `<=
        // preferred_block_size` and `>= footprint`, so we never create a block that cannot
        // hold the request.
        let footprint = self.request_footprint(desc);
        debug_assert!(
            footprint <= self.effective_max_block_size,
            "grow_and_allocate reached with a footprint the early reject should have caught",
        );

        // Calculate the optimal size for the new block.
        let mut new_block_size = self.config.preferred_block_size;
        let mut shift: u32 = 0;

        if !self.config.explicit_block_size {
            // Allocate 1/8, 1/4, 1/2 as first blocks (VMA 11868-11885 / D3D12MA
            // 8717-8734). VMA compares the halved size against `size * 2`; we compare
            // against `max(footprint * 2, footprint)` so the ramp never steps below what
            // the request actually needs in a block (i.e. below `footprint`). Saturating
            // so the doubling cannot overflow.
            let max_existing = self.calc_max_block_size();
            let footprint_times_two = footprint.saturating_mul(2).max(footprint);
            while shift < NEW_BLOCK_SIZE_SHIFT_MAX {
                let smaller = new_block_size / 2;
                if smaller > max_existing && smaller >= footprint_times_two {
                    new_block_size = smaller;
                    shift += 1;
                } else {
                    break;
                }
            }
        }

        // A closure-free helper: budget permits creating a block of `sz`?
        // `newBlockSize <= freeMemory || !canFallbackToDedicated` (VMA 11888).
        let budget_permits = |sz: u64| match budget {
            Some(free) => sz <= free || !can_fall_back,
            None => true,
        };

        // Try `new_block_size`, then (unless explicit) 1/2, 1/4, 1/8 down to `>=
        // footprint` (VMA 11891-11908 / D3D12MA 8743-8760). Because every candidate is
        // `>= footprint`, placement on a freshly created block always succeeds; the
        // rollback branch is a defensive fallback that a correct program never reaches.
        // `last_backend_err` remembers the most recent create failure so it can be
        // surfaced if nothing works.
        let mut last_backend_err: Option<B::Error> = None;
        loop {
            if budget_permits(new_block_size) {
                match self.create_block(new_block_size, backend) {
                    Ok(()) => {
                        // Allocate from the freshly created (last) block.
                        let last = self.blocks.len() - 1;
                        if let Some(alloc) = self.alloc_from_block_index(last, desc, &user_data) {
                            self.incrementally_sort_blocks();
                            return Ok(alloc);
                        }
                        // Unreachable: a block floored at `footprint` can always place the
                        // request. Kept as a defense so a future footprint miscalculation
                        // degrades to an error (and a rolled-back block) rather than a
                        // leaked unusable block. Roll it back and stop — retrying smaller
                        // sizes would only churn, exactly what the footprint floor removes.
                        debug_assert!(
                            false,
                            "fresh block of size {new_block_size} (>= footprint {footprint}) \
                             failed to place request size {}; footprint miscalculated",
                            desc.size,
                        );
                        if let Some((block, id)) = self.remove_block(last) {
                            backend.destroy_block(block, id);
                        }
                        return Err(PoolAllocError::OutOfPoolMemory);
                    }
                    Err(PoolAllocError::Backend(e)) => last_backend_err = Some(e),
                    Err(other) => return Err(other),
                }
            }

            // Halve for the next attempt, if the ramp still has room and the smaller
            // size can still hold the request footprint.
            if self.config.explicit_block_size || shift >= NEW_BLOCK_SIZE_SHIFT_MAX {
                break;
            }
            let smaller = new_block_size / 2;
            if smaller < footprint {
                break;
            }
            new_block_size = smaller;
            shift += 1;
        }

        // No usable block could be created. If the backend actually failed, surface its
        // error; otherwise the budget gate leaves an ordinary out-of-pool-memory
        // condition.
        match last_backend_err {
            Some(e) => Err(PoolAllocError::Backend(e)),
            None if can_fall_back => Err(PoolAllocError::ShouldDedicate),
            None => Err(PoolAllocError::OutOfPoolMemory),
        }
    }

    /// Returns the maximum existing block size, capped at `preferred_block_size`. Port
    /// of VMA / D3D12MA `CalcMaxBlockSize`: scan from the back (blocks are sorted
    /// ascending by free size, so the largest blocks tend to be later), stopping once a
    /// block at least the preferred size is seen.
    fn calc_max_block_size(&self) -> u64 {
        let mut result = 0u64;
        for entry in self.blocks.iter().rev() {
            result = result.max(entry.suballocator.size());
            if result >= self.config.preferred_block_size {
                break;
            }
        }
        result
    }

    /// One bubble-sort step to keep blocks incrementally sorted ascending by free size.
    /// Port of `IncrementallySortBlocks`.
    fn incrementally_sort_blocks(&mut self) {
        for i in 1..self.blocks.len() {
            if self.blocks[i - 1].suballocator.sum_free_size()
                > self.blocks[i].suballocator.sum_free_size()
            {
                self.blocks.swap(i - 1, i);
                return;
            }
        }
    }

    /// Frees the allocation identified by `alloc`, applying the empty-block hysteresis.
    ///
    /// Port of `VmaBlockVector::Free` / `D3D12MA::BlockVector::Free`. Returns a
    /// [`FreeOutcome`] whose `destroyed_block` names any block the free dropped — the
    /// caller must release its device memory (the pool has forgotten it).
    ///
    /// # Errors
    ///
    /// Returns [`FreeError::InvalidAllocation`], leaving pool state unchanged, if `alloc`
    /// is not a live allocation of *this* pool. Three cases:
    ///
    /// - **Foreign pool (deterministically detected).** `alloc.block_id`'s salt does not
    ///   match this pool's [`PoolConfig::pool_salt`]. When the two pools were given
    ///   distinct salts this is caught deterministically before any lookup (the full 64-bit
    ///   salt is compared, so distinct salts can never collide), so a foreign allocation
    ///   can never disturb this pool's live set (see [`PoolConfig::pool_salt`] for the
    ///   shared-salt caveat).
    /// - **Unknown block.** The salt matches but no current block has that counter
    ///   (the block was already dropped, or the id was fabricated).
    /// - **Stale / double free.** The block exists but the handle is stale, already
    ///   freed, or otherwise not live — the suballocator's generation-tagged handle
    ///   check rejects it without mutating the block.
    ///
    /// # Panics
    ///
    /// Never, for any caller input.
    pub fn free(
        &mut self,
        alloc: Allocation,
        ctx: FreeContext,
    ) -> Result<FreeOutcome<B::Block>, FreeError> {
        // Reject a foreign pool's allocation up front by its full salt, before any lookup:
        // pools with distinct salts never act on each other's ids (distinct salts cannot
        // collide, as the whole 64-bit salt is compared).
        if !self.owns(alloc.block_id) {
            return Err(FreeError::InvalidAllocation);
        }
        // Locate the block by its stable id.
        let block_index = self
            .blocks
            .iter()
            .position(|b| b.id == alloc.block_id)
            .ok_or(FreeError::InvalidAllocation)?;

        // Was there an empty block *before* this free? (VMA 11961 / D3D12MA tracks a
        // bool; we scan, matching VMA `HasEmptyBlock`.)
        let had_empty_block_before = self.blocks.iter().any(|b| b.suballocator.is_empty());

        // Attempt the free on the identified block; a stale/foreign handle is rejected
        // by the suballocator without mutating it.
        match self.blocks[block_index].suballocator.free(alloc.handle) {
            Ok(()) => {}
            Err(HandleError::InvalidHandle) => return Err(FreeError::InvalidAllocation),
        }

        let can_delete_block = self.blocks.len() > self.config.min_block_count;
        let mut destroyed_block: Option<(B::Block, BlockId)> = None;

        if self.blocks[block_index].suballocator.is_empty() {
            // This block became empty. Delete it only if another empty block already
            // existed, or the budget is exceeded — and only above min_block_count.
            // Otherwise retain exactly one empty block (hysteresis). VMA 11970-11979 /
            // D3D12MA 8459-8473.
            if (had_empty_block_before || ctx.budget_exceeded) && can_delete_block {
                destroyed_block = self.remove_block(block_index);
            }
            // else: we now have one empty block — leave it.
        } else if had_empty_block_before && can_delete_block {
            // This block did not become empty, but we already had an empty block: try
            // to reclaim the trailing (last) block if it is empty. VMA 11980-1990 /
            // D3D12MA 8474-8485.
            let last = self.blocks.len() - 1;
            if self.blocks[last].suballocator.is_empty() {
                destroyed_block = self.remove_block(last);
            }
        }

        self.incrementally_sort_blocks();

        // Hand the destroyed block's backend value back to the caller (deferred device
        // memory destruction). We surface it via FreeOutcome instead of destroying it
        // ourselves, because this crate never touches device memory.
        Ok(FreeOutcome { destroyed_block })
    }

    /// Removes the block at internal index `i`, returning its backend value and id.
    fn remove_block(&mut self, i: usize) -> Option<(B::Block, BlockId)> {
        let mut entry = self.blocks.remove(i);
        entry.block.take().map(|block| (block, entry.id))
    }

    /// Frees every allocation and destroys every block, handing each block's backend
    /// value to [`BlockBackend::destroy_block`]. Afterwards the pool has no blocks (not
    /// even `min_block_count`).
    ///
    /// Handles obtained before the call become stale.
    pub fn clear(&mut self, backend: &mut B) {
        self.destroy_all(backend);
    }

    /// Consumes the pool, returning every remaining block's backend value and id so the
    /// caller can release device memory. This is the drop path: the pool never destroys
    /// device memory itself, so the caller *must* drain this iterator (or call
    /// [`clear`](Self::clear) beforehand) to avoid leaking device memory.
    #[must_use = "the returned blocks own device memory that the caller must release"]
    pub fn into_blocks(self) -> impl Iterator<Item = (B::Block, BlockId)> {
        self.blocks.into_iter().filter_map(|mut entry| {
            let id = entry.id;
            entry.block.take().map(|block| (block, id))
        })
    }

    /// Accumulates basic statistics across every block (VMA / D3D12MA `AddStatistics`).
    pub fn statistics(&self) -> Statistics {
        let mut stats = Statistics::default();
        for entry in &self.blocks {
            entry.suballocator.add_statistics(&mut stats);
        }
        stats
    }

    /// Accumulates detailed statistics across every block (`AddDetailedStatistics`).
    pub fn detailed_statistics(&self) -> DetailedStatistics {
        let mut stats = DetailedStatistics::default();
        for entry in &self.blocks {
            entry.suballocator.add_detailed_statistics(&mut stats);
        }
        stats
    }

    /// Builds a [`PoolReport`]: one entry per block plus aggregate statistics. Enough
    /// to build a higher-level report such as wgpu's `wgt::AllocatorReport`.
    pub fn report(&self) -> PoolReport {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        let mut statistics = Statistics::default();
        for entry in &self.blocks {
            entry.suballocator.add_statistics(&mut statistics);
            blocks.push(BlockReport {
                block_id: entry.id,
                size: entry.suballocator.size(),
                allocation_count: entry.suballocator.allocation_count(),
                free_bytes: entry.suballocator.sum_free_size(),
                affinity: entry.affinity,
            });
        }
        PoolReport { blocks, statistics }
    }

    /// Invokes `f(block_id, offset, size, &user_data)` for every live allocation in
    /// every block, in the pool's current internal block order. Allocation-free beyond
    /// the callback. Used to build a per-allocation report.
    pub fn for_each_allocation<F: FnMut(BlockId, u64, u64, &T)>(&self, mut f: F)
    where
        T: Clone,
    {
        for entry in &self.blocks {
            let id = entry.id;
            entry
                .suballocator
                .for_each_allocation(|offset, size, ud| f(id, offset, size, ud));
        }
    }

    /// Validates every internal invariant. Returns `Ok(())` if consistent, or `Err`
    /// with a short description of the first violation. Used heavily by tests; a
    /// correct program never observes an error.
    ///
    /// Checks:
    /// - every block's suballocator validates;
    /// - the block count is at most `max_block_count` (and, for a pool that has not
    ///   been [cleared](Self::clear) or [consumed](Self::into_blocks), at least
    ///   `min_block_count`);
    /// - at most one empty block exists above `min_block_count` (the hysteresis
    ///   invariant).
    ///
    /// It deliberately does *not* assert a total sort order: the incremental sort
    /// performs only one bubble-sort step per mutation, so the vector is only
    /// approximately sorted by free size at any instant.
    #[doc(hidden)]
    pub fn validate(&self) -> Result<(), &'static str> {
        for entry in &self.blocks {
            entry.suballocator.validate()?;
        }
        if self.blocks.len() > self.config.max_block_count {
            return Err("block count exceeds max_block_count");
        }
        // `min_block_count` is the runtime lower bound: allocate/free never drop below
        // it. (`clear` / `into_blocks` intentionally do, but they leave `self.blocks`
        // empty, so this check still holds: 0 blocks with min > 0 is only reachable via
        // those consuming paths, which do not call `validate` afterwards.)
        if !self.blocks.is_empty() && self.blocks.len() < self.config.min_block_count {
            return Err("block count fell below min_block_count");
        }

        // The hysteresis invariant: at most one empty block survives a free, unless
        // `min_block_count` forces several blocks to sit empty (they cannot be dropped).
        // When count == min_block_count every block may be empty; above that, at most
        // one empty block is permitted.
        let empty = self
            .blocks
            .iter()
            .filter(|b| b.suballocator.is_empty())
            .count();
        if self.blocks.len() > self.config.min_block_count && empty > 1 {
            return Err("more than one empty block above min_block_count");
        }
        Ok(())
    }
}
