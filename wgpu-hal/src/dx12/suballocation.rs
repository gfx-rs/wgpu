//! Direct3D 12 memory suballocation for `wgpu-hal`.
//!
//! # Overview
//!
//! Every buffer, texture and acceleration structure needs backing device
//! memory. Rather than give each resource its own `ID3D12Heap` (a *committed*
//! resource), we suballocate most resources out of a smaller number of larger
//! heaps (*placed* resources). This matches the policy of the
//! [D3D12MemoryAllocator][d3d12ma] (D3D12MA) library, which this module is a
//! deliberate reimplementation of on top of wgpu's allocator crates:
//!
//! - [`wgpu_offset_allocator`] provides the per-block TLSF suballocator.
//! - [`wgpu_block_pool`] provides the block *vector*: it grows and shrinks a set
//!   of heaps, applies the VMA/D3D12MA block-size ramp, best-fit selection,
//!   one-empty-block hysteresis and budget gating. We drive it through the
//!   [`BlockBackend`] trait, which is where the actual `ID3D12Heap` creation
//!   and destruction happens ([`D3d12BlockBackend`]).
//!
//! [d3d12ma]: https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator
//!
//! # Pools
//!
//! We keep one [`Pool`] per "pool key" ([`PoolKey`]). The set of keys depends on
//! the adapter's `ResourceHeapTier`, mirroring D3D12MA's `CalcDefaultPoolIndex`:
//!
//! - **Tier 2** (`heterogeneous_resource_heaps == true`): one pool per heap
//!   *class* ([`HeapClass`], i.e. `Default`/`Upload`/`Readback`). A single heap
//!   can hold buffers and all texture kinds, so no `DENY_*` flags are needed.
//! - **Tier 1**: one pool per heap class *times* resource class
//!   ([`ResourceClass`], i.e. buffers / non-RT-DS textures / RT-DS textures),
//!   each with the matching `D3D12_HEAP_FLAG_DENY_*` combination. On Tier 1 a
//!   heap may only ever hold one resource class.
//!
//! `GPU_UPLOAD` heaps are not yet supported; [`HeapClass`] is left extensible so
//! they can be added without reshaping the keying.
//!
//! # Committed vs placed
//!
//! Not every resource is suballocated. [`placement_decision`] decides, following
//! D3D12MA (see that function for the exact heuristics and citations). The most
//! important cases: adapters where suballocation is disabled (Intel Xe) and MSAA
//! textures always go committed; resources larger than a block, and small
//! buffers, prefer committed. When the pool itself decides it cannot serve a
//! request without exceeding budget it returns
//! [`PoolAllocError::ShouldDedicate`], and we fall back to committed.
//!
//! # Budget
//!
//! [`BudgetState`] tracks per memory-segment-group usage against the adapter's
//! reported budget, refreshed from `IDXGIAdapter3::QueryVideoMemoryInfo` every
//! [`BUDGET_REFRESH_INTERVAL`] operations and estimated in between from our own
//! tracked byte counts. The [`wgt::MemoryBudgetThresholds::for_resource_creation`]
//! threshold gates *new heap creation and committed resources only*; an
//! allocation that fits in an existing heap is always allowed, no matter how
//! close to the budget the device is.
//!
//! # Soundness invariants
//!
//! - No `unwrap`/`expect`/`panic`/`assert` is reachable from API inputs or driver
//!   return values. COM errors are mapped to [`crate::DeviceError`]; the
//!   `u64::MAX` sentinel from `GetResourceAllocationInfo` is handled explicitly.
//! - All size/offset arithmetic in the glue is checked or saturating.
//! - Every [`FreeOutcome`] is routed through [`route_free_outcome`] (and thus
//!   [`BlockBackend::destroy_block`]) on every path, so a destroyed block's
//!   mirror-map entry is always removed and its heap released — no heap leaks.
//! - Freeing an allocation that is not live logs an error and returns rather than
//!   panicking.

use alloc::sync::Arc;

use wgpu_block_pool::{
    Algorithm, Allocation as PoolAllocation, AllocationContext, AllocationDesc,
    AllocationType as PoolAllocationType, BlockBackend, BlockId, FreeContext, FreeOutcome, Pool,
    PoolAllocError, PoolConfig, Strategy,
};
use wgpu_sync::Mutex;
use windows::Win32::Graphics::{Direct3D12, Dxgi};

use crate::{
    auxil::dxgi::{name::ObjectExt as _, result::HResult as _},
    dx12::conv,
};

/// The per-allocation user data stored inside the pool.
///
/// It is [`Clone`]d while the pool probes candidate blocks, so it is kept cheap
/// (a refcounted string, or `None` for unlabeled resources). It carries the
/// resource label so [`Allocator::generate_report`] can reproduce it.
type Label = Option<Arc<str>>;

/// D3D12's default heap/resource placement alignment (64 KiB).
///
/// Buffers and non-MSAA textures are placed with this alignment.
const DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT: u64 =
    Direct3D12::D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;

/// D3D12MA's default preferred block size (64 MiB).
///
/// We merge this with the [`AllocationSizes`] policy derived from the user's
/// [`wgt::MemoryHints`]; see [`Pools::preferred_block_size`].
const D3D12MA_DEFAULT_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

/// Block-size policy used by this backend's suballocator to decide how large each
/// memory block should be, derived from the user's [`wgt::MemoryHints`].
///
/// The vulkan backend replicates the same policy locally in
/// `vulkan::suballocation`.
struct AllocationSizes {
    min_device_memblock_size: u64,
    min_host_memblock_size: u64,
}

impl AllocationSizes {
    fn from_memory_hints(memory_hints: &wgt::MemoryHints) -> Self {
        // TODO: the allocator's configuration should take hardware capability into
        // account.
        const MB: u64 = 1024 * 1024;

        match memory_hints {
            wgt::MemoryHints::Performance => Self {
                min_device_memblock_size: 128 * MB,
                min_host_memblock_size: 64 * MB,
            },
            wgt::MemoryHints::MemoryUsage => Self {
                min_device_memblock_size: 8 * MB,
                min_host_memblock_size: 4 * MB,
            },
            wgt::MemoryHints::Manual {
                suballocated_device_memory_block_size,
            } => {
                // TODO: https://github.com/gfx-rs/wgpu/issues/8625
                // Would it be useful to expose the host size in memory hints
                // instead of always using half of the device size?
                let device_size = suballocated_device_memory_block_size;
                let host_min = device_size.start / 2;

                // Clamp the sizes between 4MiB and 256MiB, since we use the sizes when
                // detecting high memory pressure and want them to stay within a sane range.
                Self {
                    min_device_memblock_size: device_size.start.clamp(4 * MB, 256 * MB),
                    min_host_memblock_size: host_min.clamp(4 * MB, 256 * MB),
                }
            }
        }
    }
}

/// Refresh the DXGI budget query at most once per this many alloc/free
/// operations, matching D3D12MA's `ShouldUpdateBudget` cadence.
const BUDGET_REFRESH_INTERVAL: u32 = 30;

/// The three heap *classes* we suballocate from, keyed by CPU access pattern.
///
/// These correspond to D3D12's standard heap types. We create the underlying
/// heaps as `D3D12_HEAP_TYPE_CUSTOM` and choose the CPU page property / memory
/// pool ourselves (see [`HeapClass::heap_properties`]) so that UMA adapters get
/// every heap in the single `L0` memory pool.
///
/// The enum is deliberately extensible: `GPU_UPLOAD` would be added here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HeapClass {
    /// GPU-only memory (`D3D12_HEAP_TYPE_DEFAULT`).
    Default,
    /// CPU-write, GPU-read memory (`D3D12_HEAP_TYPE_UPLOAD`).
    Upload,
    /// GPU-write, CPU-read memory (`D3D12_HEAP_TYPE_READBACK`).
    Readback,
}

impl HeapClass {
    /// All heap classes. Adding a variant here is all that is needed to key a
    /// new class (e.g. `GPU_UPLOAD`).
    const ALL: [HeapClass; 3] = [HeapClass::Default, HeapClass::Upload, HeapClass::Readback];

    /// Selects the heap class for a buffer given its CPU access flags.
    ///
    /// A buffer that is both mappable-read and mappable-write is treated as an
    /// upload buffer.
    fn for_buffer(is_cpu_read: bool, is_cpu_write: bool) -> HeapClass {
        match (is_cpu_read, is_cpu_write) {
            (true, false) => HeapClass::Readback,
            (false, false) => HeapClass::Default,
            // (true, true) and (false, true): CPU-writable => upload.
            (_, true) => HeapClass::Upload,
        }
    }

    /// The heap properties (a `CUSTOM` heap) for this class, honoring UMA.
    fn heap_properties(self, is_uma: bool) -> Direct3D12::D3D12_HEAP_PROPERTIES {
        let cpu_page_property = match self {
            HeapClass::Default => Direct3D12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            // Upload memory is write-combined; readback is write-back so the CPU
            // can read it back cached.
            HeapClass::Upload => Direct3D12::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
            HeapClass::Readback => Direct3D12::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        };

        let memory_pool = match (is_uma, self) {
            // On dedicated GPUs, only GPU-only memory lives in the L1 (device
            // local) pool. On UMA there is only L0.
            (false, HeapClass::Default) => Direct3D12::D3D12_MEMORY_POOL_L1,
            (_, _) => Direct3D12::D3D12_MEMORY_POOL_L0,
        };

        Direct3D12::D3D12_HEAP_PROPERTIES {
            Type: Direct3D12::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: cpu_page_property,
            MemoryPoolPreference: memory_pool,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        }
    }

    /// Which DXGI memory segment group this class draws from, for budget
    /// accounting. On UMA everything is `LOCAL`; on discrete GPUs only
    /// GPU-local (`Default`) memory is `LOCAL`.
    fn segment_group(self, is_uma: bool) -> SegmentGroup {
        if is_uma {
            return SegmentGroup::Local;
        }
        match self {
            HeapClass::Default => SegmentGroup::Local,
            HeapClass::Upload | HeapClass::Readback => SegmentGroup::NonLocal,
        }
    }
}

/// On Tier 1 adapters, the resource class a heap is dedicated to. On Tier 2 all
/// resource classes share a single heap (represented by [`Self::All`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ResourceClass {
    /// Tier 2: a heap that may hold any buffers and textures.
    All,
    /// A heap that may only hold buffers.
    Buffer,
    /// A heap that may only hold non-render-target / non-depth-stencil textures.
    NonRtDsTexture,
    /// A heap that may only hold render-target / depth-stencil textures.
    RtDsTexture,
}

impl ResourceClass {
    /// The `DENY_*` heap flags for a Tier 1 heap dedicated to this class.
    ///
    /// The combinations match D3D12MA's `CalcDefaultPoolParams`: each class
    /// denies the other two. On Tier 2 (`All`) no flags are set, which is
    /// `D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES`.
    fn deny_flags(self) -> Direct3D12::D3D12_HEAP_FLAGS {
        match self {
            ResourceClass::All => Direct3D12::D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
            ResourceClass::Buffer => {
                Direct3D12::D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES
                    | Direct3D12::D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES
            }
            ResourceClass::NonRtDsTexture => {
                Direct3D12::D3D12_HEAP_FLAG_DENY_BUFFERS
                    | Direct3D12::D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES
            }
            ResourceClass::RtDsTexture => {
                Direct3D12::D3D12_HEAP_FLAG_DENY_BUFFERS
                    | Direct3D12::D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES
            }
        }
    }

    /// Derives the resource class of a request from its resource description.
    fn from_resource_desc(desc: &Direct3D12::D3D12_RESOURCE_DESC) -> ResourceClass {
        if desc.Dimension == Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER {
            ResourceClass::Buffer
        } else if desc
            .Flags
            .contains(Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
            || desc
                .Flags
                .contains(Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        {
            ResourceClass::RtDsTexture
        } else {
            ResourceClass::NonRtDsTexture
        }
    }
}

/// The identity of a pool: which heap class it draws from and which resource
/// class (Tier 1) or `All` (Tier 2) its heaps may hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PoolKey {
    heap_class: HeapClass,
    resource_class: ResourceClass,
}

/// The counters bucket a resource belongs to, so frees decrement the same
/// counter creation incremented.
#[derive(Debug, Clone, Copy)]
pub(crate) enum AllocationType {
    Buffer,
    Texture,
    AccelerationStructure,
}

/// Where a resource's backing memory came from, and the bookkeeping needed to
/// release it.
#[derive(Debug)]
enum AllocationInner {
    /// Suballocated from one of our pools. Carries the pool key so the free can
    /// be routed to the correct pool, and the pool's own allocation record.
    Placed {
        pool_key: PoolKey,
        allocation: PoolAllocation,
    },
    /// A committed resource we created. It owns its own implicit heap, so there
    /// is no pool to free, but we do track its bytes against the budget.
    Committed {
        /// Size attributed to counters (may differ from the footprint).
        size: u64,
        /// Footprint tracked against the budget; released exactly on free.
        footprint: u64,
        /// Segment group the footprint was tracked against.
        segment_group: SegmentGroup,
    },
    /// A resource whose memory we do not own (e.g. a swapchain back buffer).
    /// Freeing it is a no-op beyond counter bookkeeping, and it is not tracked
    /// against the budget.
    External { size: u64 },
}

/// A resource's allocation, stored inside the resource and passed back to
/// [`DeviceAllocationContext::free_resource`] on destruction.
#[derive(Debug)]
pub(crate) struct Allocation {
    inner: AllocationInner,
    ty: AllocationType,
}

impl Allocation {
    /// An allocation for a resource whose memory we do not own (swapchain
    /// images). `size` is only used for counter bookkeeping.
    pub fn none(ty: AllocationType, size: u64) -> Self {
        Self {
            inner: AllocationInner::External { size },
            ty,
        }
    }

    /// The size in bytes to attribute to counters for this allocation.
    pub fn size(&self) -> u64 {
        match self.inner {
            AllocationInner::Placed { ref allocation, .. } => allocation.size(),
            AllocationInner::Committed { size, .. } | AllocationInner::External { size } => size,
        }
    }
}

/// One DXGI memory segment group. Budget is tracked per group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentGroup {
    Local,
    NonLocal,
}

impl SegmentGroup {
    fn as_dxgi(self) -> Dxgi::DXGI_MEMORY_SEGMENT_GROUP {
        match self {
            SegmentGroup::Local => Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
            SegmentGroup::NonLocal => Dxgi::DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
        }
    }
}

/// Tracks memory usage of a single segment group against the adapter budget.
///
/// The adapter's reported usage is refreshed only every
/// [`BUDGET_REFRESH_INTERVAL`] operations. Between refreshes, the effective
/// usage is the last fetched usage plus the change in our own tracked bytes
/// since that fetch (saturating), so that a burst of allocations is reflected
/// immediately without a DXGI query per allocation.
#[derive(Debug, Clone, Copy)]
struct SegmentBudget {
    /// Usage reported by DXGI at the last refresh.
    fetched_usage: u64,
    /// Budget reported by DXGI at the last refresh.
    fetched_budget: u64,
    /// Our own tracked bytes for this segment at the last refresh.
    tracked_at_fetch: u64,
    /// Our current tracked bytes for this segment (placed heaps + committed).
    tracked_now: u64,
    /// Bytes attributed to committed resources in this segment. Placed bytes are
    /// derived from the pools' reserved size; keeping the committed total
    /// separate lets `sync_placed_tracked` recompute `tracked_now` without
    /// conflating the two.
    committed_tracked: u64,
    /// Bytes attributed to external allocations that still belong to the device,
    /// such as query heaps.
    external_tracked: u64,
    /// Operations against *this* segment since its last DXGI refresh. Per-segment
    /// (rather than shared) so that traffic alternating between the two segment
    /// groups cannot keep one group's DXGI snapshot stale indefinitely.
    ops_since_refresh: u32,
    /// Whether we have ever successfully fetched from DXGI.
    initialized: bool,
}

impl SegmentBudget {
    fn new() -> Self {
        Self {
            fetched_usage: 0,
            fetched_budget: 0,
            tracked_at_fetch: 0,
            tracked_now: 0,
            committed_tracked: 0,
            external_tracked: 0,
            // Force a refresh on this segment's first use.
            ops_since_refresh: BUDGET_REFRESH_INTERVAL,
            initialized: false,
        }
    }

    /// The best current estimate of this segment's usage.
    fn effective_usage(&self) -> u64 {
        let delta = self.tracked_now.saturating_sub(self.tracked_at_fetch);
        self.fetched_usage.saturating_add(delta)
    }
}

/// Budget state for both segment groups. The refresh counter lives per-segment
/// inside each [`SegmentBudget`] so the two groups refresh independently.
#[derive(Debug)]
struct BudgetState {
    local: SegmentBudget,
    non_local: SegmentBudget,
    /// The configured resource-creation threshold, as a percent of budget.
    threshold: Option<u8>,
}

impl BudgetState {
    fn new(threshold: Option<u8>) -> Self {
        Self {
            local: SegmentBudget::new(),
            non_local: SegmentBudget::new(),
            threshold,
        }
    }

    fn segment_mut(&mut self, group: SegmentGroup) -> &mut SegmentBudget {
        match group {
            SegmentGroup::Local => &mut self.local,
            SegmentGroup::NonLocal => &mut self.non_local,
        }
    }

    fn segment(&self, group: SegmentGroup) -> &SegmentBudget {
        match group {
            SegmentGroup::Local => &self.local,
            SegmentGroup::NonLocal => &self.non_local,
        }
    }

    /// Refreshes the DXGI usage/budget for `group` if that group's own refresh
    /// interval has elapsed. A failed query is treated as advisory (the previous
    /// values are kept, or left uninitialized), never fatal.
    ///
    /// The counter is per-segment (see [`SegmentBudget::ops_since_refresh`]), so
    /// consulting one group never resets the other's staleness clock.
    fn maybe_refresh(&mut self, group: SegmentGroup, adapter: &super::DxgiAdapter) {
        if self.segment(group).ops_since_refresh < BUDGET_REFRESH_INTERVAL {
            return;
        }
        // Refresh the group we are about to consult. We reset this group's
        // counter regardless of success to avoid hammering a failing adapter.
        let info = adapter.query_video_memory_info(group.as_dxgi());
        let seg = self.segment_mut(group);
        if let Ok(info) = info {
            seg.fetched_usage = info.CurrentUsage;
            seg.fetched_budget = info.Budget;
            seg.tracked_at_fetch = seg.tracked_now;
            seg.initialized = true;
        }
        seg.ops_since_refresh = 0;
    }

    /// Adds `bytes` of committed-resource footprint to the tracked usage of
    /// `group`.
    fn add_committed(&mut self, group: SegmentGroup, bytes: u64) {
        let seg = self.segment_mut(group);
        seg.committed_tracked = seg.committed_tracked.saturating_add(bytes);
        seg.tracked_now = seg.tracked_now.saturating_add(bytes);
    }

    /// Removes `bytes` of committed-resource footprint from the tracked usage of
    /// `group`.
    fn sub_committed(&mut self, group: SegmentGroup, bytes: u64) {
        let seg = self.segment_mut(group);
        seg.committed_tracked = seg.committed_tracked.saturating_sub(bytes);
        seg.tracked_now = seg.tracked_now.saturating_sub(bytes);
    }

    fn add_external(&mut self, group: SegmentGroup, bytes: u64) {
        let seg = self.segment_mut(group);
        seg.external_tracked = seg.external_tracked.saturating_add(bytes);
        seg.tracked_now = seg.tracked_now.saturating_add(bytes);
    }

    fn sub_external(&mut self, group: SegmentGroup, bytes: u64) {
        let seg = self.segment_mut(group);
        seg.external_tracked = seg.external_tracked.saturating_sub(bytes);
        seg.tracked_now = seg.tracked_now.saturating_sub(bytes);
    }

    /// Sets the placed (pool-reserved) tracked bytes for `group`, preserving the
    /// committed and external contributions.
    fn set_placed(&mut self, group: SegmentGroup, placed_bytes: u64) {
        let seg = self.segment_mut(group);
        seg.tracked_now = placed_bytes
            .saturating_add(seg.committed_tracked)
            .saturating_add(seg.external_tracked);
    }

    /// Bumps the operation counter for `group`, used to decide when that group
    /// next refreshes. Per-segment so alternating traffic cannot starve either
    /// group's DXGI snapshot.
    fn note_operation(&mut self, group: SegmentGroup) {
        let seg = self.segment_mut(group);
        seg.ops_since_refresh = seg.ops_since_refresh.saturating_add(1);
    }

    /// The threshold, in bytes, above which new heaps / committed resources are
    /// refused for `group`. `None` if no threshold is configured or the budget
    /// is unknown.
    fn threshold_bytes(&self, group: SegmentGroup) -> Option<u64> {
        let threshold = self.threshold?;
        let seg = self.segment(group);
        if !seg.initialized {
            return None;
        }
        // budget * threshold / 100, computed to avoid overflow and truncation
        // in the same order the previous code used (budget / 100 * threshold).
        Some(seg.fetched_budget / 100 * threshold as u64)
    }

    /// Returns whether allocating `size` new bytes in `group` would exceed the
    /// configured threshold.
    fn would_exceed(&self, group: SegmentGroup, size: u64) -> bool {
        let Some(threshold_bytes) = self.threshold_bytes(group) else {
            return false;
        };
        let projected = self.segment(group).effective_usage().saturating_add(size);
        projected >= threshold_bytes
    }

    /// Whether `group` is currently over the threshold (used to drive eager
    /// empty-heap destruction on free).
    fn is_over_threshold(&self, group: SegmentGroup) -> bool {
        let Some(threshold_bytes) = self.threshold_bytes(group) else {
            return false;
        };
        self.segment(group).effective_usage() >= threshold_bytes
    }

    /// The remaining headroom, in bytes, before hitting the threshold in
    /// `group`, for feeding [`AllocationContext::budget_free_bytes`]. `None`
    /// when no threshold applies (treated as unlimited by the pool).
    fn free_bytes(&self, group: SegmentGroup) -> Option<u64> {
        let threshold_bytes = self.threshold_bytes(group)?;
        Some(threshold_bytes.saturating_sub(self.segment(group).effective_usage()))
    }
}

/// The backend that turns block-pool block requests into `ID3D12Heap`s.
///
/// One backend exists per pool; it carries that pool's heap properties, heap
/// flags and alignment so [`create_block`](BlockBackend::create_block) can build
/// a correct `D3D12_HEAP_DESC` without consulting the pool key again.
///
/// The pool stores the `ID3D12Heap` it is handed and returns it on
/// destruction, but the pool never exposes it back to us at allocation time.
/// So we keep a parallel `block_id -> heap` map (a cheap COM ref-count clone
/// per block) so [`DeviceAllocationContext::place_resource`] can find the heap
/// to pass to `CreatePlacedResource`.
struct D3d12BlockBackend {
    device: Direct3D12::ID3D12Device,
    heap_properties: Direct3D12::D3D12_HEAP_PROPERTIES,
    heap_flags: Direct3D12::D3D12_HEAP_FLAGS,
    alignment: u64,
    /// Live heaps by block id, mirroring the pool's block vector.
    heaps: hashbrown::HashMap<BlockId, Direct3D12::ID3D12Heap>,
}

impl D3d12BlockBackend {
    /// Looks up the live heap for a block id (a cheap COM ref-count clone).
    fn heap(&self, block_id: BlockId) -> Option<Direct3D12::ID3D12Heap> {
        self.heaps.get(&block_id).cloned()
    }
}

impl BlockBackend for D3d12BlockBackend {
    type Block = Direct3D12::ID3D12Heap;
    type Error = crate::DeviceError;

    fn create_block(&mut self, size: u64, block_id: BlockId) -> Result<Self::Block, Self::Error> {
        let desc = Direct3D12::D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: self.heap_properties,
            Alignment: self.alignment,
            Flags: self.heap_flags,
        };

        let mut heap: Option<Direct3D12::ID3D12Heap> = None;
        // SAFETY: `desc` is a fully-initialized, locally-owned descriptor and
        // `heap` is a valid out-pointer. `CreateHeap` writes the resulting
        // interface into `heap` on success and leaves it `None` on failure.
        unsafe { self.device.CreateHeap(&desc, &mut heap) }.into_device_result("CreateHeap")?;

        let heap = heap.ok_or(crate::DeviceError::Unexpected)?;
        // Mirror the heap so we can look it up by block id later. The clone only
        // bumps the COM ref-count; both references keep the same heap alive.
        self.heaps.insert(block_id, heap.clone());
        Ok(heap)
    }

    fn destroy_block(&mut self, block: Self::Block, block_id: BlockId) {
        // Drop our mirror reference as well as the one the pool handed back;
        // releasing the last COM reference frees the heap's device memory.
        self.heaps.remove(&block_id);
        drop(block);
    }
}

/// Routes a [`FreeOutcome`] back through [`BlockBackend::destroy_block`].
///
/// [`Pool::free`] hands back the destroyed block via [`FreeOutcome`] but has
/// *already removed it from its own storage*; the backend still holds its mirror
/// entry (for `D3d12BlockBackend`, the `block_id -> ID3D12Heap` clone kept for
/// `CreatePlacedResource`). If we merely dropped the returned block, the mirror
/// entry would keep the heap's COM ref-count above zero and leak the device
/// memory until the whole map is torn down. Every consumer of `destroyed_block`
/// must go through this single chokepoint so the invariant "the mirror map holds
/// exactly the pool's live blocks" holds on every path (success, rollback and
/// error).
///
/// This deliberately releases the block under whatever lock the caller holds.
/// For `ID3D12Heap` the release is a single COM `Release`, so holding the
/// allocator mutex across it is acceptable,
/// and doing it here keeps the map-entry removal and the release atomic and
/// impossible to skip.
///
/// Kept generic over [`BlockBackend`] so it can be unit-tested with a mock
/// backend that mirrors blocks the same way `D3d12BlockBackend` does.
fn route_free_outcome<B: BlockBackend>(backend: &mut B, outcome: FreeOutcome<B::Block>) {
    if let Some((block, block_id)) = outcome.destroyed_block {
        backend.destroy_block(block, block_id);
    }
}

/// One pool together with its backend and budget segment group.
struct PoolEntry {
    key: PoolKey,
    pool: Pool<D3d12BlockBackend, Label>,
    backend: D3d12BlockBackend,
    /// The segment group this pool's heaps count against.
    segment_group: SegmentGroup,
    /// The preferred block size configured for this pool; used to bound the
    /// committed-vs-placed decision.
    preferred_block_size: u64,
}

/// All pools plus the machinery to key requests into them.
struct Pools {
    entries: alloc::vec::Vec<PoolEntry>,
    /// Whether the adapter supports Tier 2 heaps (heterogeneous resource heaps).
    heterogeneous: bool,
    is_uma: bool,
}

impl Pools {
    /// Computes the preferred block size for a heap class, merging D3D12MA's
    /// 64 MiB default with the [`AllocationSizes`] policy derived from
    /// [`wgt::MemoryHints`].
    ///
    /// Merge rule: we take the *larger* of D3D12MA's default and wgpu's
    /// configured min block size for the segment (device vs host), so that:
    /// - the `MemoryHints::Performance` profile (128 MiB device / 64 MiB host)
    ///   keeps its larger, throughput-oriented blocks, while
    /// - the `MemoryHints::MemoryUsage` profile (8 MiB device / 4 MiB host) is
    ///   floored at D3D12MA's 64 MiB rather than shrinking below it, matching
    ///   D3D12MA's out-of-the-box behavior.
    ///
    /// This keeps the *observable* sizing at least as large as before while
    /// never going below D3D12MA's tuned default.
    fn preferred_block_size(heap_class: HeapClass, sizes: &AllocationSizes) -> u64 {
        let hint = match heap_class {
            HeapClass::Default => sizes.min_device_memblock_size,
            HeapClass::Upload | HeapClass::Readback => sizes.min_host_memblock_size,
        };
        hint.max(D3D12MA_DEFAULT_BLOCK_SIZE)
    }

    /// Builds the full set of pools for the adapter.
    fn new(
        device: &Direct3D12::ID3D12Device,
        sizes: &AllocationSizes,
        heterogeneous: bool,
        is_uma: bool,
        pool_salt: &core::sync::atomic::AtomicU64,
    ) -> Result<Self, crate::DeviceError> {
        let mut entries = alloc::vec::Vec::new();

        // The resource classes each heap class is split into.
        let resource_classes: &[ResourceClass] = if heterogeneous {
            &[ResourceClass::All]
        } else {
            &[
                ResourceClass::Buffer,
                ResourceClass::NonRtDsTexture,
                ResourceClass::RtDsTexture,
            ]
        };

        for &heap_class in &HeapClass::ALL {
            let preferred_block_size = Self::preferred_block_size(heap_class, sizes);
            let heap_properties = heap_class.heap_properties(is_uma);
            let segment_group = heap_class.segment_group(is_uma);

            for &resource_class in resource_classes {
                let key = PoolKey {
                    heap_class,
                    resource_class,
                };

                // `Pool::new` takes `&mut backend`, but with `min_block_count ==
                // 0` it creates no blocks, so the backend is not actually used.
                let mut backend = D3d12BlockBackend {
                    device: device.clone(),
                    heap_properties,
                    heap_flags: resource_class.deny_flags(),
                    alignment: DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                    heaps: hashbrown::HashMap::new(),
                };

                // Distinct salt per pool is required so that an allocation from
                // one pool is deterministically rejected by another.
                let salt = pool_salt.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

                let config = PoolConfig {
                    algorithm: Algorithm::Tlsf,
                    preferred_block_size,
                    min_block_count: 0,
                    max_block_count: usize::MAX,
                    explicit_block_size: false,
                    // D3D12 placed resources must be 64 KiB aligned; the pool
                    // applies this as a floor to every allocation's alignment.
                    min_allocation_alignment: DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                    // D3D12 has no buffer-image granularity concept.
                    granularity: 1,
                    debug_margin: 0,
                    // Affinity clustering targets a niche (linear vs optimal
                    // image packing on Vulkan); D3D12 heaps are already split by
                    // resource class on Tier 1, so it has no use here.
                    affinity_clustering: false,
                    pool_salt: salt,
                };

                let pool = Pool::new(config, &mut backend).map_err(map_pool_new_error)?;

                entries.push(PoolEntry {
                    key,
                    pool,
                    backend,
                    segment_group,
                    preferred_block_size,
                });
            }
        }

        Ok(Self {
            entries,
            heterogeneous,
            is_uma,
        })
    }

    /// Returns the [`PoolKey`] for a request.
    fn key_for(&self, heap_class: HeapClass, resource_class: ResourceClass) -> PoolKey {
        let resource_class = if self.heterogeneous {
            ResourceClass::All
        } else {
            resource_class
        };
        PoolKey {
            heap_class,
            resource_class,
        }
    }

    /// Looks up the pool entry for a key. Returns `None` only if the keying
    /// logic is internally inconsistent (which would be a bug).
    fn entry_mut(&mut self, key: PoolKey) -> Option<&mut PoolEntry> {
        self.entries.iter_mut().find(|e| e.key == key)
    }
}

/// Maps a [`Pool::new`] error to a [`crate::DeviceError`]. Only `Backend` and
/// `InvalidRequest` are reachable here (no allocation is performed).
fn map_pool_new_error(err: PoolAllocError<crate::DeviceError>) -> crate::DeviceError {
    match err {
        PoolAllocError::Backend(e) => e,
        // A malformed config is a programming error on our part, not something
        // the caller can act on; surface it as an internal error.
        _ => crate::DeviceError::Unexpected,
    }
}

/// The mutable, lock-protected core of the allocator.
struct Inner {
    pools: Pools,
    budget: BudgetState,
    heap_create_not_zeroed: bool,
}

/// A cheap-to-clone handle to the D3D12 memory allocator.
///
/// This is cloned onto every [`super::CommandEncoder`], so cloning must be
/// cheap: it is an [`Arc`] plus a `Copy` thresholds struct.
#[derive(Clone)]
pub(crate) struct Allocator {
    inner: Arc<Mutex<Inner>>,
    pub memory_budget_thresholds: wgt::MemoryBudgetThresholds,
}

/// Process-wide salt source. Each pool takes a distinct salt so that block ids
/// from different pools never collide, letting the pool crate reject
/// cross-pool frees deterministically.
static POOL_SALT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

impl Allocator {
    pub(crate) fn new(
        raw: &Direct3D12::ID3D12Device,
        memory_hints: &wgt::MemoryHints,
        private_caps: &super::PrivateCapabilities,
        memory_budget_thresholds: wgt::MemoryBudgetThresholds,
    ) -> Result<Self, crate::DeviceError> {
        let sizes = AllocationSizes::from_memory_hints(memory_hints);

        let is_uma = matches!(
            private_caps.memory_architecture,
            super::MemoryArchitecture::Unified { .. }
        );

        let pools = Pools::new(
            raw,
            &sizes,
            private_caps.heterogeneous_resource_heaps,
            is_uma,
            &POOL_SALT,
        )?;

        let inner = Inner {
            pools,
            budget: BudgetState::new(memory_budget_thresholds.for_resource_creation),
            heap_create_not_zeroed: private_caps.heap_create_not_zeroed,
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            memory_budget_thresholds,
        })
    }

    /// Builds a [`wgt::AllocatorReport`] snapshot of the live pools: one entry per
    /// heap, with the label and placement of every live allocation in it.
    pub(crate) fn generate_report(&self) -> wgt::AllocatorReport {
        let inner = self.inner.lock();

        let mut allocations = alloc::vec::Vec::new();
        let mut blocks = alloc::vec::Vec::new();
        let mut total_allocated_bytes = 0u64;
        let mut total_reserved_bytes = 0u64;

        for entry in &inner.pools.entries {
            let report = entry.pool.report();

            // Group this pool's live allocations by block id in a single pass,
            // so assembling per-block ranges below is linear rather than
            // scanning every allocation once per block.
            let mut by_block: hashbrown::HashMap<BlockId, alloc::vec::Vec<wgt::AllocationReport>> =
                hashbrown::HashMap::new();
            entry
                .pool
                .for_each_allocation(|block_id, offset, size, label| {
                    let name = label
                        .as_ref()
                        .map(|s| alloc::string::String::from(s.as_ref()))
                        .unwrap_or_default();
                    by_block
                        .entry(block_id)
                        .or_default()
                        .push(wgt::AllocationReport { name, offset, size });
                });

            for block in &report.blocks {
                let start = allocations.len();
                if let Some(block_allocs) = by_block.remove(&block.block_id) {
                    allocations.extend(block_allocs);
                }
                let end = allocations.len();

                total_reserved_bytes = total_reserved_bytes.saturating_add(block.size);
                total_allocated_bytes = total_allocated_bytes
                    .saturating_add(block.size.saturating_sub(block.free_bytes));

                blocks.push(wgt::MemoryBlockReport {
                    size: block.size,
                    allocations: start..end,
                });
            }
        }

        wgt::AllocatorReport {
            allocations,
            blocks,
            total_allocated_bytes,
            total_reserved_bytes,
        }
    }

    /// Budget probe for allocations we make *outside* the pool machinery (e.g.
    /// query heaps). Returns [`crate::DeviceError::OutOfMemory`] if creating
    /// `size` bytes in the local segment would exceed the resource-creation
    /// threshold.
    ///
    /// Query heaps always live in the `LOCAL` segment group.
    pub(crate) fn add_external_allocation(
        &self,
        adapter: &super::DxgiAdapter,
        size: u64,
    ) -> Result<u64, crate::DeviceError> {
        if size == 0 {
            return Ok(0);
        }
        let mut inner = self.inner.lock();
        if inner.budget.threshold.is_none() {
            return Ok(0);
        }
        inner.budget.maybe_refresh(SegmentGroup::Local, adapter);
        inner.budget.note_operation(SegmentGroup::Local);
        if inner.budget.would_exceed(SegmentGroup::Local, size) {
            return Err(crate::DeviceError::OutOfMemory);
        }
        inner.budget.add_external(SegmentGroup::Local, size);
        Ok(size)
    }

    pub(crate) fn remove_external_allocation(&self, size: u64) {
        if size == 0 {
            return;
        }
        let mut inner = self.inner.lock();
        inner.budget.sub_external(SegmentGroup::Local, size);
        inner.budget.note_operation(SegmentGroup::Local);
    }
}

/// The result of [`placement_decision`]: whether to place or commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// Suballocate from a pool. `prefer_committed` marks requests that D3D12MA
    /// would rather commit but that we still route through the pool with a
    /// dedicated fallback allowed, so a full pool commits rather than fails.
    Placed { prefer_committed: bool },
    /// Create a committed resource directly.
    Committed,
}

/// Decides committed vs placed for a request, following D3D12MA.
///
/// `suballocation_supported` is `false` on adapters with a suballocation bug
/// (Intel Xe), where everything must be committed.
///
/// The heuristics, in order, mirror D3D12MA (`D3D12MemAlloc.cpp`):
/// 1. Suballocation disabled -> committed (preserves the Intel Xe path).
/// 2. MSAA (`sample_count > 1`) -> committed, matching D3D12MA's recommended
///    `MSAA_TEXTURES_ALWAYS_COMMITTED` flag.
/// 3. `size > preferred_block_size` -> committed (it can never fit a block).
/// 4. Small buffers (raw `Width <= 32 KiB`, i.e.
///    `D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT / 2`) -> committed. D3D12MA
///    (`PrefersCommittedAllocation`) prefers committed here because drivers pack
///    small buffers better and placed buffers waste up to 64 KiB of alignment.
///    This check uses `raw_size`, the resource's raw `Width`, **not** the
///    64 KiB-rounded `footprint`: D3D12MA compares `resourceDesc.Width`
///    (`D3D12MemAlloc.cpp`), so testing the footprint here would only ever match
///    a zero-size buffer (any non-zero buffer rounds up to at least 64 KiB).
/// 5. `footprint > preferred_block_size / 2` -> prefer committed. D3D12MA sets
///    `outPreferCommitted = true`; we route through the pool with a dedicated
///    fallback so we still reuse existing space when available.
/// 6. Otherwise -> placed.
///
/// `footprint` is the allocation footprint (buffers: size rounded up to 64 KiB;
/// textures: `GetResourceAllocationInfo`), used for every size-vs-block check.
/// `raw_size` is the resource's raw `Width`, used *only* by the small-buffer
/// rule so it is not defeated by the 64 KiB rounding.
fn placement_decision(
    suballocation_supported: bool,
    is_buffer: bool,
    sample_count: u32,
    raw_size: u64,
    footprint: u64,
    preferred_block_size: u64,
) -> Placement {
    if !suballocation_supported {
        return Placement::Committed;
    }

    if sample_count > 1 {
        return Placement::Committed;
    }

    if footprint > preferred_block_size {
        return Placement::Committed;
    }

    if is_buffer && raw_size <= DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT / 2 {
        return Placement::Committed;
    }

    if footprint > preferred_block_size / 2 {
        return Placement::Placed {
            prefer_committed: true,
        };
    }

    Placement::Placed {
        prefer_committed: false,
    }
}

/// Rounds a buffer size up to the D3D12 64 KiB placement alignment, saturating.
///
/// Buffers always have 64 KiB alignment and a footprint of the size rounded up
/// to 64 KiB, so this is computable without `GetResourceAllocationInfo`.
fn buffer_footprint(size: u64) -> u64 {
    // `next_multiple_of` would panic on overflow; do it saturating instead.
    let align = DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    size.checked_next_multiple_of(align).unwrap_or(u64::MAX)
}

/// To allow us to construct resources from both a `Device` and `CommandEncoder`
/// without needing each function to take a million arguments, we create a
/// borrowed context struct that contains the relevant members.
pub(crate) struct DeviceAllocationContext<'a> {
    pub(crate) raw: &'a Direct3D12::ID3D12Device,
    pub(crate) shared: &'a super::DeviceShared,
    pub(crate) mem_allocator: &'a Allocator,
    pub(crate) counters: &'a wgt::HalCounters,
}

impl<'a> From<&'a super::Device> for DeviceAllocationContext<'a> {
    fn from(device: &'a super::Device) -> Self {
        Self {
            raw: &device.raw,
            shared: &device.shared,
            mem_allocator: &device.mem_allocator,
            counters: &device.counters,
        }
    }
}

impl<'a> From<&'a super::CommandEncoder> for DeviceAllocationContext<'a> {
    fn from(encoder: &'a super::CommandEncoder) -> Self {
        Self {
            raw: &encoder.device,
            shared: &encoder.shared,
            mem_allocator: &encoder.mem_allocator,
            counters: &encoder.counters,
        }
    }
}

impl DeviceAllocationContext<'_> {
    ///////////////////////
    // Resource Creation //
    ///////////////////////

    pub(crate) fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), crate::DeviceError> {
        let is_cpu_read = desc.usage.contains(wgt::BufferUses::MAP_READ);
        let is_cpu_write = desc.usage.contains(wgt::BufferUses::MAP_WRITE);
        let heap_class = HeapClass::for_buffer(is_cpu_read, is_cpu_write);

        let raw_desc = conv::map_buffer_descriptor(desc);

        // Buffers always have 64 KiB alignment and a footprint of the size
        // rounded up to 64 KiB; skip the GetResourceAllocationInfo call.
        let footprint = buffer_footprint(desc.size);

        let name = desc.label.map(Arc::from);

        let (resource, allocation) = self.allocate_resource(
            raw_desc,
            heap_class,
            footprint,
            DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
            true,
            desc.size,
            Direct3D12::D3D12_RESOURCE_STATE_COMMON,
            AllocationType::Buffer,
            name,
        )?;

        self.counters.buffer_memory.add(allocation.size() as isize);

        if let Some(label) = desc.label {
            // If naming fails, free the resource (which also subtracts the counter
            // added above) so the allocation is not leaked.
            if let Err(e) = resource.set_name(label) {
                self.free_resource(resource, allocation);
                return Err(e);
            }
        }

        Ok((resource, allocation))
    }

    pub(crate) fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), crate::DeviceError> {
        // Textures always come from the GPU-local (default) heap class.
        let heap_class = HeapClass::Default;

        let (footprint, alignment) = self.texture_allocation_info(&raw_desc)?;

        let name = desc.label.map(Arc::from);
        let approx_size = desc.format.theoretical_memory_footprint(desc.size);

        let (resource, allocation) = self.allocate_resource(
            raw_desc,
            heap_class,
            footprint,
            alignment,
            false,
            approx_size,
            Direct3D12::D3D12_RESOURCE_STATE_COMMON,
            AllocationType::Texture,
            name,
        )?;

        self.counters.texture_memory.add(allocation.size() as isize);

        if let Some(label) = desc.label {
            // If naming fails, free the resource (which also subtracts the counter
            // added above) so the allocation is not leaked.
            if let Err(e) = resource.set_name(label) {
                self.free_resource(resource, allocation);
                return Err(e);
            }
        }

        Ok((resource, allocation))
    }

    pub(crate) fn create_acceleration_structure(
        &self,
        desc: &crate::AccelerationStructureDescriptor,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), crate::DeviceError> {
        // Acceleration structures are buffers on the GPU-local heap class.
        let heap_class = HeapClass::Default;

        // The descriptor is a buffer, so use the fast-path footprint.
        let footprint = buffer_footprint(desc.size);

        let name = desc.label.map(Arc::from);

        let (resource, allocation) = self.allocate_resource(
            raw_desc,
            heap_class,
            footprint,
            DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
            true,
            desc.size,
            Direct3D12::D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            AllocationType::AccelerationStructure,
            name,
        )?;

        self.counters
            .acceleration_structure_memory
            .add(allocation.size() as isize);

        if let Some(label) = desc.label {
            // If naming fails, free the resource (which also subtracts the counter
            // added above) so the allocation is not leaked.
            if let Err(e) = resource.set_name(label) {
                self.free_resource(resource, allocation);
                return Err(e);
            }
        }

        Ok((resource, allocation))
    }

    /// Queries `GetResourceAllocationInfo` for a texture, returning the
    /// `(footprint, alignment)`.
    ///
    /// A `SizeInBytes` of `u64::MAX` (or `0`) is the driver's invalid-descriptor
    /// sentinel; we return an error rather than attempting to allocate it.
    fn texture_allocation_info(
        &self,
        raw_desc: &Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<(u64, u64), crate::DeviceError> {
        // SAFETY: `raw_desc` is a valid, fully-initialized descriptor; the slice
        // borrows it for the duration of the call.
        let info = unsafe {
            self.raw
                .GetResourceAllocationInfo(0, core::slice::from_ref(raw_desc))
        };

        // `u64::MAX` is D3D12's documented invalid-descriptor sentinel. Some
        // WARP builds also return `0` for allocations that are too large; both
        // cases would be fatal to proceed with, so reject them as OOM.
        if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 {
            return Err(crate::DeviceError::OutOfMemory);
        }

        Ok((info.SizeInBytes, info.Alignment))
    }

    /// The core allocation routine shared by buffers, textures and acceleration
    /// structures. Decides committed vs placed, drives the pool, and falls back
    /// to committed on `ShouldDedicate`.
    #[allow(clippy::too_many_arguments)]
    fn allocate_resource(
        &self,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
        heap_class: HeapClass,
        footprint: u64,
        alignment: u64,
        is_buffer: bool,
        approx_committed_size: u64,
        initial_state: Direct3D12::D3D12_RESOURCE_STATES,
        ty: AllocationType,
        name: Label,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), crate::DeviceError> {
        let resource_class = ResourceClass::from_resource_desc(&raw_desc);
        let sample_count = raw_desc.SampleDesc.Count;

        let mut inner = self.inner_lock();

        let key = inner.pools.key_for(heap_class, resource_class);
        let segment_group = heap_class.segment_group(inner.pools.is_uma);

        let preferred_block_size = inner
            .pools
            .entry_mut(key)
            .map(|e| e.preferred_block_size)
            .unwrap_or(0);

        let placement = placement_decision(
            self.shared.private_caps.suballocation_supported,
            is_buffer,
            sample_count,
            // The small-buffer rule compares the raw resource Width (D3D12MA),
            // which for buffers is exactly the requested size; the footprint is
            // the 64 KiB-rounded value used for the size-vs-block checks.
            raw_desc.Width,
            footprint,
            preferred_block_size,
        );

        // Refresh budget lazily for the segment we are about to allocate from.
        inner
            .budget
            .maybe_refresh(segment_group, &self.shared.adapter);
        inner.budget.note_operation(segment_group);

        match placement {
            Placement::Committed => self.commit_resource(
                &mut inner,
                raw_desc,
                heap_class,
                segment_group,
                footprint,
                approx_committed_size,
                initial_state,
                ty,
            ),
            Placement::Placed { prefer_committed } => {
                // Try the pool first. On ShouldDedicate / a full pool, fall back
                // to committed.
                match self.place_resource(
                    &mut inner,
                    &raw_desc,
                    key,
                    segment_group,
                    footprint,
                    alignment,
                    is_buffer,
                    initial_state,
                    ty,
                    name,
                ) {
                    Ok(res) => Ok(res),
                    Err(PlaceError::ShouldFallBackToCommitted) => self.commit_resource(
                        &mut inner,
                        raw_desc,
                        heap_class,
                        segment_group,
                        footprint,
                        approx_committed_size,
                        initial_state,
                        ty,
                    ),
                    Err(PlaceError::Device(e)) => {
                        // If we preferred committed anyway, a hard pool error is
                        // still worth retrying as committed before giving up.
                        if prefer_committed {
                            self.commit_resource(
                                &mut inner,
                                raw_desc,
                                heap_class,
                                segment_group,
                                footprint,
                                approx_committed_size,
                                initial_state,
                                ty,
                            )
                        } else {
                            Err(e)
                        }
                    }
                }
            }
        }
    }

    /// Attempts to suballocate `raw_desc` from the pool identified by `key` and
    /// create a placed resource in the resulting heap.
    #[allow(clippy::too_many_arguments)]
    fn place_resource(
        &self,
        inner: &mut Inner,
        raw_desc: &Direct3D12::D3D12_RESOURCE_DESC,
        key: PoolKey,
        segment_group: SegmentGroup,
        footprint: u64,
        alignment: u64,
        is_buffer: bool,
        initial_state: Direct3D12::D3D12_RESOURCE_STATES,
        ty: AllocationType,
        name: Label,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), PlaceError> {
        let budget_free_bytes = inner.budget.free_bytes(segment_group);

        let Some(entry) = inner.pools.entry_mut(key) else {
            // Keying is internally inconsistent; treat as a fallback rather than
            // panicking.
            return Err(PlaceError::ShouldFallBackToCommitted);
        };

        let ctx = AllocationContext {
            budget_free_bytes,
            dedicated_fallback_allowed: true,
            preferred_affinity: None,
        };

        // D3D12 has no buffer-image granularity; `Buffer`/`Unknown` are
        // equivalent with `granularity == 1`. We tag buffers as `Buffer` and
        // everything else as `Unknown` for clarity in reports.
        let alloc_type = if is_buffer {
            PoolAllocationType::Buffer
        } else {
            PoolAllocationType::Unknown
        };

        let allocation = match entry.pool.allocate(
            AllocationDesc {
                size: footprint,
                alignment,
                alloc_type,
                strategy: Strategy::Balanced,
                upper_address: false,
            },
            ctx,
            name,
            &mut entry.backend,
        ) {
            Ok(allocation) => allocation,
            Err(PoolAllocError::ShouldDedicate) | Err(PoolAllocError::OutOfPoolMemory) => {
                return Err(PlaceError::ShouldFallBackToCommitted);
            }
            Err(PoolAllocError::Backend(e)) => return Err(PlaceError::Device(e)),
            Err(PoolAllocError::InvalidRequest) | Err(PoolAllocError::UpperAddressUnsupported) => {
                // These indicate a malformed request from our glue; commit as a
                // safe fallback rather than failing the user's allocation.
                log::error!("dx12 pool rejected a request as invalid; falling back to committed");
                return Err(PlaceError::ShouldFallBackToCommitted);
            }
        };

        // A new heap may have been created; refresh our tracked bytes for the
        // segment. We recompute tracked bytes from the pool's reserved size so
        // hysteresis-driven shrink/grow stays accurate.
        Self::sync_placed_tracked(inner, segment_group);

        // Look the entry back up (the borrow above ended) to read the heap.
        let Some(entry) = inner.pools.entry_mut(key) else {
            return Err(PlaceError::ShouldFallBackToCommitted);
        };

        let Some(heap) = entry.backend.heap(allocation.block_id()) else {
            // The block just allocated must exist; if not, undo and fall back.
            // Route the outcome so any block the free drops has its mirror-map
            // entry removed rather than leaking (see `route_free_outcome`).
            match entry.pool.free(allocation, FreeContext::default()) {
                Ok(outcome) => route_free_outcome(&mut entry.backend, outcome),
                Err(e) => log::error!("dx12: rollback free failed: {e:?}"),
            }
            return Err(PlaceError::ShouldFallBackToCommitted);
        };

        let mut resource = None;
        // SAFETY: `heap` is a live heap owned by the pool, `offset` is within it,
        // and `raw_desc` is a valid descriptor. `resource` is a valid out-ptr.
        let hr = unsafe {
            self.raw.CreatePlacedResource(
                &heap,
                allocation.offset(),
                raw_desc,
                initial_state,
                None,
                &mut resource,
            )
        };

        if let Err(e) = hr.into_device_result("CreatePlacedResource") {
            // Roll back the suballocation so we don't leak pool space, then
            // surface the error (or fall back to committed for the caller).
            self.free_pool_allocation(inner, key, segment_group, allocation);
            return Err(PlaceError::Device(e));
        }

        let Some(resource) = resource else {
            self.free_pool_allocation(inner, key, segment_group, allocation);
            return Err(PlaceError::Device(crate::DeviceError::Unexpected));
        };

        let wrapped = Allocation {
            inner: AllocationInner::Placed {
                pool_key: key,
                allocation,
            },
            ty,
        };

        Ok((resource, wrapped))
    }

    /// Creates a committed resource with an implicit heap of the right heap
    /// class, honoring the budget threshold on the target segment group.
    #[allow(clippy::too_many_arguments)]
    fn commit_resource(
        &self,
        inner: &mut Inner,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
        heap_class: HeapClass,
        segment_group: SegmentGroup,
        footprint: u64,
        approx_committed_size: u64,
        initial_state: Direct3D12::D3D12_RESOURCE_STATES,
        ty: AllocationType,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), crate::DeviceError> {
        // Committed resources allocate fresh device memory, so they are gated by
        // the budget threshold just like new heaps.
        if inner.budget.would_exceed(segment_group, footprint) {
            return Err(crate::DeviceError::OutOfMemory);
        }

        let is_uma = inner.pools.is_uma;
        // Committed resources use implicit heaps, which must not carry the
        // DENY_* flags; only the CREATE_NOT_ZEROED flag is meaningful.
        let heap_flags = if inner.heap_create_not_zeroed {
            Direct3D12::D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
        } else {
            Direct3D12::D3D12_HEAP_FLAG_NONE
        };

        let heap_properties = heap_class.heap_properties(is_uma);

        let mut resource = None;
        // SAFETY: `heap_properties` and `raw_desc` are valid, fully-initialized
        // descriptors; `resource` is a valid out-pointer.
        let hr = unsafe {
            self.raw.CreateCommittedResource(
                &heap_properties,
                heap_flags,
                &raw_desc,
                initial_state,
                None,
                &mut resource,
            )
        };

        hr.into_device_result("CreateCommittedResource")?;

        let resource = resource.ok_or(crate::DeviceError::Unexpected)?;

        // Track the committed bytes against the budget so subsequent estimates
        // stay accurate between refreshes.
        inner.budget.add_committed(segment_group, footprint);

        let wrapped = Allocation {
            inner: AllocationInner::Committed {
                size: approx_committed_size,
                footprint,
                segment_group,
            },
            ty,
        };

        Ok((resource, wrapped))
    }

    //////////////////////////
    // Resource Destruction //
    //////////////////////////

    pub(crate) fn free_resource(
        &self,
        resource: Direct3D12::ID3D12Resource,
        allocation: Allocation,
    ) {
        // Make sure the resource is released before we free the backing memory.
        drop(resource);

        let counter = match allocation.ty {
            AllocationType::Buffer => &self.counters.buffer_memory,
            AllocationType::Texture => &self.counters.texture_memory,
            AllocationType::AccelerationStructure => &self.counters.acceleration_structure_memory,
        };
        counter.sub(allocation.size() as isize);

        match allocation.inner {
            AllocationInner::Placed {
                pool_key,
                allocation,
            } => {
                let mut inner = self.inner_lock();
                let segment_group = self.segment_group_for_key(&inner, pool_key);
                inner
                    .budget
                    .maybe_refresh(segment_group, &self.shared.adapter);
                inner.budget.note_operation(segment_group);
                let budget_exceeded = inner.budget.is_over_threshold(segment_group);

                if let Some(entry) = inner.pools.entry_mut(pool_key) {
                    match entry.pool.free(allocation, FreeContext { budget_exceeded }) {
                        Ok(outcome) => {
                            // A destroyed block must be routed back through the
                            // backend so its mirror-map entry is removed and its
                            // heap released; otherwise the heap leaks until
                            // device teardown (see `route_free_outcome`).
                            route_free_outcome(&mut entry.backend, outcome);
                        }
                        Err(e) => {
                            // Never panic on a bad free (fixes the previous
                            // TODO panic). Log and move on; the resource is
                            // already released.
                            log::error!(
                                "dx12: failed to free {:?} allocation: {e:?}",
                                allocation.block_id()
                            );
                        }
                    }
                } else {
                    log::error!("dx12: free for unknown pool key {pool_key:?}");
                }

                // Recompute tracked bytes for the segment from the pool's
                // reserved size after the free.
                Self::sync_placed_tracked(&mut inner, segment_group);
            }
            AllocationInner::Committed {
                size: _,
                footprint,
                segment_group,
            } => {
                // The committed resource's implicit heap was released with the
                // resource above; release the exact footprint we tracked at
                // creation so the budget estimate stays balanced.
                let mut inner = self.inner_lock();
                inner.budget.sub_committed(segment_group, footprint);
                inner.budget.note_operation(segment_group);
            }
            AllocationInner::External { size: _ } => {
                // We do not own this memory; nothing to release.
            }
        }
    }

    /// Frees a pool allocation and reconciles tracked bytes, releasing any
    /// destroyed heap. Used for rollback paths where placed-resource creation
    /// failed after suballocation succeeded.
    fn free_pool_allocation(
        &self,
        inner: &mut Inner,
        key: PoolKey,
        segment_group: SegmentGroup,
        allocation: PoolAllocation,
    ) {
        if let Some(entry) = inner.pools.entry_mut(key) {
            match entry.pool.free(allocation, FreeContext::default()) {
                Ok(outcome) => {
                    // Route any destroyed block back through the backend so its
                    // mirror-map entry is removed and its heap released, even on
                    // this rollback path (see `route_free_outcome`).
                    route_free_outcome(&mut entry.backend, outcome);
                }
                Err(e) => {
                    log::error!("dx12: rollback free failed: {e:?}");
                }
            }
        }
        Self::sync_placed_tracked(inner, segment_group);
    }

    /// Recomputes the placed (pool-reserved) tracked bytes for `segment_group`
    /// from the reserved size of every pool that draws from it, preserving the
    /// committed contribution. This keeps the budget estimate in sync with the
    /// pool's block-vector growth and hysteresis.
    fn sync_placed_tracked(inner: &mut Inner, segment_group: SegmentGroup) {
        let mut placed_reserved = 0u64;
        for entry in &inner.pools.entries {
            if entry.segment_group == segment_group {
                let stats = entry.pool.statistics();
                placed_reserved = placed_reserved.saturating_add(stats.block_bytes);
            }
        }
        inner.budget.set_placed(segment_group, placed_reserved);
    }

    fn segment_group_for_key(&self, inner: &Inner, key: PoolKey) -> SegmentGroup {
        key.heap_class.segment_group(inner.pools.is_uma)
    }

    fn inner_lock(&self) -> wgpu_sync::MutexGuard<'_, Inner> {
        self.mem_allocator.inner.lock()
    }
}

/// Internal error for the placed-resource path.
enum PlaceError {
    /// The pool declined; the caller should create a committed resource instead.
    ShouldFallBackToCommitted,
    /// A hard device error occurred.
    Device(crate::DeviceError),
}

#[cfg(test)]
#[allow(
    clippy::identity_op,
    reason = "`1 * MB` keeps size literals visually aligned"
)]
mod tests {
    use super::*;

    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;

    fn buffer_desc(width: u64) -> Direct3D12::D3D12_RESOURCE_DESC {
        Direct3D12::D3D12_RESOURCE_DESC {
            Dimension: Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: width,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: Direct3D12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: Direct3D12::D3D12_RESOURCE_FLAG_NONE,
        }
    }

    fn texture_desc(flags: Direct3D12::D3D12_RESOURCE_FLAGS) -> Direct3D12::D3D12_RESOURCE_DESC {
        Direct3D12::D3D12_RESOURCE_DESC {
            Dimension: Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: Direct3D12::D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
        }
    }

    // ----- Buffer fast-path footprint math -----

    #[test]
    fn buffer_footprint_rounds_up_to_64k() {
        assert_eq!(buffer_footprint(0), 0);
        assert_eq!(buffer_footprint(1), 64 * KB);
        assert_eq!(buffer_footprint(64 * KB), 64 * KB);
        assert_eq!(buffer_footprint(64 * KB + 1), 128 * KB);
        assert_eq!(buffer_footprint(100 * KB), 128 * KB);
    }

    #[test]
    fn buffer_footprint_saturates_on_overflow() {
        // A near-u64::MAX size cannot be rounded up without overflow; we
        // saturate rather than panic.
        assert_eq!(buffer_footprint(u64::MAX), u64::MAX);
        assert_eq!(buffer_footprint(u64::MAX - 10), u64::MAX);
    }

    // ----- Heap-class keying -----

    #[test]
    fn buffer_heap_class_mapping() {
        // (is_cpu_read, is_cpu_write)
        assert_eq!(HeapClass::for_buffer(false, false), HeapClass::Default);
        assert_eq!(HeapClass::for_buffer(true, false), HeapClass::Readback);
        assert_eq!(HeapClass::for_buffer(false, true), HeapClass::Upload);
        // Read+write is treated as upload, matching the previous mapping.
        assert_eq!(HeapClass::for_buffer(true, true), HeapClass::Upload);
    }

    // ----- Resource-class derivation -----

    #[test]
    fn resource_class_from_desc() {
        assert_eq!(
            ResourceClass::from_resource_desc(&buffer_desc(1024)),
            ResourceClass::Buffer
        );
        assert_eq!(
            ResourceClass::from_resource_desc(&texture_desc(Direct3D12::D3D12_RESOURCE_FLAG_NONE)),
            ResourceClass::NonRtDsTexture
        );
        assert_eq!(
            ResourceClass::from_resource_desc(&texture_desc(
                Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
            )),
            ResourceClass::RtDsTexture
        );
        assert_eq!(
            ResourceClass::from_resource_desc(&texture_desc(
                Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
            )),
            ResourceClass::RtDsTexture
        );
        // A storage texture that is not a render target stays non-RT/DS.
        assert_eq!(
            ResourceClass::from_resource_desc(&texture_desc(
                Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            )),
            ResourceClass::NonRtDsTexture
        );
    }

    #[test]
    fn tier1_deny_flags_match_d3d12ma() {
        // Buffers deny both texture classes.
        assert_eq!(
            ResourceClass::Buffer.deny_flags(),
            Direct3D12::D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES
                | Direct3D12::D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES
        );
        // Non-RT/DS textures deny buffers and RT/DS textures.
        assert_eq!(
            ResourceClass::NonRtDsTexture.deny_flags(),
            Direct3D12::D3D12_HEAP_FLAG_DENY_BUFFERS
                | Direct3D12::D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES
        );
        // RT/DS textures deny buffers and non-RT/DS textures.
        assert_eq!(
            ResourceClass::RtDsTexture.deny_flags(),
            Direct3D12::D3D12_HEAP_FLAG_DENY_BUFFERS
                | Direct3D12::D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES
        );
        // Tier 2 heaps allow everything (no deny flags).
        assert_eq!(
            ResourceClass::All.deny_flags(),
            Direct3D12::D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES
        );
    }

    // ----- UMA segment mapping -----

    #[test]
    fn segment_group_mapping() {
        // Discrete GPU: only GPU-local (Default) memory is LOCAL.
        assert_eq!(HeapClass::Default.segment_group(false), SegmentGroup::Local);
        assert_eq!(
            HeapClass::Upload.segment_group(false),
            SegmentGroup::NonLocal
        );
        assert_eq!(
            HeapClass::Readback.segment_group(false),
            SegmentGroup::NonLocal
        );

        // UMA: everything is LOCAL.
        assert_eq!(HeapClass::Default.segment_group(true), SegmentGroup::Local);
        assert_eq!(HeapClass::Upload.segment_group(true), SegmentGroup::Local);
        assert_eq!(HeapClass::Readback.segment_group(true), SegmentGroup::Local);
    }

    // ----- Committed vs placed heuristic -----

    /// Decides placement for a *buffer* the way production does: the footprint is
    /// the raw size rounded up to 64 KiB (see [`buffer_footprint`]), and the raw
    /// size drives the small-buffer rule. Using this instead of passing an
    /// arbitrary `size` for both is what makes the small-buffer test exercise the
    /// real call shape rather than masking the footprint-rounding.
    fn place_buffer(supported: bool, raw_size: u64, block: u64) -> Placement {
        placement_decision(
            supported,
            true,
            1,
            raw_size,
            buffer_footprint(raw_size),
            block,
        )
    }

    /// Decides placement for a *texture*: textures have no 64 KiB fast-path, so
    /// the footprint comes from `GetResourceAllocationInfo`. The small-buffer
    /// rule never applies (it is gated on `is_buffer`), so `raw_size` is
    /// irrelevant and we pass the footprint for both.
    fn place_texture(supported: bool, samples: u32, footprint: u64, block: u64) -> Placement {
        placement_decision(supported, false, samples, footprint, footprint, block)
    }

    #[test]
    fn placement_intel_xe_always_committed() {
        // Suballocation unsupported => always committed, regardless of size.
        assert_eq!(place_buffer(false, 1024, 64 * MB), Placement::Committed);
        assert_eq!(place_texture(false, 1, 1024, 64 * MB), Placement::Committed);
    }

    #[test]
    fn placement_msaa_committed() {
        // sample_count > 1 => committed even for a small texture.
        assert_eq!(
            place_texture(true, 4, 1 * MB, 64 * MB),
            Placement::Committed
        );
    }

    #[test]
    fn placement_larger_than_block_committed() {
        // footprint > preferred block => committed.
        assert_eq!(
            place_texture(true, 1, 65 * MB, 64 * MB),
            Placement::Committed
        );
    }

    #[test]
    fn placement_small_buffer_committed() {
        // A 4 KiB buffer (raw Width <= 32 KiB) prefers committed even though its
        // footprint rounds up to 64 KiB. This is the real call shape: passing the
        // footprint would round any non-zero buffer past 32 KiB and never match.
        assert_eq!(place_buffer(true, 4 * KB, 64 * MB), Placement::Committed);
        // Exactly the 32 KiB threshold is committed.
        assert_eq!(place_buffer(true, 32 * KB, 64 * MB), Placement::Committed);
        // A 100 KiB buffer is above the small-buffer threshold => placed (with
        // default 64 MiB blocks, well under half a block).
        assert_eq!(
            place_buffer(true, 100 * KB, 64 * MB),
            Placement::Placed {
                prefer_committed: false
            }
        );
        // Just above 32 KiB is placed (not the small-buffer case).
        assert_eq!(
            place_buffer(true, 32 * KB + 1, 64 * MB),
            Placement::Placed {
                prefer_committed: false
            }
        );
        // A small *texture* whose footprint is 32 KiB is not covered by the
        // small-buffer rule (it is buffer-only) and is placed.
        assert_eq!(
            place_texture(true, 1, 32 * KB, 64 * MB),
            Placement::Placed {
                prefer_committed: false
            }
        );
    }

    #[test]
    fn placement_zero_size_buffer_committed() {
        // A zero-size buffer has raw Width 0 <= 32 KiB, so it is committed; the
        // change to a raw-size check must not alter this pre-existing behavior.
        assert_eq!(place_buffer(true, 0, 64 * MB), Placement::Committed);
    }

    #[test]
    fn placement_half_block_prefers_committed() {
        // footprint > half the preferred block => prefer committed, but still
        // routed through the pool with a dedicated fallback.
        assert_eq!(
            place_texture(true, 1, 33 * MB, 64 * MB),
            Placement::Placed {
                prefer_committed: true
            }
        );
    }

    #[test]
    fn placement_normal_placed() {
        // A mid-sized buffer well within a block is placed without preference.
        assert_eq!(
            place_buffer(true, 1 * MB, 64 * MB),
            Placement::Placed {
                prefer_committed: false
            }
        );
    }

    // ----- Preferred block-size merge rule -----

    #[test]
    fn preferred_block_size_merge() {
        // MemoryUsage hint (8 MiB device / 4 MiB host) is floored at D3D12MA's
        // 64 MiB default.
        let usage = AllocationSizes::from_memory_hints(&wgt::MemoryHints::MemoryUsage);
        assert_eq!(
            Pools::preferred_block_size(HeapClass::Default, &usage),
            D3D12MA_DEFAULT_BLOCK_SIZE
        );
        assert_eq!(
            Pools::preferred_block_size(HeapClass::Upload, &usage),
            D3D12MA_DEFAULT_BLOCK_SIZE
        );

        // Performance hint (128 MiB device / 64 MiB host) keeps the larger
        // device block, and the host block equals the default.
        let perf = AllocationSizes::from_memory_hints(&wgt::MemoryHints::Performance);
        assert_eq!(
            Pools::preferred_block_size(HeapClass::Default, &perf),
            128 * MB
        );
        assert_eq!(
            Pools::preferred_block_size(HeapClass::Readback, &perf),
            64 * MB
        );
    }

    // ----- Budget math -----

    #[test]
    fn segment_budget_effective_usage_estimates_between_refreshes() {
        let mut seg = SegmentBudget::new();
        seg.fetched_usage = 1000;
        seg.tracked_at_fetch = 200;
        seg.tracked_now = 200;
        seg.initialized = true;

        // No change since fetch: effective == fetched.
        assert_eq!(seg.effective_usage(), 1000);

        // Allocated 500 more bytes since fetch: reflected immediately.
        seg.tracked_now = 700;
        assert_eq!(seg.effective_usage(), 1500);

        // Freed below the fetch baseline: saturating, never underflows.
        seg.tracked_now = 100;
        assert_eq!(seg.effective_usage(), 1000);
    }

    #[test]
    fn budget_threshold_gating() {
        let mut budget = BudgetState::new(Some(80));
        {
            let seg = budget.segment_mut(SegmentGroup::Local);
            seg.fetched_budget = 1000;
            seg.fetched_usage = 700;
            seg.tracked_at_fetch = 0;
            seg.tracked_now = 0;
            seg.initialized = true;
        }
        // Threshold is 80% of 1000 = 800.
        assert_eq!(budget.threshold_bytes(SegmentGroup::Local), Some(800));

        // 700 + 50 = 750 < 800 => allowed.
        assert!(!budget.would_exceed(SegmentGroup::Local, 50));
        // 700 + 100 = 800 >= 800 => refused.
        assert!(budget.would_exceed(SegmentGroup::Local, 100));

        // Free headroom is 800 - 700 = 100.
        assert_eq!(budget.free_bytes(SegmentGroup::Local), Some(100));
    }

    #[test]
    fn budget_no_threshold_never_gates() {
        let budget = BudgetState::new(None);
        assert!(!budget.would_exceed(SegmentGroup::Local, u64::MAX));
        assert!(!budget.is_over_threshold(SegmentGroup::Local));
        assert_eq!(budget.free_bytes(SegmentGroup::Local), None);
        assert_eq!(budget.threshold_bytes(SegmentGroup::Local), None);
    }

    #[test]
    fn budget_uninitialized_never_gates() {
        // With a threshold but no successful DXGI fetch yet, nothing is gated
        // (we cannot know the budget), matching the previous behavior of only
        // enforcing once budget info is available.
        let budget = BudgetState::new(Some(50));
        assert_eq!(budget.threshold_bytes(SegmentGroup::Local), None);
        assert!(!budget.would_exceed(SegmentGroup::Local, u64::MAX));
    }

    #[test]
    fn budget_committed_tracking_balances() {
        let mut budget = BudgetState::new(Some(90));
        budget.add_committed(SegmentGroup::NonLocal, 500);
        assert_eq!(
            budget.segment(SegmentGroup::NonLocal).committed_tracked,
            500
        );
        assert_eq!(budget.segment(SegmentGroup::NonLocal).tracked_now, 500);

        // set_placed preserves the committed contribution.
        budget.set_placed(SegmentGroup::NonLocal, 300);
        assert_eq!(budget.segment(SegmentGroup::NonLocal).tracked_now, 800);

        budget.sub_committed(SegmentGroup::NonLocal, 500);
        assert_eq!(budget.segment(SegmentGroup::NonLocal).committed_tracked, 0);
        // tracked_now went 800 - 500 = 300 (the placed contribution remains).
        assert_eq!(budget.segment(SegmentGroup::NonLocal).tracked_now, 300);
    }

    #[test]
    fn budget_external_tracking_balances() {
        let mut budget = BudgetState::new(Some(90));
        budget.add_committed(SegmentGroup::Local, 200);
        budget.add_external(SegmentGroup::Local, 500);
        assert_eq!(budget.segment(SegmentGroup::Local).external_tracked, 500);
        assert_eq!(budget.segment(SegmentGroup::Local).tracked_now, 700);

        budget.set_placed(SegmentGroup::Local, 300);
        assert_eq!(budget.segment(SegmentGroup::Local).tracked_now, 1000);

        budget.sub_external(SegmentGroup::Local, 500);
        assert_eq!(budget.segment(SegmentGroup::Local).external_tracked, 0);
        assert_eq!(budget.segment(SegmentGroup::Local).tracked_now, 500);
    }

    // ----- Per-segment refresh counter (Finding 3) -----

    #[test]
    fn refresh_counter_is_per_segment() {
        // The refresh counter is per segment group, so operations against one
        // group must never advance (or reset) the other group's staleness clock.
        // A single shared counter (the bug) would let alternating traffic keep
        // one group's DXGI snapshot stale forever.
        let mut budget = BudgetState::new(Some(50));

        // Both segments start "forced to refresh" (counter == interval).
        assert_eq!(
            budget.segment(SegmentGroup::Local).ops_since_refresh,
            BUDGET_REFRESH_INTERVAL
        );
        assert_eq!(
            budget.segment(SegmentGroup::NonLocal).ops_since_refresh,
            BUDGET_REFRESH_INTERVAL
        );

        // Simulate a refresh landing on Local only (as `maybe_refresh(Local)`
        // does after a successful query): reset Local's counter.
        budget.segment_mut(SegmentGroup::Local).ops_since_refresh = 0;

        // Now drive a long burst of operations that only ever touches Local.
        for _ in 0..(BUDGET_REFRESH_INTERVAL * 4) {
            budget.note_operation(SegmentGroup::Local);
        }

        // Local has accumulated its own operations...
        assert!(budget.segment(SegmentGroup::Local).ops_since_refresh >= BUDGET_REFRESH_INTERVAL);
        // ...but NonLocal's counter is untouched by Local traffic. With the old
        // shared counter, NonLocal would have been reset to 0 by the Local
        // refresh above and never climb back on its own, starving its refresh.
        assert_eq!(
            budget.segment(SegmentGroup::NonLocal).ops_since_refresh,
            BUDGET_REFRESH_INTERVAL,
            "NonLocal staleness clock must not be disturbed by Local traffic"
        );

        // Symmetric check: NonLocal traffic does not disturb Local.
        budget.segment_mut(SegmentGroup::NonLocal).ops_since_refresh = 0;
        budget.segment_mut(SegmentGroup::Local).ops_since_refresh = 7;
        budget.note_operation(SegmentGroup::NonLocal);
        assert_eq!(budget.segment(SegmentGroup::Local).ops_since_refresh, 7);
        assert_eq!(budget.segment(SegmentGroup::NonLocal).ops_since_refresh, 1);
    }

    // ----- Mirror-map / FreeOutcome routing (Finding 1) -----

    /// A [`BlockBackend`] that mirrors live blocks in a `block_id -> size` map,
    /// exactly as [`D3d12BlockBackend`] mirrors `block_id -> ID3D12Heap`. It lets
    /// us assert the mirror map shrinks in lockstep with the pool's block count
    /// as long as every [`FreeOutcome`] is routed through [`route_free_outcome`]
    /// (which is what removes the mirror entry). Without that routing the map
    /// would retain destroyed blocks — the exact heap leak in Finding 1.
    #[derive(Default)]
    struct MirrorMockBackend {
        /// The mirror map, analogous to `D3d12BlockBackend::heaps`.
        mirror: hashbrown::HashMap<BlockId, u64>,
    }

    /// The opaque per-block value the mock hands the pool: carries its id so
    /// `destroy_block` (and `route_free_outcome`) can key the mirror removal.
    #[derive(Clone, Copy, Debug)]
    struct MockBlock {
        id: BlockId,
    }

    impl BlockBackend for MirrorMockBackend {
        type Block = MockBlock;
        type Error = crate::DeviceError;

        fn create_block(
            &mut self,
            size: u64,
            block_id: BlockId,
        ) -> Result<Self::Block, Self::Error> {
            // Mirror the block, just as the real backend clones the heap into its
            // `heaps` map.
            self.mirror.insert(block_id, size);
            Ok(MockBlock { id: block_id })
        }

        fn destroy_block(&mut self, block: Self::Block, block_id: BlockId) {
            assert_eq!(block.id, block_id, "destroy_block id mismatch");
            // The removal that `route_free_outcome` must trigger for every
            // destroyed block.
            self.mirror.remove(&block_id);
        }
    }

    fn mock_pool_config() -> PoolConfig {
        PoolConfig {
            algorithm: Algorithm::Tlsf,
            preferred_block_size: 4096,
            min_block_count: 0,
            max_block_count: usize::MAX,
            // Fixed block size so each full-block allocation forces a new block,
            // making block growth/shrink deterministic.
            explicit_block_size: true,
            min_allocation_alignment: 1,
            granularity: 1,
            debug_margin: 0,
            affinity_clustering: false,
            pool_salt: 0,
        }
    }

    #[test]
    fn route_free_outcome_keeps_mirror_in_lockstep() {
        let mut backend = MirrorMockBackend::default();
        let mut pool =
            Pool::<MirrorMockBackend, Label>::new(mock_pool_config(), &mut backend).unwrap();

        let alloc = |pool: &mut Pool<MirrorMockBackend, Label>, backend: &mut MirrorMockBackend| {
            pool.allocate(
                AllocationDesc {
                    size: 4096,
                    alignment: 1,
                    alloc_type: PoolAllocationType::Unknown,
                    strategy: Strategy::Balanced,
                    upper_address: false,
                },
                AllocationContext::default(),
                None,
                backend,
            )
            .expect("allocation of a full block should succeed")
        };

        // Grow the pool to two blocks (each 4096-byte allocation fills a block).
        let a = alloc(&mut pool, &mut backend);
        let b = alloc(&mut pool, &mut backend);
        assert_eq!(pool.block_count(), 2);
        assert_eq!(backend.mirror.len(), 2, "mirror grows with the pool");

        // Free a: block becomes empty, no other empty existed -> retained.
        let outcome = pool.free(a, FreeContext::default()).unwrap();
        assert!(outcome.destroyed_block.is_none());
        route_free_outcome(&mut backend, outcome);
        assert_eq!(pool.block_count(), 2);
        assert_eq!(backend.mirror.len(), pool.block_count());

        // Free b: a second block empties, hysteresis drops one block. The
        // FreeOutcome carries the destroyed block; routing it removes the mirror
        // entry so the map shrinks in lockstep. This is the regression: before
        // the fix the outcome was discarded and the mirror stayed at 2.
        let outcome = pool.free(b, FreeContext::default()).unwrap();
        assert!(
            outcome.destroyed_block.is_some(),
            "second empty free must drop a block"
        );
        route_free_outcome(&mut backend, outcome);
        assert_eq!(pool.block_count(), 1);
        assert_eq!(
            backend.mirror.len(),
            pool.block_count(),
            "mirror map must shrink in lockstep with the pool's block count"
        );
    }
}
