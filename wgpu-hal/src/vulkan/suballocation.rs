//! Vulkan device-memory suballocation, VMA-style.
//!
//! This is the glue layer that turns wgpu's GPU-API-agnostic allocator crates
//! ([`wgpu_offset_allocator`] + [`wgpu_block_pool`]) into a working Vulkan allocator
//! by supplying the policy that VMA (AMD's VulkanMemoryAllocator) bakes into its C++:
//! memory-type selection, per-heap block sizing, dedicated-allocation heuristics,
//! budget gating, and persistent mapping of host-visible memory.
//!
//! # Structure
//!
//! - [`Allocator`] is the facade the [`super::Device`] owns (behind a `Mutex`). It owns
//!   one lazily-created [`Pool`] per Vulkan memory type (at most
//!   [`vk::MAX_MEMORY_TYPES`]), a snapshot of the physical device's memory properties,
//!   the block-size policy, and cached budget state.
//! - [`Allocation`] is the per-resource value stored in `Buffer` / `Texture` /
//!   `AccelerationStructure`. It caches everything the call sites read:
//!   [`memory`](Allocation::memory), [`offset`](Allocation::offset),
//!   [`size`](Allocation::size), [`mapped_ptr`](Allocation::mapped_ptr), and
//!   [`memory_properties`](Allocation::memory_properties).
//! - [`VkBlockBackend`] is the [`BlockBackend`] the pool drives: it wraps a borrowed
//!   `&ash::Device` plus the target memory type, and creates/destroys `vk::DeviceMemory`
//!   blocks. Host-visible blocks are *persistently mapped*: mapped once at creation and
//!   unmapped only at destruction, so a suballocation's
//!   [`mapped_ptr`](Allocation::mapped_ptr) is always available without any per-map
//!   driver call.
//!
//! # Design and invariants (for auditors)
//!
//! Soundness and panic-freedom are the top priority; this ships in Firefox's WebGPU.
//!
//! - **No reachable panic.** No `unwrap`/`expect`/`panic`/`assert` is reachable from API
//!   inputs or driver return values. Every `Result` from the pool / suballocator crates
//!   is matched explicitly, and every size/offset computation uses checked or saturating
//!   arithmetic. The pool crate itself is documented panic-free for any input.
//! - **Every `FreeOutcome` is consumed.** [`Pool::free`]
//!   returns a `#[must_use]` [`FreeOutcome`](wgpu_block_pool::FreeOutcome); if it names a
//!   destroyed block, this module unmaps and `vkFreeMemory`s it (after dropping the pool
//!   lock). Failing to do so would leak `VkDeviceMemory`.
//! - **`unsafe` is confined to FFI.** The only `unsafe` is the handful of Vulkan calls
//!   (`vkAllocateMemory` / `vkMapMemory` / `vkUnmapMemory` / `vkFreeMemory`) and the
//!   pointer arithmetic for a mapped allocation, each with a `SAFETY` comment.
//! - **Dedicated allocations bypass the pools** (via `vkAllocateMemory` with a
//!   `VkMemoryDedicatedAllocateInfo`) and are tracked in a side list so
//!   [`generate_report`](Allocator::generate_report) and the memory-object count include
//!   them.
//!
//! # Policy provenance
//!
//! The policy is ported from VMA (`include/vk_mem_alloc.h`); deviations are noted at each
//! site. In summary:
//!
//! 1. **Memory-type selection** (`FindMemoryPreferences` + `vmaFindMemoryTypeIndex`):
//!    candidate mask = `requirements.memory_type_bits & valid_ash_memory_types`; derive
//!    required / preferred / not-preferred property-flag sets from the usage class; the
//!    winning type minimizes `popcount(preferred & !flags) + popcount(flags &
//!    not_preferred)` among types holding all required flags; on allocation failure the
//!    chosen bit is cleared and the next-best type tried, until the mask is empty.
//! 2. **Preferred block size** (`CalcPreferredBlockSize`): `heap <= 1 GiB ? heap / 8 :
//!    256 MiB`, then merged with the [`wgt::MemoryHints`] block-size policy (this
//!    backend's local [`BlockSizePolicy`]).
//! 3. **Dedicated heuristics** (`AllocateMemoryOfType`): required when the driver
//!    requires it or for lazily-allocated (TRANSIENT) images; preferred when the driver
//!    prefers it or `size > preferred_block_size / 2`; a mere preference is suppressed
//!    when the device memory-object count is strictly above `3/4 *
//!    maxMemoryAllocationCount`.
//! 4. **Budget** (`VK_EXT_memory_budget`): the [`wgt::MemoryBudgetThresholds`]
//!    `for_resource_creation` percentage gates *new blocks and dedicated allocations
//!    only* — suballocating within an existing block is always allowed.

use alloc::borrow::ToOwned as _;
use alloc::{sync::Arc, vec::Vec};
use core::ptr::NonNull;

use ash::vk;
use hashbrown::HashMap;
use wgpu_sync::Mutex;
use wgt::MemoryBudgetThresholds;

pub(super) use wgpu_block_pool::AllocationType;
use wgpu_block_pool::{
    Algorithm, AllocationContext, AllocationDesc, BlockBackend, BlockId, FreeContext, Pool,
    PoolAllocError, PoolConfig, Strategy,
};

use crate::DeviceError;

/// One mebibyte, for readable block-size constants.
const MB: u64 = 1024 * 1024;

/// `VMA_SMALL_HEAP_MAX_SIZE`: heaps at or below this size use `heap / 8` as the preferred
/// block size instead of the large-heap default (`CalcPreferredBlockSize`).
const SMALL_HEAP_MAX_SIZE: u64 = 1024 * MB;

/// VMA's default `preferredLargeHeapBlockSize` (256 MiB), used for heaps above
/// [`SMALL_HEAP_MAX_SIZE`].
const LARGE_HEAP_BLOCK_SIZE: u64 = 256 * MB;

/// How many allocate/free operations pass between refreshes of the cached
/// `VK_EXT_memory_budget` figures. Between refreshes, usage is estimated by adding the
/// net change in tracked block bytes to the last fetched usage.
const BUDGET_REFRESH_INTERVAL: u32 = 30;

/// Process-wide source of unique [`PoolConfig::pool_salt`] values.
///
/// Every [`Pool`] created anywhere in the process takes a distinct salt via
/// `fetch_add`, so an [`Allocation`] handed to the wrong pool is rejected
/// deterministically (see [`wgpu_block_pool::PoolConfig::pool_salt`]).
static NEXT_POOL_SALT: wgpu_sync::atomic::AtomicU64 = wgpu_sync::atomic::AtomicU64::new(1);

/// Which of wgpu-hal's three memory-usage classes an allocation belongs to.
///
/// Each maps to a VMA `AUTO`-style required / preferred / not-preferred property-flag
/// set (see [`MemoryUsage::preferences`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MemoryUsage {
    /// GPU-only memory (no host access).
    GpuOnly,
    /// Host-visible memory optimised for CPU writes (upload; a buffer with
    /// `MAP_WRITE`).
    CpuToGpu,
    /// Host-visible memory optimised for CPU reads (readback; a buffer with
    /// `MAP_READ`).
    GpuToCpu,
}

/// The required / preferred / not-preferred memory property flags for a usage class.
#[derive(Clone, Copy, Debug, Default)]
struct MemoryPreferences {
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
    not_preferred: vk::MemoryPropertyFlags,
}

impl MemoryUsage {
    /// Derives the VMA `AUTO`-style property-flag preferences for this usage class.
    ///
    /// This is a faithful reduction of VMA's `FindMemoryPreferences`
    /// (`vk_mem_alloc.h`) for the resource kinds wgpu creates, specialised to the
    /// non-integrated-GPU path (so host-visible usages still prefer `DEVICE_LOCAL`
    /// where available, e.g. a resizable BAR):
    ///
    /// - [`GpuOnly`](Self::GpuOnly): the "no CPU access" branch — prefer `DEVICE_LOCAL`.
    /// - [`CpuToGpu`](Self::CpuToGpu): the `HOST_ACCESS_SEQUENTIAL_WRITE` branch with
    ///   device access — require `HOST_VISIBLE`, prefer `DEVICE_LOCAL`, and mark
    ///   `HOST_CACHED` not-preferred (want uncached / write-combined).
    /// - [`GpuToCpu`](Self::GpuToCpu): the `HOST_ACCESS_RANDOM` branch — require
    ///   `HOST_VISIBLE`, prefer `HOST_CACHED`.
    fn preferences(self) -> MemoryPreferences {
        use vk::MemoryPropertyFlags as F;
        match self {
            MemoryUsage::GpuOnly => MemoryPreferences {
                required: F::empty(),
                preferred: F::DEVICE_LOCAL,
                not_preferred: F::empty(),
            },
            MemoryUsage::CpuToGpu => MemoryPreferences {
                required: F::HOST_VISIBLE,
                preferred: F::DEVICE_LOCAL,
                not_preferred: F::HOST_CACHED,
            },
            MemoryUsage::GpuToCpu => MemoryPreferences {
                required: F::HOST_VISIBLE,
                preferred: F::HOST_CACHED,
                not_preferred: F::empty(),
            },
        }
    }
}

/// The block-size policy derived from [`wgt::MemoryHints`], kept local to the vulkan
/// backend. The dx12 backend expresses the equivalent `MemoryHints`-derived policy with
/// its own local `AllocationSizes` type.
///
/// The values are the maximum device / host memory-block sizes; the minimum block size
/// is not needed here because the pool's own `1/8 -> 1/4 -> 1/2 -> full` ramp handles
/// small first blocks.
#[derive(Clone, Copy, Debug)]
struct BlockSizePolicy {
    max_device_memblock_size: u64,
    max_host_memblock_size: u64,
}

impl BlockSizePolicy {
    /// Derives the block-size policy from the user's [`wgt::MemoryHints`].
    fn from_memory_hints(hints: &wgt::MemoryHints) -> Self {
        match hints {
            wgt::MemoryHints::Performance => Self {
                max_device_memblock_size: 256 * MB,
                max_host_memblock_size: 128 * MB,
            },
            wgt::MemoryHints::MemoryUsage => Self {
                max_device_memblock_size: 64 * MB,
                max_host_memblock_size: 32 * MB,
            },
            wgt::MemoryHints::Manual {
                suballocated_device_memory_block_size,
            } => {
                let device_end = suballocated_device_memory_block_size.end;
                let host_end = device_end / 2;
                // Clamp to gpu-allocator's historical [4 MiB, 256 MiB] range.
                Self {
                    max_device_memblock_size: device_end.clamp(4 * MB, 256 * MB),
                    max_host_memblock_size: host_end.clamp(4 * MB, 256 * MB),
                }
            }
        }
    }
}

/// A snapshot of the physical device's memory properties, taken once at
/// [`Allocator::new`]. Stored so allocation decisions never re-query the driver.
#[derive(Debug)]
struct MemoryProperties {
    /// One entry per memory type (`memory_type_count` long).
    types: Vec<vk::MemoryType>,
    /// One entry per memory heap (`memory_heap_count` long).
    heaps: Vec<vk::MemoryHeap>,
}

/// A block of `vk::DeviceMemory` backing a [`Pool`], with its optional persistent
/// mapping. This is the [`BlockBackend::Block`] the pool stores and hands back on
/// destruction.
#[derive(Debug)]
pub(super) struct VkBlock {
    memory: vk::DeviceMemory,
    /// The base pointer of this block's persistent mapping, if it is host-visible. VMA
    /// maps host-visible blocks once at creation and never unmaps until free. The size
    /// and mapped base are mirrored into the [`BlockRegistry`] for suballocation lookup.
    mapped: Option<NonNull<u8>>,
}

// SAFETY: `VkBlock`'s `mapped` pointer names Vulkan-owned device memory. The pointer is
// only ever offset (never dereferenced within this module) and access is externally
// synchronized through the `Allocator`'s `Mutex`, matching how the previous
// `gpu-allocator` `Allocation` was `Send + Sync`.
unsafe impl Send for VkBlock {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for VkBlock {}

/// The subset of a block's data needed to resolve a suballocation's device memory and
/// mapped pointer. The pool hides its stored [`VkBlock`] values while they are live, so
/// this module keeps a parallel [`BlockId`]-keyed registry (maintained by the backend as
/// it creates and destroys blocks) to look them up.
#[derive(Clone, Copy, Debug)]
struct BlockInfo {
    memory: vk::DeviceMemory,
    mapped: Option<NonNull<u8>>,
    size: u64,
}

// SAFETY: as for `VkBlock`; the mapped pointer is only offset and access is serialized by
// the `Allocator`'s `Mutex`.
unsafe impl Send for BlockInfo {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for BlockInfo {}

/// A [`BlockId`]-keyed registry of live block memory, shared between the [`Allocator`]
/// and the [`VkBlockBackend`] it drives.
type BlockRegistry = Mutex<HashMap<BlockId, BlockInfo>>;

/// The [`BlockBackend`] the [`Pool`] drives: it creates and destroys the
/// `vk::DeviceMemory` blocks for a single memory type, mapping host-visible ones.
///
/// The [`Pool`] stores no backend value — it only names `B`'s associated `Block` and
/// `Error` types — so this backend can be built fresh for each operation that needs it.
/// The `ash::Device` is held as a raw pointer (rather than a reference) purely to keep
/// this type free of a lifetime parameter, so the stored [`Pool`] type is `'static`;
/// see [`VkBlockBackend::new`] for the safety contract that keeps the pointer valid.
pub(super) struct VkBlockBackend {
    /// The device to allocate/free memory on. Dereferenced only while a
    /// [`VkBlockBackend`] built by [`new`](Self::new) is alive, during which the caller
    /// guarantees the device outlives it.
    device: NonNull<ash::Device>,
    memory_type_index: u32,
    host_visible: bool,
    /// Whether the `buffer_device_address` feature is enabled, requiring every block that
    /// might back a `SHADER_DEVICE_ADDRESS` buffer to be allocated with
    /// `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` (matching the old `gpu-allocator`).
    buffer_device_address: bool,
    /// Counter of live `vk::DeviceMemory` objects (blocks + dedicated), shared with the
    /// [`Allocator`] so the dedicated heuristic can honour `maxMemoryAllocationCount`.
    memory_objects: NonNull<Mutex<u64>>,
    /// The shared block registry, updated as blocks are created/destroyed so the
    /// [`Allocator`] can resolve a suballocation's block memory.
    registry: NonNull<BlockRegistry>,
}

impl VkBlockBackend {
    /// Builds a backend bound to `device` for the target memory type.
    ///
    /// # Safety
    ///
    /// `device`, `memory_objects`, and `registry` must outlive the returned
    /// `VkBlockBackend` and all [`BlockBackend`] calls made through it. This holds because
    /// callers construct the backend on the stack, pass `&mut` it to a single pool call,
    /// and drop it before the borrowed device could go away (all three are owned by the
    /// same `Device` that owns the `Allocator`).
    unsafe fn new(
        device: &ash::Device,
        memory_type_index: u32,
        host_visible: bool,
        buffer_device_address: bool,
        memory_objects: &Mutex<u64>,
        registry: &BlockRegistry,
    ) -> Self {
        VkBlockBackend {
            device: NonNull::from(device),
            memory_type_index,
            host_visible,
            buffer_device_address,
            memory_objects: NonNull::from(memory_objects),
            registry: NonNull::from(registry),
        }
    }

    /// Borrows the device. See [`new`](Self::new) for why this is sound.
    fn device(&self) -> &ash::Device {
        // SAFETY: `self.device` was built from a live `&ash::Device` in `new`, whose
        // safety contract requires that device to outlive `self`.
        unsafe { self.device.as_ref() }
    }

    /// Borrows the shared memory-object counter. See [`new`](Self::new).
    fn memory_objects(&self) -> &Mutex<u64> {
        // SAFETY: as for `device`; the counter outlives `self` by `new`'s contract.
        unsafe { self.memory_objects.as_ref() }
    }

    /// Borrows the shared block registry. See [`new`](Self::new).
    fn registry(&self) -> &BlockRegistry {
        // SAFETY: as for `device`; the registry outlives `self` by `new`'s contract.
        unsafe { self.registry.as_ref() }
    }
}

impl BlockBackend for VkBlockBackend {
    type Block = VkBlock;
    type Error = DeviceError;

    fn create_block(&mut self, size: u64, block_id: BlockId) -> Result<VkBlock, DeviceError> {
        let device = self.device();
        let memory = allocate_device_memory(
            device,
            size,
            self.memory_type_index,
            self.buffer_device_address,
            None,
        )?;

        let mapped = if self.host_visible {
            // SAFETY: `memory` was just allocated from a host-visible type; mapping the
            // whole range is valid. On failure we free the memory before returning.
            match unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(ptr) => NonNull::new(ptr.cast::<u8>()),
                Err(e) => {
                    // SAFETY: `memory` is a live allocation owned by this module and has
                    // no bound resources yet.
                    unsafe { device.free_memory(memory, None) };
                    return Err(super::map_host_device_oom_err(e));
                }
            }
        } else {
            None
        };

        increment_memory_objects(self.memory_objects());
        self.registry().lock().insert(
            block_id,
            BlockInfo {
                memory,
                mapped,
                size,
            },
        );

        Ok(VkBlock { memory, mapped })
    }

    fn destroy_block(&mut self, block: VkBlock, block_id: BlockId) {
        self.registry().lock().remove(&block_id);
        destroy_vk_block(self.device(), block);
        decrement_memory_objects(self.memory_objects());
    }
}

/// Allocates a `vk::DeviceMemory` object, adding a `VkMemoryAllocateFlagsInfo` with
/// `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` when `buffer_device_address` is enabled, and an
/// optional `VkMemoryDedicatedAllocateInfo` for a dedicated allocation.
///
/// The old `gpu-allocator` set the device-address flag on *every* allocation whenever the
/// `buffer_device_address` feature was on; this preserves that behaviour so buffers created
/// with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` bind successfully.
fn allocate_device_memory(
    device: &ash::Device,
    size: u64,
    memory_type_index: u32,
    buffer_device_address: bool,
    dedicated: Option<&mut vk::MemoryDedicatedAllocateInfo<'_>>,
) -> Result<vk::DeviceMemory, DeviceError> {
    let mut allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(memory_type_index);

    let mut flags_info =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    if buffer_device_address {
        allocate_info = allocate_info.push_next(&mut flags_info);
    }
    if let Some(dedicated) = dedicated {
        allocate_info = allocate_info.push_next(dedicated);
    }

    // SAFETY: `allocate_info` is fully initialized with a valid, in-range memory type index
    // (chosen from this device's memory properties) and a non-zero size; any chained
    // extension structs (`flags_info`, `dedicated`) outlive the call.
    unsafe { device.allocate_memory(&allocate_info, None) }.map_err(super::map_host_device_oom_err)
}

/// Adds one to the shared live-`vk::DeviceMemory`-object counter (saturating).
///
/// This is the single writer used by every site that creates a device-memory object
/// (pool block creation and dedicated allocation), so the counter can only ever be bumped
/// through one correctly-scoped `lock()`. Taking the lock and releasing it inside this
/// function makes the double-`lock()`-in-one-statement recursive-deadlock pattern
/// impossible to reintroduce at a call site.
fn increment_memory_objects(counter: &Mutex<u64>) {
    let mut count = counter.lock();
    *count = count.saturating_add(1);
}

/// Subtracts one from the shared live-`vk::DeviceMemory`-object counter (saturating).
///
/// The decrement counterpart of [`increment_memory_objects`]; used by every site that
/// destroys a device-memory object (pool block destruction and dedicated free).
fn decrement_memory_objects(counter: &Mutex<u64>) {
    let mut count = counter.lock();
    *count = count.saturating_sub(1);
}

/// Unmaps (if mapped) and frees a block's device memory. Shared by the backend's
/// `destroy_block` and the free path that consumes a `FreeOutcome`.
fn destroy_vk_block(device: &ash::Device, block: VkBlock) {
    if block.mapped.is_some() {
        // SAFETY: the block was mapped over its whole range at creation and has not been
        // unmapped since; unmapping a currently-mapped memory object is valid.
        unsafe { device.unmap_memory(block.memory) };
    }
    // SAFETY: `block.memory` is a live allocation owned by this module. The pool
    // guarantees no live suballocations remain, and the resource bound to it has already
    // been destroyed by the caller before this runs.
    unsafe { device.free_memory(block.memory, None) };
}

/// A live allocation, stored per-resource in `Buffer` / `Texture` /
/// `AccelerationStructure`.
///
/// It caches everything the call sites read so they need no further pool access:
/// the device memory, offset and size, the mapped base pointer (for host-visible
/// memory), and the memory type's property flags.
///
/// This type is `pub` only so it can appear in the (public) [`super::TextureMemory`]
/// enum; its fields and methods remain crate-internal.
#[derive(Debug)]
pub struct Allocation {
    /// The kind of the underlying allocation: pooled or dedicated.
    kind: AllocationKind,
    /// The device memory backing this allocation.
    memory: vk::DeviceMemory,
    /// The offset of this allocation within `memory`. Always 0 for a dedicated
    /// allocation.
    offset: u64,
    /// The usable size of this allocation.
    size: u64,
    /// The mapped pointer for this allocation (block base + offset), if the memory type
    /// is host-visible. Bounds-checked against the block size at construction.
    mapped_ptr: Option<NonNull<u8>>,
    /// The property flags of the memory type this allocation came from.
    memory_properties: vk::MemoryPropertyFlags,
    /// The memory type index this allocation came from.
    memory_type_index: u32,
}

// SAFETY: as for `VkBlock`, the mapped pointer names Vulkan-owned memory and is only
// offset, never dereferenced within this module; the previous `gpu-allocator`
// `Allocation` was likewise `Send + Sync`.
unsafe impl Send for Allocation {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for Allocation {}

/// Whether an [`Allocation`] is a suballocation of a pooled block or its own dedicated
/// device-memory object.
#[derive(Debug)]
enum AllocationKind {
    /// Suballocated from a [`Pool`] block. Carries the pool-layer allocation so it can
    /// be freed back to the pool.
    Pooled(wgpu_block_pool::Allocation),
    /// Its own `vk::DeviceMemory`, allocated with `vkAllocateMemory`. Freed directly.
    Dedicated,
}

impl Allocation {
    /// The device memory backing this allocation.
    pub(super) fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    /// The offset of this allocation within its device memory.
    pub(super) fn offset(&self) -> u64 {
        self.offset
    }

    /// The usable size of this allocation, in bytes.
    pub(super) fn size(&self) -> u64 {
        self.size
    }

    /// The mapped pointer for this allocation, if the memory is host-visible.
    ///
    /// This is `None` only for non-host-visible memory: host-visible blocks are
    /// persistently mapped at creation, so a host-visible allocation always has a
    /// pointer. The pointer already includes this allocation's offset.
    pub(super) fn mapped_ptr(&self) -> Option<NonNull<core::ffi::c_void>> {
        self.mapped_ptr.map(NonNull::cast)
    }

    /// The property flags of the memory type this allocation came from.
    pub(super) fn memory_properties(&self) -> vk::MemoryPropertyFlags {
        self.memory_properties
    }
}

/// The Vulkan device-memory suballocator facade owned by [`super::Device`].
pub(super) struct Allocator {
    /// One [`Pool`] per memory type, created lazily on first use. Indexed by memory type
    /// index; `None` until a type is first allocated from.
    pools: Vec<Option<Pool<VkBlockBackend, Option<Arc<str>>>>>,
    /// Snapshot of the physical device's memory properties.
    mem_props: MemoryProperties,
    /// Mask of memory types the backend is willing to use (`valid_ash_memory_types`).
    valid_memory_types: u32,
    /// The block-size policy (from `MemoryHints`).
    block_size_policy: BlockSizePolicy,
    /// `maxMemoryAllocationCount` from the device limits.
    max_memory_allocation_count: u32,
    /// `bufferImageGranularity` from the device limits.
    buffer_image_granularity: u64,
    /// `nonCoherentAtomSize` from the device limits (min alignment for host-visible
    /// mapped ranges).
    non_coherent_atom_size: u64,
    /// Budget thresholds from the instance descriptor.
    memory_budget_thresholds: MemoryBudgetThresholds,
    /// Whether `VK_EXT_memory_budget` is enabled (budget figures are trustworthy).
    memory_budget_supported: bool,
    /// Whether the `buffer_device_address` feature is enabled, so every device-memory
    /// allocation must carry `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`.
    buffer_device_address: bool,
    /// Cached budget state, refreshed every [`BUDGET_REFRESH_INTERVAL`] operations.
    budget: BudgetState,
    /// Live device-memory-object count (blocks + dedicated). Referenced by the backend
    /// (via a raw pointer that never outlives a call) so block creation/destruction keeps
    /// it in step; also bumped directly for dedicated allocations.
    memory_objects: Mutex<u64>,
    /// [`BlockId`]-keyed registry of live block memory, maintained by the backend and
    /// read when resolving a pooled suballocation's device memory / mapped pointer.
    block_registry: BlockRegistry,
    /// The dedicated allocations, kept for reporting.
    dedicated: Vec<DedicatedRecord>,
}

/// A record of a live dedicated allocation, for reporting.
#[derive(Debug)]
struct DedicatedRecord {
    memory: vk::DeviceMemory,
    size: u64,
    label: Option<Arc<str>>,
}

/// Cached per-heap budget figures plus the bookkeeping needed to estimate usage between
/// refreshes (VMA refreshes budgets periodically rather than on every call).
///
/// All vectors are indexed by heap and are `heap_count` long.
#[derive(Debug, Default)]
struct BudgetState {
    /// Operations since the last refresh; a refresh happens once this reaches
    /// [`BUDGET_REFRESH_INTERVAL`].
    ops_since_refresh: u32,
    /// Per-heap budget in bytes, at the last refresh
    /// (`VkPhysicalDeviceMemoryBudgetPropertiesEXT::heapBudget`, or 80% of the heap size
    /// when `VK_EXT_memory_budget` is unavailable).
    heap_budget: Vec<u64>,
    /// Per-heap usage in bytes reported by the driver at the last refresh (0 without the
    /// extension).
    heap_usage_at_fetch: Vec<u64>,
    /// Per-heap tracked block+dedicated bytes at the last refresh, so the delta since can
    /// be added to `heap_usage_at_fetch`.
    tracked_bytes_at_fetch: Vec<u64>,
    /// Per-heap tracked block+dedicated bytes right now (blocks and dedicated allocations
    /// this allocator has created).
    tracked_bytes: Vec<u64>,
}

/// A pure snapshot of everything the allocation-decision logic needs about one memory
/// type. Extracted so the decision functions can be unit-tested without a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemTypeInfo {
    property_flags: vk::MemoryPropertyFlags,
    heap_index: u32,
}

/// Scores and orders the candidate memory types for a request, best (lowest cost) first.
///
/// Port of VMA's `vmaFindMemoryTypeIndex` scan (`vk_mem_alloc.h`): among the types whose
/// bit is set in `candidate_mask` and that hold every `required` flag, cost is
/// `popcount(preferred & !flags) + popcount(flags & not_preferred)`; the list is returned
/// sorted by ascending cost so the caller can try the best type, then the next best on
/// failure (VMA clears the failed bit and re-scans; sorting once is equivalent and
/// avoids a rescan). Types missing a required flag are excluded entirely.
///
/// Returns memory type indices. Deterministic: ties keep the lower index first (stable
/// order), matching VMA's "first type with this minimal cost wins" bias.
fn rank_memory_types(
    candidate_mask: u32,
    types: &[MemTypeInfo],
    prefs: MemoryPreferences,
) -> Vec<u32> {
    let mut scored: Vec<(u32, u32)> = Vec::new();
    for (i, ty) in types.iter().enumerate() {
        if i >= 32 || (candidate_mask & (1u32 << i)) == 0 {
            continue;
        }
        // Must hold every required flag.
        if !ty.property_flags.contains(prefs.required) {
            continue;
        }
        let cost = (prefs.preferred & !ty.property_flags).as_raw().count_ones()
            + (ty.property_flags & prefs.not_preferred)
                .as_raw()
                .count_ones();
        scored.push((i as u32, cost));
    }
    // Stable sort by cost keeps the lower index first among equal costs.
    scored.sort_by_key(|&(_, cost)| cost);
    scored.into_iter().map(|(i, _)| i).collect()
}

/// VMA's `CalcPreferredBlockSize` merged with the [`BlockSizePolicy`].
///
/// The VMA base size is `heap <= 1 GiB ? heap / 8 : 256 MiB`, aligned up to 32. It is
/// then capped by the `MemoryHints`-derived maximum for the memory class (device vs host
/// visible), so the observable block sizing tracks the old `gpu-allocator` block-size
/// policy while never exceeding what a small heap can hold.
///
/// # Merge rule
///
/// `min(align_up(vma_base, 32), hints_max)`, then floored at 32 so a pathologically small
/// heap still yields a usable (non-zero) block size. `hints_max` is the device or host
/// maximum depending on `host_visible`.
fn preferred_block_size(heap_size: u64, host_visible: bool, policy: BlockSizePolicy) -> u64 {
    let base = if heap_size <= SMALL_HEAP_MAX_SIZE {
        heap_size / 8
    } else {
        LARGE_HEAP_BLOCK_SIZE
    };
    // Align up to 32 (saturating), matching VmaAlignUp(.., 32).
    let aligned = base.saturating_add(31) & !31u64;
    let hints_max = if host_visible {
        policy.max_host_memblock_size
    } else {
        policy.max_device_memblock_size
    };
    aligned.min(hints_max).max(32)
}

/// The dedicated-allocation decision, a port of the heuristics in VMA's
/// `AllocateMemoryOfType` (`vk_mem_alloc.h`).
///
/// - If the driver *requires* a dedicated allocation, one is required.
/// - Otherwise a dedicated allocation is *preferred* when the driver prefers it, or when
///   `size > preferred_block_size / 2`.
/// - A mere preference is *suppressed* when the device is close to
///   `maxMemoryAllocationCount` (strictly above 3/4 of it, matching VMA's `>`), to avoid
///   exhausting the memory object budget — but a hard requirement is never suppressed.
///
/// Returns whether a dedicated allocation should be used.
fn should_use_dedicated(
    requires_dedicated: bool,
    prefers_dedicated: bool,
    size: u64,
    preferred_block_size: u64,
    memory_object_count: u64,
    max_memory_allocation_count: u32,
) -> bool {
    if requires_dedicated {
        return true;
    }
    let mut dedicated_preferred = prefers_dedicated || size > preferred_block_size / 2;

    // VMA: don't prefer dedicated when strictly above 3/4 of maxMemoryAllocationCount
    // (`vk_mem_alloc.h`'s `AllocateMemoryOfType` uses `>`, so exactly 3/4 is still allowed
    // to prefer dedicated). Guarded against the count being near u32::MAX, where the 3/4
    // product would overflow.
    if max_memory_allocation_count < u32::MAX / 4 {
        let threshold = (max_memory_allocation_count as u64) * 3 / 4;
        if memory_object_count > threshold {
            dedicated_preferred = false;
        }
    }
    dedicated_preferred
}

/// What the caller wants to bind a fresh allocation to, for a possible dedicated
/// allocation (`VkMemoryDedicatedAllocateInfo` names one of these).
///
/// Every resource this allocator serves (buffers, images, acceleration-structure
/// buffers) has a handle, so a dedicated allocation always names its resource. A
/// handle-less dedicated allocation is not needed here (the external/import paths in
/// `device.rs` allocate their own memory directly).
#[derive(Clone, Copy, Debug)]
pub(super) enum DedicatedHandle {
    /// Bind a dedicated allocation to this buffer.
    Buffer(vk::Buffer),
    /// Bind a dedicated allocation to this image.
    Image(vk::Image),
}

/// A fully-described allocation request handed to [`Allocator::allocate`].
#[derive(Clone, Copy, Debug)]
pub(super) struct AllocationRequest<'a> {
    /// Human-readable label, stored as the pool user-data and surfaced in reports.
    pub name: &'a str,
    /// The Vulkan memory requirements for the resource. `memory_type_bits` is ANDed with
    /// `valid_memory_types` inside [`allocate`](Allocator::allocate).
    pub requirements: vk::MemoryRequirements,
    /// The usage class (drives memory-type selection).
    pub usage: MemoryUsage,
    /// The suballocation type (buffer vs image tiling) for buffer-image granularity.
    pub alloc_type: AllocationType,
    /// The resource handle for a `VkMemoryDedicatedAllocateInfo`, if a dedicated
    /// allocation may be made.
    pub dedicated_handle: DedicatedHandle,
    /// Whether the driver *requires* a dedicated allocation for this resource
    /// (`requiresDedicatedAllocation`, or a lazily-allocated/TRANSIENT image).
    pub requires_dedicated: bool,
    /// Whether the driver *prefers* a dedicated allocation (`prefersDedicatedAllocation`).
    pub prefers_dedicated: bool,
}

impl Allocator {
    /// Builds the facade from the device's memory properties, limits, memory hints, and
    /// budget thresholds. Creates no device memory (pools are created lazily).
    pub(super) fn new(
        mem_properties: &vk::PhysicalDeviceMemoryProperties,
        limits: &vk::PhysicalDeviceLimits,
        valid_memory_types: u32,
        memory_hints: &wgt::MemoryHints,
        memory_budget_thresholds: MemoryBudgetThresholds,
        memory_budget_supported: bool,
        buffer_device_address: bool,
    ) -> Self {
        let type_count = (mem_properties.memory_type_count as usize).min(vk::MAX_MEMORY_TYPES);
        let heap_count = (mem_properties.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
        let types = mem_properties.memory_types[..type_count].to_vec();
        let heaps = mem_properties.memory_heaps[..heap_count].to_vec();

        let mut pools = Vec::with_capacity(types.len());
        pools.resize_with(types.len(), || None);

        let budget = BudgetState {
            // Force a refresh on the first operation.
            ops_since_refresh: BUDGET_REFRESH_INTERVAL,
            heap_budget: alloc::vec![0; heaps.len()],
            heap_usage_at_fetch: alloc::vec![0; heaps.len()],
            tracked_bytes_at_fetch: alloc::vec![0; heaps.len()],
            tracked_bytes: alloc::vec![0; heaps.len()],
        };

        Allocator {
            pools,
            mem_props: MemoryProperties { types, heaps },
            valid_memory_types,
            block_size_policy: BlockSizePolicy::from_memory_hints(memory_hints),
            max_memory_allocation_count: limits.max_memory_allocation_count,
            buffer_image_granularity: limits.buffer_image_granularity.max(1),
            non_coherent_atom_size: limits.non_coherent_atom_size.max(1),
            memory_budget_thresholds,
            memory_budget_supported,
            buffer_device_address,
            budget,
            memory_objects: Mutex::new(0),
            block_registry: Mutex::new(HashMap::new()),
            dedicated: Vec::new(),
        }
    }

    /// Whether the given memory type index is host-visible.
    fn is_host_visible(&self, memory_type_index: u32) -> bool {
        self.mem_props
            .types
            .get(memory_type_index as usize)
            .is_some_and(|t| {
                t.property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
    }

    /// The heap index for a memory type, or `None` if the index is out of range.
    fn heap_index_of(&self, memory_type_index: u32) -> Option<usize> {
        self.mem_props
            .types
            .get(memory_type_index as usize)
            .map(|t| t.heap_index as usize)
    }

    /// The property flags of a memory type, or empty if the index is out of range.
    fn type_flags(&self, memory_type_index: u32) -> vk::MemoryPropertyFlags {
        self.mem_props
            .types
            .get(memory_type_index as usize)
            .map(|t| t.property_flags)
            .unwrap_or_else(vk::MemoryPropertyFlags::empty)
    }

    /// A `MemTypeInfo` snapshot for the ranking function.
    fn type_infos(&self) -> Vec<MemTypeInfo> {
        self.mem_props
            .types
            .iter()
            .map(|t| MemTypeInfo {
                property_flags: t.property_flags,
                heap_index: t.heap_index,
            })
            .collect()
    }

    /// The number of live device-memory objects (blocks + dedicated).
    fn memory_object_count(&self) -> u64 {
        *self.memory_objects.lock()
    }

    /// Refreshes cached budget figures from the driver if the refresh interval elapsed.
    ///
    /// With `VK_EXT_memory_budget`, `heap_budget` / `heap_usage_at_fetch` come from the
    /// driver; without it, `heap_budget` is 80% of each heap's size and usage stays 0
    /// (effective usage is then just tracked bytes). Also snapshots the current tracked
    /// bytes so between-refresh estimates have a baseline.
    fn refresh_budget_if_due(&mut self, shared: &super::DeviceShared) {
        if self.budget.ops_since_refresh < BUDGET_REFRESH_INTERVAL {
            return;
        }
        self.budget.ops_since_refresh = 0;

        let heap_count = self.mem_props.heaps.len();

        // Try the driver's real figures via VK_EXT_memory_budget +
        // VK_KHR_get_physical_device_properties2.
        if self.memory_budget_supported {
            if let Some(get_device_properties) =
                shared.instance.get_physical_device_properties.as_ref()
            {
                let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
                let mut props2 =
                    vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget_props);
                // SAFETY: `physical_device` belongs to `shared.instance`; `props2` is a
                // valid output structure with a budget extension chained.
                unsafe {
                    get_device_properties.get_physical_device_memory_properties2(
                        shared.physical_device,
                        &mut props2,
                    );
                }
                for i in 0..heap_count {
                    self.budget.heap_budget[i] = budget_props.heap_budget[i];
                    self.budget.heap_usage_at_fetch[i] = budget_props.heap_usage[i];
                    self.budget.tracked_bytes_at_fetch[i] = self.budget.tracked_bytes[i];
                }
                return;
            }
        }

        // No trustworthy budget from the driver: budget = 80% of heap size, usage = 0
        // (effective usage becomes tracked bytes via the estimate path).
        for i in 0..heap_count {
            let heap_size = self.mem_props.heaps[i].size;
            self.budget.heap_budget[i] = heap_size / 100 * 80;
            self.budget.heap_usage_at_fetch[i] = 0;
            self.budget.tracked_bytes_at_fetch[i] = self.budget.tracked_bytes[i];
        }
    }

    /// The estimated current usage of a heap: the last fetched usage plus the net change
    /// in tracked bytes since the fetch (saturating), per the module's budget model.
    fn estimated_heap_usage(&self, heap_index: usize) -> u64 {
        let fetched = self
            .budget
            .heap_usage_at_fetch
            .get(heap_index)
            .copied()
            .unwrap_or(0);
        let tracked_now = self
            .budget
            .tracked_bytes
            .get(heap_index)
            .copied()
            .unwrap_or(0);
        let tracked_then = self
            .budget
            .tracked_bytes_at_fetch
            .get(heap_index)
            .copied()
            .unwrap_or(0);
        // effective = fetched + (tracked_now - tracked_then), saturating both ways.
        let grew = tracked_now.saturating_sub(tracked_then);
        let shrank = tracked_then.saturating_sub(tracked_now);
        fetched.saturating_add(grew).saturating_sub(shrank)
    }

    /// The cap (in bytes) at which resource creation on `heap_index` is refused, from
    /// `for_resource_creation`, or `None` if no threshold is configured.
    fn resource_creation_cap(&self, heap_index: usize) -> Option<u64> {
        let threshold = self.memory_budget_thresholds.for_resource_creation? as u64;
        let budget = self
            .budget
            .heap_budget
            .get(heap_index)
            .copied()
            .unwrap_or(0);
        Some(budget / 100 * threshold)
    }

    /// Whether creating `extra_bytes` of *new* device memory on `heap_index` would cross
    /// the `for_resource_creation` threshold. Suballocation within an existing block does
    /// not call this (that is the improvement over the deleted OOM predictor).
    fn would_exceed_resource_cap(&self, heap_index: usize, extra_bytes: u64) -> bool {
        let Some(cap) = self.resource_creation_cap(heap_index) else {
            return false;
        };
        let projected = self
            .estimated_heap_usage(heap_index)
            .saturating_add(extra_bytes);
        projected >= cap
    }

    /// The headroom (in bytes) left on `heap_index` before the `for_resource_creation`
    /// cap, fed to the pool as `AllocationContext::budget_free_bytes`. `None` when no
    /// threshold is configured (unlimited from the pool's perspective).
    fn heap_budget_free(&self, heap_index: usize) -> Option<u64> {
        let cap = self.resource_creation_cap(heap_index)?;
        Some(cap.saturating_sub(self.estimated_heap_usage(heap_index)))
    }

    /// Adds `delta` to the tracked bytes for `heap_index` (may be negative).
    fn adjust_tracked_bytes(&mut self, heap_index: usize, delta: i64) {
        if let Some(slot) = self.budget.tracked_bytes.get_mut(heap_index) {
            *slot = if delta >= 0 {
                slot.saturating_add(delta as u64)
            } else {
                slot.saturating_sub(delta.unsigned_abs())
            };
        }
    }

    /// Allocates device memory for a resource, suballocating from a pool where possible
    /// and falling back to a dedicated `vkAllocateMemory` where required, preferred, or
    /// when a pool cannot serve the request.
    ///
    /// Tries memory types in VMA cost order (see [`rank_memory_types`]); on failure it
    /// moves to the next-best type, so a full or budget-blocked type does not fail the
    /// whole allocation while another type could serve it.
    ///
    /// # Errors
    ///
    /// - [`DeviceError::OutOfMemory`] if no memory type can satisfy the request within
    ///   budget, or the driver returns out-of-memory.
    /// - [`DeviceError::Unexpected`] if no candidate memory type exists at all, or the
    ///   driver returns an unexpected error.
    pub(super) fn allocate(
        &mut self,
        shared: &super::DeviceShared,
        request: AllocationRequest<'_>,
    ) -> Result<Allocation, DeviceError> {
        self.refresh_budget_if_due(shared);
        self.budget.ops_since_refresh = self.budget.ops_since_refresh.saturating_add(1);

        let candidate_mask = request.requirements.memory_type_bits & self.valid_memory_types;
        let ranked = rank_memory_types(
            candidate_mask,
            &self.type_infos(),
            request.usage.preferences(),
        );
        if ranked.is_empty() {
            log::error!(
                "No compatible Vulkan memory type for request (mask {candidate_mask:#x}, usage {:?})",
                request.usage,
            );
            return Err(DeviceError::Unexpected);
        }

        // Remember the last "real" error so it can be surfaced if every type fails.
        let mut last_err = DeviceError::OutOfMemory;

        for memory_type_index in ranked {
            match self.allocate_from_type(shared, &request, memory_type_index) {
                Ok(allocation) => return Ok(allocation),
                Err(e) => last_err = e,
            }
        }

        Err(last_err)
    }

    /// Attempts to allocate from a single memory type: decides pooled vs dedicated, then
    /// carries it out. Returns `Err` (to try the next type) on any failure.
    fn allocate_from_type(
        &mut self,
        shared: &super::DeviceShared,
        request: &AllocationRequest<'_>,
        memory_type_index: u32,
    ) -> Result<Allocation, DeviceError> {
        let heap_index = self
            .heap_index_of(memory_type_index)
            .ok_or(DeviceError::Unexpected)?;
        let host_visible = self.is_host_visible(memory_type_index);
        let heap_size = self
            .mem_props
            .heaps
            .get(heap_index)
            .map(|h| h.size)
            .unwrap_or(0);
        let preferred_block = preferred_block_size(heap_size, host_visible, self.block_size_policy);

        let dedicated = should_use_dedicated(
            request.requires_dedicated,
            request.prefers_dedicated,
            request.requirements.size,
            preferred_block,
            self.memory_object_count(),
            self.max_memory_allocation_count,
        );

        if dedicated {
            return self.allocate_dedicated(shared, request, memory_type_index, heap_index);
        }

        // Pooled path. If the pool signals ShouldDedicate (request too large for a block,
        // or budget-gated with a dedicated fallback allowed), fall back to dedicated.
        match self.allocate_pooled(
            shared,
            request,
            memory_type_index,
            heap_index,
            preferred_block,
            host_visible,
        ) {
            Ok(allocation) => Ok(allocation),
            Err(PooledOutcome::ShouldDedicate) => {
                self.allocate_dedicated(shared, request, memory_type_index, heap_index)
            }
            Err(PooledOutcome::Error(e)) => Err(e),
        }
    }
}

/// The failure modes of [`Allocator::allocate_pooled`]: either the pool asked for a
/// dedicated fallback, or a genuine error occurred.
enum PooledOutcome {
    /// The pool cannot serve this request; try a dedicated allocation.
    ShouldDedicate,
    /// A real error (out of memory, or a backend/device failure).
    Error(DeviceError),
}

impl Allocator {
    /// Suballocates from the (lazily-created) pool for `memory_type_index`.
    #[allow(clippy::too_many_arguments)]
    fn allocate_pooled(
        &mut self,
        shared: &super::DeviceShared,
        request: &AllocationRequest<'_>,
        memory_type_index: u32,
        heap_index: usize,
        preferred_block: u64,
        host_visible: bool,
    ) -> Result<Allocation, PooledOutcome> {
        // Budget gate for a *new* block: if we're already over the resource-creation cap,
        // decline to grow (an existing block can still be suballocated from, which the
        // pool decides on its own). We only fully block the type when no block exists yet.
        let budget_free = self.heap_budget_free(heap_index);

        // Lazily create the pool.
        if self.pool_missing(memory_type_index) {
            let config = self.make_pool_config(memory_type_index, preferred_block, host_visible);
            let mut backend = self.make_backend(shared, memory_type_index, host_visible);
            match Pool::new(config, &mut backend) {
                Ok(pool) => self.set_pool(memory_type_index, pool),
                Err(e) => return Err(PooledOutcome::Error(map_pool_alloc_error(e))),
            }
        }

        let ctx = AllocationContext {
            budget_free_bytes: budget_free,
            // Every request carries a resource handle, so a dedicated fallback is always
            // possible if the pool declines to grow.
            dedicated_fallback_allowed: true,
            preferred_affinity: None,
        };

        let label: Option<Arc<str>> = if request.name.is_empty() {
            None
        } else {
            Some(Arc::from(request.name))
        };

        let mut backend = self.make_backend(shared, memory_type_index, host_visible);
        // Measure the pool's total block bytes before and after so budget tracking
        // reflects exactly the device memory the pool created (or reused).
        let block_bytes_before = self
            .pool(memory_type_index)
            .map(|p| p.statistics().block_bytes)
            .unwrap_or(0);

        let alloc_result = {
            let pool = match self.pool_mut(memory_type_index) {
                Some(pool) => pool,
                None => return Err(PooledOutcome::Error(DeviceError::Unexpected)),
            };
            pool.allocate(
                AllocationDesc {
                    size: request.requirements.size,
                    alignment: request.requirements.alignment.max(1),
                    alloc_type: request.alloc_type,
                    strategy: Strategy::Balanced,
                    upper_address: false,
                },
                ctx,
                label,
                &mut backend,
            )
        };

        match alloc_result {
            Ok(pool_alloc) => {
                let block_bytes_after = self
                    .pool(memory_type_index)
                    .map(|p| p.statistics().block_bytes)
                    .unwrap_or(block_bytes_before);
                let grew = block_bytes_after.saturating_sub(block_bytes_before);
                if grew > 0 {
                    self.adjust_tracked_bytes(heap_index, grew as i64);
                }
                let allocation = self
                    .build_pooled_allocation(memory_type_index, pool_alloc)
                    .ok_or(PooledOutcome::Error(DeviceError::Unexpected))?;
                Ok(allocation)
            }
            Err(PoolAllocError::ShouldDedicate) => Err(PooledOutcome::ShouldDedicate),
            Err(e) => Err(PooledOutcome::Error(map_pool_alloc_error(e))),
        }
    }

    // --- pool bookkeeping helpers -------------------------------------------------

    /// Whether the pool for `memory_type_index` has not been created yet.
    fn pool_missing(&self, memory_type_index: u32) -> bool {
        self.pools
            .get(memory_type_index as usize)
            .map(|p| p.is_none())
            .unwrap_or(true)
    }

    /// Shared reference to the pool for `memory_type_index`, if it exists.
    fn pool(&self, memory_type_index: u32) -> Option<&Pool<VkBlockBackend, Option<Arc<str>>>> {
        self.pools
            .get(memory_type_index as usize)
            .and_then(|p| p.as_ref())
    }

    /// Mutable reference to the pool for `memory_type_index`, if it exists.
    fn pool_mut(
        &mut self,
        memory_type_index: u32,
    ) -> Option<&mut Pool<VkBlockBackend, Option<Arc<str>>>> {
        self.pools
            .get_mut(memory_type_index as usize)
            .and_then(|p| p.as_mut())
    }

    /// Stores a freshly created pool for `memory_type_index`.
    fn set_pool(&mut self, memory_type_index: u32, pool: Pool<VkBlockBackend, Option<Arc<str>>>) {
        if let Some(slot) = self.pools.get_mut(memory_type_index as usize) {
            *slot = Some(pool);
        }
    }

    /// Builds the [`PoolConfig`] for a memory type.
    ///
    /// - TLSF algorithm, one block vector per type.
    /// - `granularity` = `bufferImageGranularity` so buffers and optimal-tiling images
    ///   never share a granularity page within a block.
    /// - `min_allocation_alignment` = `nonCoherentAtomSize` for host-visible pools so
    ///   mapped-range flush/invalidate alignment is always satisfiable.
    /// - `affinity_clustering` disabled: every host-visible block is persistently mapped,
    ///   so there is no unmapped/mapped distinction to cluster on.
    /// - `pool_salt` is a unique process-wide value so allocations cannot be freed to the
    ///   wrong pool.
    fn make_pool_config(
        &self,
        _memory_type_index: u32,
        preferred_block: u64,
        host_visible: bool,
    ) -> PoolConfig {
        let salt = NEXT_POOL_SALT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        PoolConfig {
            algorithm: Algorithm::Tlsf,
            preferred_block_size: preferred_block.max(1),
            min_block_count: 0,
            // Enough blocks that a single memory type never runs out of block slots in
            // practice; the budget gate is the real limiter on growth.
            max_block_count: usize::MAX,
            explicit_block_size: false,
            min_allocation_alignment: if host_visible {
                self.non_coherent_atom_size
            } else {
                1
            },
            granularity: self.buffer_image_granularity,
            debug_margin: 0,
            affinity_clustering: false,
            pool_salt: salt,
        }
    }

    /// Builds a short-lived [`VkBlockBackend`] bound to `shared`'s device for a pool call.
    fn make_backend(
        &self,
        shared: &super::DeviceShared,
        memory_type_index: u32,
        host_visible: bool,
    ) -> VkBlockBackend {
        // SAFETY: the returned backend is used only for the duration of the current
        // `&mut self` call; `shared.raw` (the device), `self.memory_objects`, and
        // `self.block_registry` all outlive that call because they are owned by the same
        // `Device` that owns `self`.
        unsafe {
            VkBlockBackend::new(
                &shared.raw,
                memory_type_index,
                host_visible,
                self.buffer_device_address,
                &self.memory_objects,
                &self.block_registry,
            )
        }
    }

    /// Turns a pool-layer allocation into the module's [`Allocation`], resolving the
    /// backing memory and mapped pointer from the pool's block metadata.
    ///
    /// Returns `None` only if the block cannot be found (which would be an internal
    /// inconsistency) — the caller maps that to [`DeviceError::Unexpected`].
    fn build_pooled_allocation(
        &self,
        memory_type_index: u32,
        pool_alloc: wgpu_block_pool::Allocation,
    ) -> Option<Allocation> {
        // Resolve the block's memory and mapped base from the registry.
        let block = *self.block_registry.lock().get(&pool_alloc.block_id())?;
        let memory = block.memory;
        let block_size = block.size;

        // Compute the mapped pointer for this allocation, bounds-checked.
        let mapped_ptr = match block.mapped {
            Some(base) => {
                // offset + size must fit within the block.
                let end = pool_alloc.offset().checked_add(pool_alloc.size())?;
                if end > block_size {
                    return None;
                }
                // SAFETY: `base` is the block's mapped base; `offset` is within the block
                // (checked just above), so the resulting pointer is within the mapping.
                let ptr = unsafe { base.as_ptr().add(pool_alloc.offset() as usize) };
                NonNull::new(ptr)
            }
            None => None,
        };

        Some(Allocation {
            offset: pool_alloc.offset(),
            size: pool_alloc.size(),
            kind: AllocationKind::Pooled(pool_alloc),
            memory,
            mapped_ptr,
            memory_properties: self.type_flags(memory_type_index),
            memory_type_index,
        })
    }

    // --- dedicated path -----------------------------------------------------------

    /// Makes a dedicated `vkAllocateMemory` allocation (bypassing the pools), optionally
    /// with a `VkMemoryDedicatedAllocateInfo` naming the buffer/image, and maps it if the
    /// memory is host-visible.
    fn allocate_dedicated(
        &mut self,
        shared: &super::DeviceShared,
        request: &AllocationRequest<'_>,
        memory_type_index: u32,
        heap_index: usize,
    ) -> Result<Allocation, DeviceError> {
        let size = request.requirements.size;

        // Budget gate for a new device-memory object.
        if self.would_exceed_resource_cap(heap_index, size) {
            return Err(DeviceError::OutOfMemory);
        }

        let host_visible = self.is_host_visible(memory_type_index);

        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default();
        dedicated_info = match request.dedicated_handle {
            DedicatedHandle::Buffer(buffer) => dedicated_info.buffer(buffer),
            DedicatedHandle::Image(image) => dedicated_info.image(image),
        };

        let memory = allocate_device_memory(
            &shared.raw,
            size,
            memory_type_index,
            self.buffer_device_address,
            Some(&mut dedicated_info),
        )?;

        let mapped_ptr = if host_visible {
            // SAFETY: `memory` was allocated from a host-visible type; mapping its whole
            // range is valid. On failure the memory is freed before returning.
            match unsafe {
                shared
                    .raw
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(ptr) => NonNull::new(ptr.cast::<u8>()),
                Err(e) => {
                    // SAFETY: `memory` is live and has no bound resources yet.
                    unsafe { shared.raw.free_memory(memory, None) };
                    return Err(super::map_host_device_oom_err(e));
                }
            }
        } else {
            None
        };

        increment_memory_objects(&self.memory_objects);
        self.adjust_tracked_bytes(heap_index, size as i64);

        let label: Option<Arc<str>> = if request.name.is_empty() {
            None
        } else {
            Some(Arc::from(request.name))
        };
        self.dedicated.push(DedicatedRecord {
            memory,
            size,
            label,
        });

        Ok(Allocation {
            kind: AllocationKind::Dedicated,
            memory,
            offset: 0,
            size,
            mapped_ptr,
            memory_properties: self.type_flags(memory_type_index),
            memory_type_index,
        })
    }

    // --- free ---------------------------------------------------------------------

    /// Frees an allocation made by [`allocate`](Self::allocate).
    ///
    /// For a pooled allocation this frees back to the pool; if the free emptied a block
    /// the pool wants dropped, the block is unmapped and `vkFreeMemory`d here (after the
    /// pool bookkeeping, still under the caller's `Mutex`). For a dedicated allocation the
    /// device memory is unmapped and freed directly.
    ///
    /// Never panics: a stale/foreign pooled allocation is logged and ignored.
    pub(super) fn free(&mut self, shared: &super::DeviceShared, allocation: Allocation) {
        self.refresh_budget_if_due(shared);
        self.budget.ops_since_refresh = self.budget.ops_since_refresh.saturating_add(1);

        let heap_index = self.heap_index_of(allocation.memory_type_index);

        match allocation.kind {
            AllocationKind::Pooled(pool_alloc) => {
                let memory_type_index = allocation.memory_type_index;
                let budget_exceeded = heap_index
                    .map(|h| self.would_exceed_resource_cap(h, 0))
                    .unwrap_or(false);

                let block_bytes_before = self
                    .pool(memory_type_index)
                    .map(|p| p.statistics().block_bytes)
                    .unwrap_or(0);

                let free_result = match self.pool_mut(memory_type_index) {
                    Some(pool) => pool.free(pool_alloc, FreeContext { budget_exceeded }),
                    None => {
                        log::error!(
                            "Vulkan allocation freed against a memory type with no pool ({memory_type_index})"
                        );
                        return;
                    }
                };

                match free_result {
                    Ok(outcome) => {
                        // Destroy any block the pool dropped (its device memory). The pool
                        // hands the block back instead of calling `destroy_block`, so its
                        // registry entry must be removed here as well.
                        if let Some((block, block_id)) = outcome.destroyed_block {
                            self.block_registry.lock().remove(&block_id);
                            destroy_vk_block(&shared.raw, block);
                            decrement_memory_objects(&self.memory_objects);
                        }
                        let block_bytes_after = self
                            .pool(memory_type_index)
                            .map(|p| p.statistics().block_bytes)
                            .unwrap_or(block_bytes_before);
                        let shrank = block_bytes_before.saturating_sub(block_bytes_after);
                        if shrank > 0 {
                            if let Some(h) = heap_index {
                                self.adjust_tracked_bytes(h, -(shrank as i64));
                            }
                        }
                    }
                    Err(_) => {
                        log::error!(
                            "Failed to free Vulkan suballocation (stale or foreign handle)"
                        );
                    }
                }
            }
            AllocationKind::Dedicated => {
                // Remove the record and free the memory.
                if let Some(pos) = self
                    .dedicated
                    .iter()
                    .position(|d| d.memory == allocation.memory)
                {
                    self.dedicated.swap_remove(pos);
                }
                if allocation.mapped_ptr.is_some() {
                    // SAFETY: dedicated host-visible memory was mapped at allocation and
                    // not unmapped since.
                    unsafe { shared.raw.unmap_memory(allocation.memory) };
                }
                // SAFETY: `allocation.memory` is a live dedicated allocation owned by this
                // module; the resource bound to it was destroyed by the caller first.
                unsafe { shared.raw.free_memory(allocation.memory, None) };
                decrement_memory_objects(&self.memory_objects);
                if let Some(h) = heap_index {
                    self.adjust_tracked_bytes(h, -(allocation.size as i64));
                }
            }
        }
    }

    // --- external / query-pool budget probe --------------------------------------

    /// Preserves the query-pool OOM behaviour of the deleted predictor for allocations
    /// that do **not** go through this allocator (e.g. `vkCreateQueryPool`, which manages
    /// its own memory).
    ///
    /// Returns [`DeviceError::OutOfMemory`] when the estimated `estimated_size` bytes on
    /// the best-fit heap for `usage` would cross the `for_resource_creation` threshold.
    /// A no-op when no threshold is configured.
    pub(super) fn check_external_allocation(
        &mut self,
        shared: &super::DeviceShared,
        usage: MemoryUsage,
        estimated_size: u64,
    ) -> Result<(), DeviceError> {
        if self
            .memory_budget_thresholds
            .for_resource_creation
            .is_none()
        {
            return Ok(());
        }
        self.refresh_budget_if_due(shared);
        self.budget.ops_since_refresh = self.budget.ops_since_refresh.saturating_add(1);

        // Choose the heap the same way the allocator would: the best-ranked memory type
        // among all valid types for this usage.
        let ranked = rank_memory_types(
            self.valid_memory_types,
            &self.type_infos(),
            usage.preferences(),
        );
        let Some(&best) = ranked.first() else {
            // No suitable heap; mirror the predictor's conservative OOM.
            return Err(DeviceError::OutOfMemory);
        };
        let Some(heap_index) = self.heap_index_of(best) else {
            return Err(DeviceError::OutOfMemory);
        };

        if self.would_exceed_resource_cap(heap_index, estimated_size) {
            log::warn!(
                "External allocation of {estimated_size}B would exceed the resource-creation budget on heap {heap_index}"
            );
            return Err(DeviceError::OutOfMemory);
        }
        Ok(())
    }

    // --- report -------------------------------------------------------------------

    /// Builds a [`wgt::AllocatorReport`] across all pools and dedicated allocations.
    pub(super) fn generate_report(&self) -> wgt::AllocatorReport {
        let mut allocations: Vec<wgt::AllocationReport> = Vec::new();
        let mut blocks: Vec<wgt::MemoryBlockReport> = Vec::new();
        let mut total_allocated_bytes: u64 = 0;
        let mut total_reserved_bytes: u64 = 0;

        for pool in self.pools.iter().flatten() {
            // One MemoryBlockReport per block, with the range of allocations it owns.
            for report_block in pool.report().blocks {
                let start = allocations.len();
                pool.for_each_allocation(|block_id, offset, size, user_data| {
                    if block_id == report_block.block_id {
                        allocations.push(wgt::AllocationReport {
                            name: user_data
                                .as_ref()
                                .map(|s| s.as_ref().to_owned())
                                .unwrap_or_default(),
                            offset,
                            size,
                        });
                        total_allocated_bytes = total_allocated_bytes.saturating_add(size);
                    }
                });
                let end = allocations.len();
                total_reserved_bytes = total_reserved_bytes.saturating_add(report_block.size);
                blocks.push(wgt::MemoryBlockReport {
                    size: report_block.size,
                    allocations: start..end,
                });
            }
        }

        // Dedicated allocations: each is its own block with a single allocation.
        for record in &self.dedicated {
            let start = allocations.len();
            allocations.push(wgt::AllocationReport {
                name: record
                    .label
                    .as_ref()
                    .map(|s| s.as_ref().to_owned())
                    .unwrap_or_default(),
                offset: 0,
                size: record.size,
            });
            let end = allocations.len();
            total_allocated_bytes = total_allocated_bytes.saturating_add(record.size);
            total_reserved_bytes = total_reserved_bytes.saturating_add(record.size);
            blocks.push(wgt::MemoryBlockReport {
                size: record.size,
                allocations: start..end,
            });
        }

        wgt::AllocatorReport {
            allocations,
            blocks,
            total_allocated_bytes,
            total_reserved_bytes,
        }
    }

    /// Destroys every pool block and dedicated allocation. Called from [`Device`](super::Device)'s drop
    /// path (via [`cleanup`](Self::cleanup)); after this the allocator holds no device
    /// memory.
    pub(super) fn cleanup(&mut self, shared: &super::DeviceShared) {
        // Drain the pools, destroying every remaining block.
        for slot in &mut self.pools {
            if let Some(pool) = slot.take() {
                for (block, _id) in pool.into_blocks() {
                    destroy_vk_block(&shared.raw, block);
                }
            }
        }
        self.block_registry.lock().clear();
        // Free dedicated allocations. `vkFreeMemory` implicitly unmaps any mapping, so an
        // explicit `vkUnmapMemory` is unnecessary here (and we do not track per-record
        // mapping state).
        for record in self.dedicated.drain(..) {
            // SAFETY: `record.memory` is a live dedicated allocation owned by this module,
            // and by the drop ordering no resource still references it.
            unsafe { shared.raw.free_memory(record.memory, None) };
        }
        *self.memory_objects.lock() = 0;
    }
}

/// Maps a [`PoolAllocError`] to a [`DeviceError`]: out-of-memory conditions become
/// [`DeviceError::OutOfMemory`]; errors the pool should never report for the requests
/// this module makes (invalid request, upper address) are logged and mapped to
/// [`DeviceError::Unexpected`].
fn map_pool_alloc_error(err: PoolAllocError<DeviceError>) -> DeviceError {
    match err {
        PoolAllocError::OutOfPoolMemory | PoolAllocError::ShouldDedicate => {
            DeviceError::OutOfMemory
        }
        PoolAllocError::Backend(e) => e,
        PoolAllocError::InvalidRequest | PoolAllocError::UpperAddressUnsupported => {
            log::error!("wgpu-block-pool rejected an allocation request: {err}");
            DeviceError::Unexpected
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::MemoryPropertyFlags as F;

    fn ty(flags: F, heap: u32) -> MemTypeInfo {
        MemTypeInfo {
            property_flags: flags,
            heap_index: heap,
        }
    }

    #[test]
    fn preferences_match_vma_auto_style() {
        let g = MemoryUsage::GpuOnly.preferences();
        assert_eq!(g.required, F::empty());
        assert_eq!(g.preferred, F::DEVICE_LOCAL);
        assert_eq!(g.not_preferred, F::empty());

        let w = MemoryUsage::CpuToGpu.preferences();
        assert_eq!(w.required, F::HOST_VISIBLE);
        assert_eq!(w.preferred, F::DEVICE_LOCAL);
        assert_eq!(w.not_preferred, F::HOST_CACHED);

        let r = MemoryUsage::GpuToCpu.preferences();
        assert_eq!(r.required, F::HOST_VISIBLE);
        assert_eq!(r.preferred, F::HOST_CACHED);
        assert_eq!(r.not_preferred, F::empty());
    }

    #[test]
    fn rank_prefers_device_local_for_gpu_only() {
        // type 0: host-visible only; type 1: device-local. GpuOnly prefers device-local.
        let types = [
            ty(F::HOST_VISIBLE | F::HOST_COHERENT, 0),
            ty(F::DEVICE_LOCAL, 1),
        ];
        let ranked = rank_memory_types(0b11, &types, MemoryUsage::GpuOnly.preferences());
        assert_eq!(ranked, alloc::vec![1, 0]);
    }

    #[test]
    fn rank_excludes_types_missing_required_flags() {
        // Only type 1 is host-visible; a MAP_WRITE (CpuToGpu) request requires it.
        let types = [ty(F::DEVICE_LOCAL, 0), ty(F::HOST_VISIBLE, 1)];
        let ranked = rank_memory_types(0b11, &types, MemoryUsage::CpuToGpu.preferences());
        assert_eq!(ranked, alloc::vec![1]);
    }

    #[test]
    fn rank_respects_candidate_mask() {
        let types = [ty(F::DEVICE_LOCAL, 0), ty(F::DEVICE_LOCAL, 1)];
        // Only type 0 allowed by the mask.
        let ranked = rank_memory_types(0b01, &types, MemoryUsage::GpuOnly.preferences());
        assert_eq!(ranked, alloc::vec![0]);
    }

    #[test]
    fn rank_readback_prefers_host_cached() {
        // Both host-visible; type 1 additionally host-cached. GpuToCpu prefers cached.
        let types = [
            ty(F::HOST_VISIBLE | F::HOST_COHERENT, 0),
            ty(F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED, 1),
        ];
        let ranked = rank_memory_types(0b11, &types, MemoryUsage::GpuToCpu.preferences());
        assert_eq!(ranked[0], 1);
    }

    #[test]
    fn rank_empty_when_no_type_satisfies() {
        let types = [ty(F::DEVICE_LOCAL, 0)];
        // CpuToGpu requires HOST_VISIBLE, which no type has.
        let ranked = rank_memory_types(0b1, &types, MemoryUsage::CpuToGpu.preferences());
        assert!(ranked.is_empty());
    }

    #[test]
    fn rank_ignores_zero_mask() {
        let types = [ty(F::DEVICE_LOCAL, 0)];
        assert!(rank_memory_types(0, &types, MemoryUsage::GpuOnly.preferences()).is_empty());
    }

    #[test]
    fn preferred_block_size_small_heap() {
        let policy = BlockSizePolicy {
            max_device_memblock_size: 256 * MB,
            max_host_memblock_size: 128 * MB,
        };
        // 512 MiB heap (<= 1 GiB): base = heap / 8 = 64 MiB, under both caps.
        assert_eq!(preferred_block_size(512 * MB, false, policy), 64 * MB);
    }

    #[test]
    fn preferred_block_size_large_heap_capped_by_hints() {
        let policy = BlockSizePolicy {
            max_device_memblock_size: 128 * MB,
            max_host_memblock_size: 32 * MB,
        };
        // 8 GiB heap (> 1 GiB): base = 256 MiB, but device cap is 128 MiB.
        assert_eq!(preferred_block_size(8 * 1024 * MB, false, policy), 128 * MB);
        // Host-visible uses the host cap.
        assert_eq!(preferred_block_size(8 * 1024 * MB, true, policy), 32 * MB);
    }

    #[test]
    fn preferred_block_size_floored_at_32() {
        let policy = BlockSizePolicy {
            max_device_memblock_size: 256 * MB,
            max_host_memblock_size: 128 * MB,
        };
        // Tiny heap: heap / 8 could be < 32; floor keeps it usable.
        assert_eq!(preferred_block_size(64, false, policy), 32);
    }

    #[test]
    fn preferred_block_size_no_overflow_on_max_heap() {
        let policy = BlockSizePolicy {
            max_device_memblock_size: u64::MAX,
            max_host_memblock_size: u64::MAX,
        };
        // Should not panic; large heap path returns 256 MiB aligned.
        assert_eq!(preferred_block_size(u64::MAX, false, policy), 256 * MB);
    }

    #[test]
    fn dedicated_required_always_wins() {
        // Even near the allocation-count ceiling, a hard requirement is honoured.
        assert!(should_use_dedicated(true, false, 1, 1024, 1_000_000, 100));
    }

    #[test]
    fn dedicated_preferred_by_large_size() {
        // size > preferred/2 prefers dedicated.
        assert!(should_use_dedicated(false, false, 600, 1000, 0, 4096));
        // size <= preferred/2 does not.
        assert!(!should_use_dedicated(false, false, 400, 1000, 0, 4096));
    }

    #[test]
    fn dedicated_preference_suppressed_near_alloc_limit() {
        // VMA (`vk_mem_alloc.h` `AllocateMemoryOfType`) suppresses a dedicated *preference*
        // only when the memory-object count is *strictly greater than* 3/4 of
        // maxMemoryAllocationCount (`m_DeviceMemoryCount.load() > max * 3 / 4`). For
        // max = 4096, 3/4 * max = 3072.

        // Exactly at 3/4 (3072): NOT suppressed — the preference still stands (VMA uses `>`,
        // not `>=`). This is the boundary the M2 fix corrects.
        assert!(should_use_dedicated(false, true, 1, 1024, 3072, 4096));
        // Strictly above 3/4 (3073): suppressed.
        assert!(!should_use_dedicated(false, true, 1, 1024, 3073, 4096));
        // Just below 3/4 (3071): allowed.
        assert!(should_use_dedicated(false, true, 1, 1024, 3071, 4096));
    }

    #[test]
    fn dedicated_suppression_skipped_for_huge_max_count() {
        // max near u32::MAX: the guard skips suppression, preference stands.
        assert!(should_use_dedicated(
            false,
            true,
            1,
            1024,
            u64::MAX,
            u32::MAX
        ));
    }

    #[test]
    fn memory_object_counter_helpers_are_reentrancy_safe() {
        // Regression test for finding C1: the dedicated-allocation path once bumped the
        // counter with `*m.lock() = m.lock().saturating_add(1)`, which holds two guards of
        // the same non-reentrant `Mutex` alive within one statement and
        // deadlocks permanently. The helpers take and release the lock once each, so calling
        // them back to back (as the real allocate/free paths do) must complete instantly.
        //
        // With the old broken code inlined here this test would hang and be killed by
        // nextest's timeout; with the helpers it returns immediately.
        let counter = Mutex::new(0u64);
        increment_memory_objects(&counter);
        increment_memory_objects(&counter);
        assert_eq!(*counter.lock(), 2);
        decrement_memory_objects(&counter);
        assert_eq!(*counter.lock(), 1);
        // Saturating behaviour: decrementing past zero stays at zero.
        decrement_memory_objects(&counter);
        decrement_memory_objects(&counter);
        assert_eq!(*counter.lock(), 0);
    }

    #[test]
    fn block_size_policy_from_hints() {
        let perf = BlockSizePolicy::from_memory_hints(&wgt::MemoryHints::Performance);
        assert_eq!(perf.max_device_memblock_size, 256 * MB);
        assert_eq!(perf.max_host_memblock_size, 128 * MB);

        let mem = BlockSizePolicy::from_memory_hints(&wgt::MemoryHints::MemoryUsage);
        assert_eq!(mem.max_device_memblock_size, 64 * MB);
        assert_eq!(mem.max_host_memblock_size, 32 * MB);

        let manual = BlockSizePolicy::from_memory_hints(&wgt::MemoryHints::Manual {
            suballocated_device_memory_block_size: (16 * MB)..(96 * MB),
        });
        // device end 96 MiB within [4, 256] MiB; host end = 48 MiB.
        assert_eq!(manual.max_device_memblock_size, 96 * MB);
        assert_eq!(manual.max_host_memblock_size, 48 * MB);
    }
}
