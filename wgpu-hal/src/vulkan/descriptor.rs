// Copyright 2020 The gpu-descriptor Project Developers
//
// Copied and modified from https://github.com/zakarumych/gpu-descriptor
// which is licenced under MIT/APACHE

use alloc::{collections::VecDeque, vec::Vec};
use arrayvec::ArrayVec;
use ash::vk;
use core::{
    convert::TryFrom as _,
    fmt::{self, Debug, Display},
};
use hashbrown::HashMap;

bitflags::bitflags! {
    /// Flags to augment descriptor pool creation.
    ///
    /// Match corresponding bits in Vulkan.
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
    pub struct DescriptorPoolCreateFlags: u32 {
        /// Allows freeing individual sets.
        const FREE_DESCRIPTOR_SET = 0x1;

        /// Allows allocating sets with layout created with matching backend-specific flag.
        const UPDATE_AFTER_BIND = 0x2;
    }
}

/// Number of descriptors of each type.
///
/// For `InlineUniformBlock` this value is number of bytes instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DescriptorTotalCount {
    pub sampler: u32,
    pub combined_image_sampler: u32,
    pub sampled_image: u32,
    pub storage_image: u32,
    pub uniform_texel_buffer: u32,
    pub storage_texel_buffer: u32,
    pub uniform_buffer: u32,
    pub storage_buffer: u32,
    pub uniform_buffer_dynamic: u32,
    pub storage_buffer_dynamic: u32,
    pub input_attachment: u32,
    pub acceleration_structure: u32,
    pub inline_uniform_block_bytes: u32,
    pub inline_uniform_block_bindings: u32,
}

impl DescriptorTotalCount {
    pub fn total(&self) -> u32 {
        self.sampler
            + self.combined_image_sampler
            + self.sampled_image
            + self.storage_image
            + self.uniform_texel_buffer
            + self.storage_texel_buffer
            + self.uniform_buffer
            + self.storage_buffer
            + self.uniform_buffer_dynamic
            + self.storage_buffer_dynamic
            + self.input_attachment
            + self.acceleration_structure
            + self.inline_uniform_block_bytes
            + self.inline_uniform_block_bindings
    }
}

/// Memory exhausted error.
#[derive(Debug)]
pub enum CreatePoolError {
    /// Device memory exhausted.
    OutOfDeviceMemory,

    /// Host memory exhausted.
    OutOfHostMemory,

    /// A descriptor pool creation has failed due to fragmentation.
    Fragmentation,

    Unexpected,
}

/// Memory exhausted error.
#[derive(Debug)]
pub enum DeviceAllocationError {
    /// Device memory exhausted.
    OutOfDeviceMemory,

    /// Host memory exhausted.
    OutOfHostMemory,

    /// Pool allocation failed due to fragmentation of pool's memory.
    FragmentedPool,

    Unexpected,
}

impl super::DeviceShared {
    /// Creates a new descriptor pool.
    fn create_descriptor_pool(
        &self,
        descriptor_count: &DescriptorTotalCount,
        max_sets: u32,
        flags: DescriptorPoolCreateFlags,
    ) -> Result<vk::DescriptorPool, CreatePoolError> {
        //Note: ignoring other types, since they can't appear here
        let unfiltered_counts = [
            (vk::DescriptorType::SAMPLER, descriptor_count.sampler),
            (
                vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count.sampled_image,
            ),
            (
                vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count.storage_image,
            ),
            (
                vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count.uniform_buffer,
            ),
            (
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count.uniform_buffer_dynamic,
            ),
            (
                vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count.storage_buffer,
            ),
            (
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count.storage_buffer_dynamic,
            ),
            (
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count.acceleration_structure,
            ),
        ];

        let filtered_counts = unfiltered_counts
            .iter()
            .cloned()
            .filter(|&(_, count)| count != 0)
            .map(|(ty, count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count: count,
            })
            .collect::<ArrayVec<_, 8>>();

        let mut vk_flags = if flags.contains(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND) {
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
        } else {
            vk::DescriptorPoolCreateFlags::empty()
        };
        if flags.contains(DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET) {
            vk_flags |= vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        }
        let vk_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .flags(vk_flags)
            .pool_sizes(&filtered_counts);

        match unsafe { self.raw.create_descriptor_pool(&vk_info, None) } {
            Ok(pool) => Ok(pool),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => Err(CreatePoolError::OutOfHostMemory),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => Err(CreatePoolError::OutOfDeviceMemory),
            Err(vk::Result::ERROR_FRAGMENTATION) => Err(CreatePoolError::Fragmentation),
            Err(err) => {
                log::error!("Unexpected Vulkan error: `{err}`");
                Err(CreatePoolError::Unexpected)
            }
        }
    }

    /// Allocates descriptor sets.
    ///
    /// # Safety
    ///
    /// Pool must be created from this device.
    unsafe fn alloc_descriptor_sets<'a>(
        &self,
        pool: &mut vk::DescriptorPool,
        layouts: impl ExactSizeIterator<Item = &'a vk::DescriptorSetLayout>,
    ) -> Result<Vec<vk::DescriptorSet>, DeviceAllocationError> {
        let result = unsafe {
            self.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*pool)
                    .set_layouts(
                        &smallvec::SmallVec::<[vk::DescriptorSetLayout; 32]>::from_iter(
                            layouts.cloned(),
                        ),
                    ),
            )
        };

        match result {
            Ok(vk_sets) => Ok(vk_sets),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
            | Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                Err(DeviceAllocationError::OutOfHostMemory)
            }
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(DeviceAllocationError::OutOfDeviceMemory)
            }
            Err(vk::Result::ERROR_FRAGMENTED_POOL) => Err(DeviceAllocationError::FragmentedPool),
            Err(err) => {
                log::error!("Unexpected Vulkan error: `{err}`");
                Err(DeviceAllocationError::Unexpected)
            }
        }
    }

    /// Deallocates descriptor sets.
    ///
    /// # Safety
    ///
    /// Sets must be allocated from specified pool and not deallocated before.
    unsafe fn dealloc_descriptor_sets(
        &self,
        pool: &mut vk::DescriptorPool,
        sets: impl Iterator<Item = vk::DescriptorSet>,
    ) {
        let result = unsafe {
            self.raw.free_descriptor_sets(
                *pool,
                &smallvec::SmallVec::<[vk::DescriptorSet; 32]>::from_iter(sets),
            )
        };
        match result {
            Ok(()) => {}
            Err(err) => super::device::handle_unexpected(err),
        }
    }
}

bitflags::bitflags! {
    /// Flags to augment descriptor set allocation.
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
    pub struct DescriptorSetLayoutCreateFlags: u32 {
        /// Specified that descriptor set must be allocated from\
        /// pool with `DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`.
        ///
        /// This flag must be specified when and only when layout was created with matching backend-specific flag,
        /// that allows layout to have UpdateAfterBind bindings.
        const UPDATE_AFTER_BIND = 0x2;
    }
}

/// Descriptor set from allocator.
#[derive(Debug)]
pub struct DescriptorSet {
    raw: vk::DescriptorSet,
    pool_id: u64,
    size: DescriptorTotalCount,
    update_after_bind: bool,
}

impl DescriptorSet {
    /// Returns reference to raw descriptor set.
    pub fn raw(&self) -> &vk::DescriptorSet {
        &self.raw
    }
}

/// AllocationError that may occur during descriptor sets allocation.
#[derive(Debug)]
pub enum AllocationError {
    /// Backend reported that device memory has been exhausted.\
    /// Deallocating device memory or other resources may increase chance
    /// that another allocation would succeed.
    OutOfDeviceMemory,

    /// Backend reported that host memory has been exhausted.\
    /// Deallocating host memory may increase chance that another allocation would succeed.
    OutOfHostMemory,

    /// The total number of descriptors across all pools created\
    /// with flag `CREATE_UPDATE_AFTER_BIND_BIT` set exceeds `max_update_after_bind_descriptors_in_all_pools`
    /// Or fragmentation of the underlying hardware resources occurs.
    Fragmentation,

    Unexpected,
}

impl Display for AllocationError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationError::OutOfDeviceMemory => fmt.write_str("Device memory exhausted"),
            AllocationError::OutOfHostMemory => fmt.write_str("Host memory exhausted"),
            AllocationError::Fragmentation => fmt.write_str("Fragmentation"),
            AllocationError::Unexpected => fmt.write_str("Unexpected error"),
        }
    }
}

impl core::error::Error for AllocationError {}

impl From<CreatePoolError> for AllocationError {
    fn from(err: CreatePoolError) -> Self {
        match err {
            CreatePoolError::OutOfDeviceMemory => AllocationError::OutOfDeviceMemory,
            CreatePoolError::OutOfHostMemory => AllocationError::OutOfHostMemory,
            CreatePoolError::Fragmentation => AllocationError::Fragmentation,
            CreatePoolError::Unexpected => AllocationError::Unexpected,
        }
    }
}

const MIN_SETS: u32 = 64;
const MAX_SETS: u32 = 512;

#[derive(Debug)]
struct DescriptorPool {
    raw: vk::DescriptorPool,

    /// Number of sets allocated from pool.
    allocated: u32,

    /// Expected number of sets available.
    available: u32,
}

#[derive(Debug)]
struct DescriptorBucket {
    offset: u64,
    pools: VecDeque<DescriptorPool>,
    total: u32,
    update_after_bind: bool,
    size: DescriptorTotalCount,
}

impl Drop for DescriptorBucket {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        if self.total > 0 {
            log::error!("Descriptor sets were not deallocated")
        }
    }
}

impl DescriptorBucket {
    fn new(update_after_bind: bool, size: DescriptorTotalCount) -> Self {
        DescriptorBucket {
            offset: 0,
            pools: VecDeque::new(),
            total: 0,
            update_after_bind,
            size,
        }
    }

    fn new_pool_size(&self, minimal_set_count: u32) -> (DescriptorTotalCount, u32) {
        let mut max_sets = MIN_SETS // at least MIN_SETS
            .max(minimal_set_count) // at least enough for allocation
            .max(self.total.min(MAX_SETS)) // at least as much as was allocated so far capped to MAX_SETS
            .checked_next_power_of_two() // rounded up to nearest 2^N
            .unwrap_or(i32::MAX as u32);

        max_sets = (u32::MAX / self.size.sampler.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.combined_image_sampler.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.sampled_image.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.storage_image.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.uniform_texel_buffer.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.storage_texel_buffer.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.uniform_buffer.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.storage_buffer.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.uniform_buffer_dynamic.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.storage_buffer_dynamic.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.input_attachment.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.acceleration_structure.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.inline_uniform_block_bytes.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.inline_uniform_block_bindings.max(1)).min(max_sets);

        let mut pool_size = DescriptorTotalCount {
            sampler: self.size.sampler * max_sets,
            combined_image_sampler: self.size.combined_image_sampler * max_sets,
            sampled_image: self.size.sampled_image * max_sets,
            storage_image: self.size.storage_image * max_sets,
            uniform_texel_buffer: self.size.uniform_texel_buffer * max_sets,
            storage_texel_buffer: self.size.storage_texel_buffer * max_sets,
            uniform_buffer: self.size.uniform_buffer * max_sets,
            storage_buffer: self.size.storage_buffer * max_sets,
            uniform_buffer_dynamic: self.size.uniform_buffer_dynamic * max_sets,
            storage_buffer_dynamic: self.size.storage_buffer_dynamic * max_sets,
            input_attachment: self.size.input_attachment * max_sets,
            acceleration_structure: self.size.acceleration_structure * max_sets,
            inline_uniform_block_bytes: self.size.inline_uniform_block_bytes * max_sets,
            inline_uniform_block_bindings: self.size.inline_uniform_block_bindings * max_sets,
        };

        if pool_size == Default::default() {
            pool_size.sampler = 1;
        }

        (pool_size, max_sets)
    }

    unsafe fn allocate(
        &mut self,
        device: &super::DeviceShared,
        layout: &vk::DescriptorSetLayout,
        mut count: u32,
        allocated_sets: &mut Vec<DescriptorSet>,
    ) -> Result<(), AllocationError> {
        debug_assert!(usize::try_from(count).is_ok(), "Must be ensured by caller");

        if count == 0 {
            return Ok(());
        }

        for (index, pool) in self.pools.iter_mut().enumerate().rev() {
            if pool.available == 0 {
                continue;
            }

            let allocate = pool.available.min(count);

            log::trace!("Allocate `{}` sets from existing pool", allocate);

            let result = unsafe {
                device.alloc_descriptor_sets(&mut pool.raw, (0..allocate).map(|_| layout))
            };

            match result {
                Ok(vk_sets) => {
                    allocated_sets.extend(vk_sets.into_iter().map(|raw| DescriptorSet {
                        raw,
                        pool_id: index as u64 + self.offset,
                        update_after_bind: self.update_after_bind,
                        size: self.size,
                    }))
                }
                Err(DeviceAllocationError::OutOfDeviceMemory) => {
                    return Err(AllocationError::OutOfDeviceMemory)
                }
                Err(DeviceAllocationError::OutOfHostMemory) => {
                    return Err(AllocationError::OutOfHostMemory)
                }
                Err(DeviceAllocationError::Unexpected) => return Err(AllocationError::Unexpected),
                Err(DeviceAllocationError::FragmentedPool) => {
                    // Should not happen, but better this than panicing.

                    log::error!("Unexpectedly failed to allocated descriptor sets due to pool fragmentation");
                    pool.available = 0;
                    continue;
                }
            }

            count -= allocate;
            pool.available -= allocate;
            pool.allocated += allocate;
            self.total += allocate;

            if count == 0 {
                return Ok(());
            }
        }

        while count > 0 {
            let (pool_size, max_sets) = self.new_pool_size(count);

            log::trace!(
                "Create new pool with {} sets and {:?} descriptors",
                max_sets,
                pool_size,
            );

            let mut raw = device.create_descriptor_pool(
                &pool_size,
                max_sets,
                if self.update_after_bind {
                    DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                        | DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
                } else {
                    DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                },
            )?;

            let pool_id = self.pools.len() as u64 + self.offset;

            let allocate = max_sets.min(count);
            let result =
                unsafe { device.alloc_descriptor_sets(&mut raw, (0..allocate).map(|_| layout)) };

            match result {
                Ok(vk_sets) => {
                    allocated_sets.extend(vk_sets.into_iter().map(|raw| DescriptorSet {
                        raw,
                        pool_id,
                        size: self.size,
                        update_after_bind: self.update_after_bind,
                    }))
                }
                Err(err) => {
                    unsafe { device.raw.destroy_descriptor_pool(raw, None) };
                    match err {
                        DeviceAllocationError::OutOfDeviceMemory => {
                            return Err(AllocationError::OutOfDeviceMemory)
                        }
                        DeviceAllocationError::OutOfHostMemory => {
                            return Err(AllocationError::OutOfHostMemory)
                        }
                        DeviceAllocationError::Unexpected => {
                            return Err(AllocationError::Unexpected)
                        }
                        DeviceAllocationError::FragmentedPool => {
                            // Should not happen, but better this than panicking.

                            log::error!("Unexpectedly failed to allocated descriptor sets due to pool fragmentation");
                        }
                    }
                    panic!("Failed to allocate descriptor sets from fresh pool");
                }
            }

            count -= allocate;
            self.pools.push_back(DescriptorPool {
                raw,
                allocated: allocate,
                available: max_sets - allocate,
            });
            self.total += allocate;
        }

        Ok(())
    }

    unsafe fn free(
        &mut self,
        device: &super::DeviceShared,
        raw_sets: impl ExactSizeIterator<Item = vk::DescriptorSet>,
        pool_id: u64,
    ) {
        let pool = usize::try_from(pool_id - self.offset)
            .ok()
            .and_then(|index| self.pools.get_mut(index))
            .expect("Invalid pool id");

        let count = raw_sets.len() as u32;
        unsafe { device.dealloc_descriptor_sets(&mut pool.raw, raw_sets) };

        pool.available += count;
        pool.allocated -= count;
        self.total -= count;

        log::trace!("Freed {} from descriptor bucket", count);

        while let Some(pool) = self.pools.pop_front() {
            if self.pools.is_empty() || pool.allocated != 0 {
                self.pools.push_front(pool);
                break;
            }

            log::trace!("Destroying old descriptor pool");

            unsafe { device.raw.destroy_descriptor_pool(pool.raw, None) };
            self.offset += 1;
        }
    }

    unsafe fn cleanup(&mut self, device: &super::DeviceShared) {
        while let Some(pool) = self.pools.pop_front() {
            if pool.allocated != 0 {
                self.pools.push_front(pool);
                break;
            }

            log::trace!("Destroying old descriptor pool");

            unsafe { device.raw.destroy_descriptor_pool(pool.raw, None) };
            self.offset += 1;
        }
    }
}

/// Descriptor allocator.
/// Can be used to allocate descriptor sets for any layout.
#[derive(Debug)]
pub struct DescriptorAllocator {
    buckets: HashMap<(DescriptorTotalCount, bool), DescriptorBucket>,
    sets_cache: Vec<DescriptorSet>,
    raw_sets_cache: Vec<vk::DescriptorSet>,
    max_update_after_bind_descriptors_in_all_pools: u32,
    current_update_after_bind_descriptors_in_all_pools: u32,
    total: u32,
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        if self.buckets.values().any(|bucket| bucket.total != 0) {
            log::error!(
                "`DescriptorAllocator` is dropped while some descriptor sets were not deallocated"
            );
        }
    }
}

impl DescriptorAllocator {
    /// Create new allocator instance.
    pub fn new(max_update_after_bind_descriptors_in_all_pools: u32) -> Self {
        DescriptorAllocator {
            buckets: HashMap::default(),
            total: 0,
            sets_cache: Vec::new(),
            raw_sets_cache: Vec::new(),
            max_update_after_bind_descriptors_in_all_pools,
            current_update_after_bind_descriptors_in_all_pools: 0,
        }
    }

    /// Allocate descriptor set with specified layout.
    ///
    /// # Safety
    ///
    /// * Same `device` instance must be passed to all method calls of
    ///   one `DescriptorAllocator` instance.
    /// * `flags` must match flags that were used to create the layout.
    /// * `layout_descriptor_count` must match descriptor numbers in the layout.
    pub unsafe fn allocate(
        &mut self,
        device: &super::DeviceShared,
        layout: &vk::DescriptorSetLayout,
        flags: DescriptorSetLayoutCreateFlags,
        layout_descriptor_count: &DescriptorTotalCount,
        count: u32,
    ) -> Result<Vec<DescriptorSet>, AllocationError> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let descriptor_count = count * layout_descriptor_count.total();

        let update_after_bind = flags.contains(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND);

        if update_after_bind
            && self.max_update_after_bind_descriptors_in_all_pools
                - self.current_update_after_bind_descriptors_in_all_pools
                < descriptor_count
        {
            return Err(AllocationError::Fragmentation);
        }

        log::trace!(
            "Allocating {} sets with layout {:?} @ {:?}",
            count,
            layout,
            layout_descriptor_count
        );

        let bucket = self
            .buckets
            .entry((*layout_descriptor_count, update_after_bind))
            .or_insert_with(|| DescriptorBucket::new(update_after_bind, *layout_descriptor_count));
        match unsafe { bucket.allocate(device, layout, count, &mut self.sets_cache) } {
            Ok(()) => {
                self.total += descriptor_count;
                if update_after_bind {
                    self.current_update_after_bind_descriptors_in_all_pools += descriptor_count;
                }

                Ok(core::mem::take(&mut self.sets_cache))
            }
            Err(err) => {
                debug_assert!(self.raw_sets_cache.is_empty());

                // Free sets allocated so far.
                let mut last = None;

                for set in self.sets_cache.drain(..) {
                    if Some(set.pool_id) != last {
                        if let Some(last_id) = last {
                            // Free contiguous range of sets from one pool in one go.
                            unsafe { bucket.free(device, self.raw_sets_cache.drain(..), last_id) };
                        }
                    }
                    last = Some(set.pool_id);
                    self.raw_sets_cache.push(set.raw);
                }

                if let Some(last_id) = last {
                    unsafe { bucket.free(device, self.raw_sets_cache.drain(..), last_id) };
                }

                Err(err)
            }
        }
    }

    /// Free descriptor sets.
    ///
    /// # Safety
    ///
    /// * Same `device` instance must be passed to all method calls of
    ///   one `DescriptorAllocator` instance.
    /// * None of descriptor sets can be referenced in any pending command buffers.
    /// * All command buffers where at least one of descriptor sets referenced
    ///   move to invalid state.
    pub unsafe fn free<I>(&mut self, device: &super::DeviceShared, sets: I)
    where
        I: IntoIterator<Item = DescriptorSet>,
    {
        debug_assert!(self.raw_sets_cache.is_empty());

        let mut last_key = (EMPTY_COUNT, false);
        let mut last_pool_id = None;

        let mut descriptor_count = 0;

        // Batch freeing of adjacent descriptor sets that belong to the same bucket and pool.
        for set in sets {
            descriptor_count += set.size.total();

            if last_key != (set.size, set.update_after_bind) || last_pool_id != Some(set.pool_id) {
                if let Some(pool_id) = last_pool_id {
                    unsafe {
                        self.free_raw_sets_cache(device, &last_key, pool_id, descriptor_count)
                    };
                    descriptor_count = 0;
                }

                last_key = (set.size, set.update_after_bind);
                last_pool_id = Some(set.pool_id);
            }
            self.raw_sets_cache.push(set.raw);
        }

        if let Some(pool_id) = last_pool_id {
            unsafe { self.free_raw_sets_cache(device, &last_key, pool_id, descriptor_count) };
        }
    }

    /// Frees the cached descriptor sets which must be allocated from the same bucket and pool.
    unsafe fn free_raw_sets_cache(
        &mut self,
        device: &super::DeviceShared,
        bucket_key: &(DescriptorTotalCount, bool),
        pool_id: u64,
        descriptor_count: u32,
    ) {
        let bucket = self
            .buckets
            .get_mut(bucket_key)
            .expect("Set must be allocated from this allocator");

        debug_assert!(u32::try_from(self.raw_sets_cache.len())
            .ok()
            .is_some_and(|count| count <= bucket.total));

        unsafe { bucket.free(device, self.raw_sets_cache.drain(..), pool_id) };

        self.total -= descriptor_count;
        if bucket.update_after_bind {
            self.current_update_after_bind_descriptors_in_all_pools -= descriptor_count;
        }
    }

    /// Perform cleanup to allow resources reuse.
    ///
    /// # Safety
    ///
    /// * Same `device` instance must be passed to all method calls of
    ///   one `DescriptorAllocator` instance.
    pub unsafe fn cleanup(&mut self, device: &super::DeviceShared) {
        for bucket in self.buckets.values_mut() {
            unsafe { bucket.cleanup(device) }
        }
        self.buckets.retain(|_, bucket| !bucket.pools.is_empty());
    }
}

/// Empty descriptor per_type.
const EMPTY_COUNT: DescriptorTotalCount = DescriptorTotalCount {
    sampler: 0,
    combined_image_sampler: 0,
    sampled_image: 0,
    storage_image: 0,
    uniform_texel_buffer: 0,
    storage_texel_buffer: 0,
    uniform_buffer: 0,
    storage_buffer: 0,
    uniform_buffer_dynamic: 0,
    storage_buffer_dynamic: 0,
    input_attachment: 0,
    acceleration_structure: 0,
    inline_uniform_block_bytes: 0,
    inline_uniform_block_bindings: 0,
};
