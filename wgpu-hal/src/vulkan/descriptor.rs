// Copyright 2020 The gpu-descriptor Project Developers
//
// Copied and modified from https://github.com/zakarumych/gpu-descriptor
// which is licenced under MIT/APACHE

use alloc::{collections::VecDeque, vec::Vec};
use arrayvec::ArrayVec;
use ash::vk;
use core::convert::TryFrom as _;
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

impl super::DeviceShared {
    /// Creates a new descriptor pool.
    fn create_descriptor_pool(
        &self,
        descriptor_count: &DescriptorTotalCount,
        max_sets: u32,
        flags: DescriptorPoolCreateFlags,
    ) -> Result<vk::DescriptorPool, crate::DeviceError> {
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

        unsafe { self.raw.create_descriptor_pool(&vk_info, None) }
            .map_err(super::map_host_device_oom_err)
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
    ) -> Result<Vec<vk::DescriptorSet>, crate::DeviceError> {
        unsafe {
            self.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*pool)
                    .set_layouts(
                        &smallvec::SmallVec::<[vk::DescriptorSetLayout; 32]>::from_iter(
                            layouts.cloned(),
                        ),
                    ),
            )
        }
        .map_err(super::map_host_device_oom_err)
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
    ) -> Result<(), crate::DeviceError> {
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

            let vk_sets = unsafe {
                device.alloc_descriptor_sets(&mut pool.raw, (0..allocate).map(|_| layout))
            }?;
            allocated_sets.extend(vk_sets.into_iter().map(|raw| DescriptorSet {
                raw,
                pool_id: index as u64 + self.offset,
                update_after_bind: self.update_after_bind,
                size: self.size,
            }));

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
                    return Err(err);
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
    ) -> Result<Vec<DescriptorSet>, crate::DeviceError> {
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
            return Err(crate::DeviceError::OutOfMemory);
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

    /// Free a descriptor set.
    ///
    /// # Safety
    ///
    /// * Same `device` instance must be passed to all method calls of
    ///   one `DescriptorAllocator` instance.
    /// * The descriptor set cannot be referenced in any pending command buffers.
    /// * All command buffers where the descriptor set is referenced
    ///   move to invalid state.
    pub unsafe fn free(&mut self, device: &super::DeviceShared, set: DescriptorSet) {
        debug_assert!(self.raw_sets_cache.is_empty());

        self.raw_sets_cache.push(set.raw);

        let bucket = self
            .buckets
            .get_mut(&(set.size, set.update_after_bind))
            .expect("Set must be allocated from this allocator");

        debug_assert!(u32::try_from(self.raw_sets_cache.len())
            .ok()
            .is_some_and(|count| count <= bucket.total));

        unsafe { bucket.free(device, self.raw_sets_cache.drain(..), set.pool_id) };

        self.total -= set.size.total();
        if bucket.update_after_bind {
            self.current_update_after_bind_descriptors_in_all_pools -= set.size.total();
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
