//! Device memory sub-allocation.
//!
//! gpu-allocator frees a block only after every allocation in it is freed,
//! so short-lived resources get their own pool.

use alloc::vec::Vec;
use core::ops::Deref;

use ash::vk;
use wgpu_sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub(super) enum MemoryPool {
    General,
    Transient,
}

impl MemoryPool {
    pub(super) fn from_memory_flags(flags: crate::MemoryFlags) -> Self {
        if flags.contains(crate::MemoryFlags::TRANSIENT) {
            Self::Transient
        } else {
            Self::General
        }
    }
}

/// A sub-allocation that records which pool to free it to.
#[derive(Debug)]
pub struct MemoryAllocation {
    pool: MemoryPool,
    inner: gpu_allocator::vulkan::Allocation,
}

impl Deref for MemoryAllocation {
    type Target = gpu_allocator::vulkan::Allocation;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub(super) struct MemoryAllocators {
    general: Mutex<gpu_allocator::vulkan::Allocator>,
    transient: Mutex<gpu_allocator::vulkan::Allocator>,
}

impl MemoryAllocators {
    pub(super) fn new(
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        buffer_device_address: bool,
        allocation_sizes: crate::AllocationSizes,
    ) -> Result<Self, gpu_allocator::AllocationError> {
        let mut desc = gpu_allocator::vulkan::AllocatorCreateDesc {
            instance,
            device,
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address,
            allocation_sizes: allocation_sizes.into(),
        };
        let general = gpu_allocator::vulkan::Allocator::new(&desc)?;
        desc.allocation_sizes = allocation_sizes.transient().into();
        let transient = gpu_allocator::vulkan::Allocator::new(&desc)?;
        Ok(Self {
            general: Mutex::new(general),
            transient: Mutex::new(transient),
        })
    }

    fn allocator(&self, pool: MemoryPool) -> &Mutex<gpu_allocator::vulkan::Allocator> {
        match pool {
            MemoryPool::General => &self.general,
            MemoryPool::Transient => &self.transient,
        }
    }

    pub(super) fn allocate(
        &self,
        pool: MemoryPool,
        desc: &gpu_allocator::vulkan::AllocationCreateDesc<'_>,
    ) -> Result<MemoryAllocation, gpu_allocator::AllocationError> {
        let inner = self.allocator(pool).lock().allocate(desc)?;
        Ok(MemoryAllocation { pool, inner })
    }

    pub(super) fn free(
        &self,
        allocation: MemoryAllocation,
    ) -> Result<(), gpu_allocator::AllocationError> {
        self.allocator(allocation.pool)
            .lock()
            .free(allocation.inner)
    }

    pub(super) fn generate_report(&self) -> wgt::AllocatorReport {
        let mut allocations = Vec::new();
        let mut blocks = Vec::new();
        let mut total_allocated_bytes = 0;
        let mut total_reserved_bytes = 0;

        for allocator in [&self.general, &self.transient] {
            let report = allocator.lock().generate_report();

            // Each block's range indexes into its own report's allocation list.
            let allocation_base = allocations.len();
            allocations.extend(
                report
                    .allocations
                    .into_iter()
                    .map(|alloc| wgt::AllocationReport {
                        name: alloc.name,
                        offset: alloc.offset,
                        size: alloc.size,
                    }),
            );
            blocks.extend(
                report
                    .blocks
                    .into_iter()
                    .map(|block| wgt::MemoryBlockReport {
                        size: block.size,
                        allocations: (block.allocations.start + allocation_base)
                            ..(block.allocations.end + allocation_base),
                    }),
            );
            total_allocated_bytes += report.total_allocated_bytes;
            total_reserved_bytes += report.total_capacity_bytes;
        }

        wgt::AllocatorReport {
            allocations,
            blocks,
            total_allocated_bytes,
            total_reserved_bytes,
        }
    }
}
