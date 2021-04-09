/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::DeviceError;
use hal::device::Device as _;
use std::{borrow::Cow, iter, ptr::NonNull};

#[derive(Debug)]
pub struct MemoryAllocator<B: hal::Backend>(gpu_alloc::GpuAllocator<B::Memory>);
#[derive(Debug)]
pub struct MemoryBlock<B: hal::Backend>(gpu_alloc::MemoryBlock<B::Memory>);
struct MemoryDevice<'a, B: hal::Backend>(&'a B::Device);

impl<B: hal::Backend> MemoryAllocator<B> {
    pub fn new(mem_props: hal::adapter::MemoryProperties, limits: hal::Limits) -> Self {
        let mem_config = gpu_alloc::Config {
            dedicated_threshold: 32 << 20,
            preferred_dedicated_threshold: 8 << 20,
            transient_dedicated_threshold: 128 << 20,
            linear_chunk: 128 << 20,
            minimal_buddy_size: 1 << 10,
            initial_buddy_dedicated_size: 8 << 20,
        };
        let properties = gpu_alloc::DeviceProperties {
            memory_types: Cow::Owned(
                mem_props
                    .memory_types
                    .iter()
                    .map(|mt| gpu_alloc::MemoryType {
                        heap: mt.heap_index as u32,
                        props: gpu_alloc::MemoryPropertyFlags::from_bits_truncate(
                            mt.properties.bits() as u8,
                        ),
                    })
                    .collect::<Vec<_>>(),
            ),
            memory_heaps: Cow::Owned(
                mem_props
                    .memory_heaps
                    .iter()
                    .map(|mh| gpu_alloc::MemoryHeap { size: mh.size })
                    .collect::<Vec<_>>(),
            ),
            max_memory_allocation_count: if limits.max_memory_allocation_count == 0 {
                log::warn!("max_memory_allocation_count is not set by gfx-rs backend");
                !0
            } else {
                limits.max_memory_allocation_count.min(!0u32 as usize) as u32
            },
            max_memory_allocation_size: !0,
            non_coherent_atom_size: limits.non_coherent_atom_size as u64,
            buffer_device_address: false,
        };
        MemoryAllocator(gpu_alloc::GpuAllocator::new(mem_config, properties))
    }

    pub fn allocate(
        &mut self,
        device: &B::Device,
        requirements: hal::memory::Requirements,
        usage: gpu_alloc::UsageFlags,
    ) -> Result<MemoryBlock<B>, DeviceError> {
        assert!(requirements.alignment.is_power_of_two());
        let request = gpu_alloc::Request {
            size: requirements.size,
            align_mask: requirements.alignment - 1,
            memory_types: requirements.type_mask,
            usage,
        };

        unsafe { self.0.alloc(&MemoryDevice::<B>(device), request) }
            .map(MemoryBlock)
            .map_err(|err| match err {
                gpu_alloc::AllocationError::OutOfHostMemory
                | gpu_alloc::AllocationError::OutOfDeviceMemory => DeviceError::OutOfMemory,
                _ => panic!("Unable to allocate memory: {:?}", err),
            })
    }

    pub fn free(&mut self, device: &B::Device, block: MemoryBlock<B>) {
        unsafe { self.0.dealloc(&MemoryDevice::<B>(device), block.0) }
    }

    pub fn clear(&mut self, device: &B::Device) {
        unsafe { self.0.cleanup(&MemoryDevice::<B>(device)) }
    }
}

impl<B: hal::Backend> MemoryBlock<B> {
    pub fn bind_buffer(
        &self,
        device: &B::Device,
        buffer: &mut B::Buffer,
    ) -> Result<(), DeviceError> {
        let mem = self.0.memory();
        unsafe {
            device
                .bind_buffer_memory(mem, self.0.offset(), buffer)
                .map_err(DeviceError::from_bind)
        }
    }

    pub fn bind_image(&self, device: &B::Device, image: &mut B::Image) -> Result<(), DeviceError> {
        let mem = self.0.memory();
        unsafe {
            device
                .bind_image_memory(mem, self.0.offset(), image)
                .map_err(DeviceError::from_bind)
        }
    }

    pub fn is_coherent(&self) -> bool {
        self.0
            .props()
            .contains(gpu_alloc::MemoryPropertyFlags::HOST_COHERENT)
    }

    pub fn map(
        &mut self,
        device: &B::Device,
        inner_offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    ) -> Result<NonNull<u8>, DeviceError> {
        let offset = inner_offset;
        unsafe {
            self.0
                .map(&MemoryDevice::<B>(device), offset, size as usize)
                .map_err(DeviceError::from)
        }
    }

    pub fn unmap(&mut self, device: &B::Device) {
        unsafe { self.0.unmap(&MemoryDevice::<B>(device)) };
    }

    pub fn write_bytes(
        &mut self,
        device: &B::Device,
        inner_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        let offset = inner_offset;
        unsafe {
            self.0
                .write_bytes(&MemoryDevice::<B>(device), offset, data)
                .map_err(DeviceError::from)
        }
    }

    pub fn read_bytes(
        &mut self,
        device: &B::Device,
        inner_offset: wgt::BufferAddress,
        data: &mut [u8],
    ) -> Result<(), DeviceError> {
        let offset = inner_offset;
        unsafe {
            self.0
                .read_bytes(&MemoryDevice::<B>(device), offset, data)
                .map_err(DeviceError::from)
        }
    }

    fn segment(
        &self,
        inner_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    ) -> hal::memory::Segment {
        hal::memory::Segment {
            offset: self.0.offset() + inner_offset,
            size: size.or_else(|| Some(self.0.size())),
        }
    }

    pub fn flush_range(
        &self,
        device: &B::Device,
        inner_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    ) -> Result<(), DeviceError> {
        let segment = self.segment(inner_offset, size);
        let mem = self.0.memory();
        unsafe {
            device
                .flush_mapped_memory_ranges(iter::once((mem, segment)))
                .or(Err(DeviceError::OutOfMemory))
        }
    }

    pub fn invalidate_range(
        &self,
        device: &B::Device,
        inner_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    ) -> Result<(), DeviceError> {
        let segment = self.segment(inner_offset, size);
        let mem = self.0.memory();
        unsafe {
            device
                .invalidate_mapped_memory_ranges(iter::once((mem, segment)))
                .or(Err(DeviceError::OutOfMemory))
        }
    }
}

impl<B: hal::Backend> gpu_alloc::MemoryDevice<B::Memory> for MemoryDevice<'_, B> {
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: gpu_alloc::AllocationFlags,
    ) -> Result<B::Memory, gpu_alloc::OutOfMemory> {
        profiling::scope!("Allocate Memory");

        assert!(flags.is_empty());

        self.0
            .allocate_memory(hal::MemoryTypeId(memory_type as _), size)
            .map_err(|_| gpu_alloc::OutOfMemory::OutOfDeviceMemory)
    }

    unsafe fn deallocate_memory(&self, memory: B::Memory) {
        profiling::scope!("Deallocate Memory");
        self.0.free_memory(memory);
    }

    unsafe fn map_memory(
        &self,
        memory: &mut B::Memory,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, gpu_alloc::DeviceMapError> {
        profiling::scope!("Map memory");
        match self.0.map_memory(
            memory,
            hal::memory::Segment {
                offset,
                size: Some(size),
            },
        ) {
            Ok(ptr) => Ok(NonNull::new(ptr).expect("Pointer to memory mapping must not be null")),
            Err(hal::device::MapError::OutOfMemory(_)) => {
                Err(gpu_alloc::DeviceMapError::OutOfDeviceMemory)
            }
            Err(hal::device::MapError::MappingFailed) => Err(gpu_alloc::DeviceMapError::MapFailed),
            Err(other) => panic!("Unexpected map error: {:?}", other),
        }
    }

    unsafe fn unmap_memory(&self, memory: &mut B::Memory) {
        profiling::scope!("Unmap memory");
        self.0.unmap_memory(memory);
    }

    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[gpu_alloc::MappedMemoryRange<'_, B::Memory>],
    ) -> Result<(), gpu_alloc::OutOfMemory> {
        profiling::scope!("Invalidate memory ranges");
        self.0
            .invalidate_mapped_memory_ranges(ranges.iter().map(|r| {
                (
                    r.memory,
                    hal::memory::Segment {
                        offset: r.offset,
                        size: Some(r.size),
                    },
                )
            }))
            .map_err(|_| gpu_alloc::OutOfMemory::OutOfHostMemory)
    }

    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[gpu_alloc::MappedMemoryRange<'_, B::Memory>],
    ) -> Result<(), gpu_alloc::OutOfMemory> {
        profiling::scope!("Flush memory ranges");
        self.0
            .flush_mapped_memory_ranges(ranges.iter().map(|r| {
                (
                    r.memory,
                    hal::memory::Segment {
                        offset: r.offset,
                        size: Some(r.size),
                    },
                )
            }))
            .map_err(|_| gpu_alloc::OutOfMemory::OutOfHostMemory)
    }
}
