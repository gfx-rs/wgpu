use gpu_allocator::{
    d3d12::{AllocationCreateDesc, Allocator},
    MemoryLocation,
};
use parking_lot::Mutex;
use windows::Win32::Graphics::Direct3D12;

use crate::auxil::dxgi::result::HResult as _;

#[derive(Debug)]
pub(crate) enum AllocationType {
    Buffer,
    Texture,
    AccelerationStructure,
}

#[derive(Debug)]
pub(crate) struct Allocation {
    inner: Option<gpu_allocator::d3d12::Allocation>,
    ty: AllocationType,
}

impl Allocation {
    pub fn none(ty: AllocationType) -> Self {
        Self { inner: None, ty }
    }
}

pub(crate) fn create_allocator(
    raw: &Direct3D12::ID3D12Device,
    memory_hints: &wgt::MemoryHints,
) -> Result<Mutex<Allocator>, crate::DeviceError> {
    // TODO: the allocator's configuration should take hardware capability into
    // account.
    let mb = 1024 * 1024;
    let allocation_sizes = match memory_hints {
        wgt::MemoryHints::Performance => gpu_allocator::AllocationSizes::default(),
        wgt::MemoryHints::MemoryUsage => gpu_allocator::AllocationSizes::new(8 * mb, 4 * mb),
        wgt::MemoryHints::Manual {
            suballocated_device_memory_block_size,
        } => {
            // TODO: Would it be useful to expose the host size in memory hints
            // instead of always using half of the device size?
            let device_size = suballocated_device_memory_block_size.start;
            let host_size = device_size / 2;
            gpu_allocator::AllocationSizes::new(device_size, host_size)
        }
    };

    match Allocator::new(&gpu_allocator::d3d12::AllocatorCreateDesc {
        device: gpu_allocator::d3d12::ID3D12DeviceVersion::Device(raw.clone()),
        debug_settings: Default::default(),
        allocation_sizes,
    }) {
        Ok(allocator) => Ok(Mutex::new(allocator)),
        Err(e) => {
            log::error!("Failed to create d3d12 allocator, error: {}", e);
            Err(e)?
        }
    }
}

/// To allow us to construct buffers from both a `Device` and `CommandEncoder`
/// without needing each function to take a million arguments, we create a
/// borrowed context struct that contains the relevant members.
pub(crate) struct DeviceAllocationContext<'a> {
    pub(crate) raw: &'a Direct3D12::ID3D12Device,
    pub(crate) shared: &'a super::DeviceShared,
    pub(crate) mem_allocator: &'a Mutex<Allocator>,
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

impl<'a> DeviceAllocationContext<'a> {
    pub(crate) fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), crate::DeviceError> {
        let is_cpu_read = desc.usage.contains(wgt::BufferUses::MAP_READ);
        let is_cpu_write = desc.usage.contains(wgt::BufferUses::MAP_WRITE);

        // Workaround for Intel Xe drivers
        if !self.shared.private_caps.suballocation_supported {
            let resource = self.create_committed_buffer(desc, raw_desc)?;
            return Ok((resource, Allocation::none(AllocationType::Buffer)));
        }

        let location = match (is_cpu_read, is_cpu_write) {
            (true, true) => MemoryLocation::CpuToGpu,
            (true, false) => MemoryLocation::GpuToCpu,
            (false, true) => MemoryLocation::CpuToGpu,
            (false, false) => MemoryLocation::GpuOnly,
        };

        let name = desc.label.unwrap_or("Unlabeled buffer");

        let mut allocator = self.mem_allocator.lock();

        let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
            allocator.device(),
            &raw_desc,
            name,
            location,
        );
        let allocation = allocator.allocate(&allocation_desc)?;
        let mut resource = None;

        unsafe {
            self.raw.CreatePlacedResource(
                allocation.heap(),
                allocation.offset(),
                &raw_desc,
                Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
        }
        .into_device_result("Placed buffer creation")?;

        let resource = resource.ok_or(crate::DeviceError::Unexpected)?;

        self.counters.buffer_memory.add(allocation.size() as isize);

        Ok((
            resource,
            Allocation {
                inner: Some(allocation),
                ty: AllocationType::Buffer,
            },
        ))
    }

    pub(crate) fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), crate::DeviceError> {
        // Workaround for Intel Xe drivers
        if !self.shared.private_caps.suballocation_supported {
            let resource = self.create_committed_texture(desc, raw_desc)?;
            return Ok((resource, Allocation::none(AllocationType::Texture)));
        }

        let location = MemoryLocation::GpuOnly;

        let name = desc.label.unwrap_or("Unlabeled texture");

        let mut allocator = self.mem_allocator.lock();
        let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
            allocator.device(),
            &raw_desc,
            name,
            location,
        );
        let allocation = allocator.allocate(&allocation_desc)?;
        let mut resource = None;

        unsafe {
            self.raw.CreatePlacedResource(
                allocation.heap(),
                allocation.offset(),
                &raw_desc,
                Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                None, // clear value
                &mut resource,
            )
        }
        .into_device_result("Placed texture creation")?;

        let resource = resource.ok_or(crate::DeviceError::Unexpected)?;

        self.counters.texture_memory.add(allocation.size() as isize);

        Ok((
            resource,
            Allocation {
                inner: Some(allocation),
                ty: AllocationType::Texture,
            },
        ))
    }

    pub(crate) fn create_acceleration_structure(
        &self,
        desc: &crate::AccelerationStructureDescriptor,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<(Direct3D12::ID3D12Resource, Allocation), crate::DeviceError> {
        // Workaround for Intel Xe drivers
        if !self.shared.private_caps.suballocation_supported {
            let resource = self.create_committed_acceleration_structure(desc, raw_desc)?;
            return Ok((
                resource,
                Allocation::none(AllocationType::AccelerationStructure),
            ));
        }

        let location = MemoryLocation::GpuOnly;

        let name = desc.label.unwrap_or("Unlabeled acceleration structure");

        let mut allocator = self.mem_allocator.lock();

        let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
            allocator.device(),
            &raw_desc,
            name,
            location,
        );
        let allocation = allocator.allocate(&allocation_desc)?;
        let mut resource = None;

        unsafe {
            self.raw.CreatePlacedResource(
                allocation.heap(),
                allocation.offset(),
                &raw_desc,
                Direct3D12::D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                None,
                &mut resource,
            )
        }
        .into_device_result("Placed acceleration structure creation")?;

        let resource = resource.ok_or(crate::DeviceError::Unexpected)?;

        self.counters
            .acceleration_structure_memory
            .add(allocation.size() as isize);

        Ok((
            resource,
            Allocation {
                inner: Some(allocation),
                ty: AllocationType::AccelerationStructure,
            },
        ))
    }

    pub(crate) fn free_resource(
        &self,
        resource: Direct3D12::ID3D12Resource,
        allocation: Allocation,
    ) {
        // Make sure the resource is released before we free the allocation.
        drop(resource);

        let Some(inner) = allocation.inner else {
            return;
        };

        let counter = match allocation.ty {
            AllocationType::Buffer => &self.counters.buffer_memory,
            AllocationType::Texture => &self.counters.texture_memory,
            AllocationType::AccelerationStructure => &self.counters.acceleration_structure_memory,
        };
        counter.sub(inner.size() as isize);

        match self.mem_allocator.lock().free(inner) {
            Ok(_) => (),
            // TODO: Don't panic here
            Err(e) => panic!("Failed to destroy dx12 {:?}, {e}", allocation.ty),
        };
    }

    pub(crate) fn create_committed_buffer(
        &self,
        desc: &crate::BufferDescriptor,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<Direct3D12::ID3D12Resource, crate::DeviceError> {
        let is_cpu_read = desc.usage.contains(wgt::BufferUses::MAP_READ);
        let is_cpu_write = desc.usage.contains(wgt::BufferUses::MAP_WRITE);

        let heap_properties = Direct3D12::D3D12_HEAP_PROPERTIES {
            Type: Direct3D12::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: if is_cpu_read {
                Direct3D12::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
            } else if is_cpu_write {
                Direct3D12::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE
            } else {
                Direct3D12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE
            },
            MemoryPoolPreference: match self.shared.private_caps.memory_architecture {
                crate::dx12::MemoryArchitecture::NonUnified if !is_cpu_read && !is_cpu_write => {
                    Direct3D12::D3D12_MEMORY_POOL_L1
                }
                _ => Direct3D12::D3D12_MEMORY_POOL_L0,
            },
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let mut resource = None;

        unsafe {
            self.raw.CreateCommittedResource(
                &heap_properties,
                if self.shared.private_caps.heap_create_not_zeroed {
                    Direct3D12::D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
                } else {
                    Direct3D12::D3D12_HEAP_FLAG_NONE
                },
                &raw_desc,
                Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
        }
        .into_device_result("Committed buffer creation")?;

        resource.ok_or(crate::DeviceError::Unexpected)
    }

    pub(crate) fn create_committed_texture(
        &self,
        _desc: &crate::TextureDescriptor,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<Direct3D12::ID3D12Resource, crate::DeviceError> {
        let heap_properties = Direct3D12::D3D12_HEAP_PROPERTIES {
            Type: Direct3D12::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: Direct3D12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: match self.shared.private_caps.memory_architecture {
                crate::dx12::MemoryArchitecture::NonUnified => Direct3D12::D3D12_MEMORY_POOL_L1,
                crate::dx12::MemoryArchitecture::Unified { .. } => Direct3D12::D3D12_MEMORY_POOL_L0,
            },
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let mut resource = None;

        unsafe {
            self.raw.CreateCommittedResource(
                &heap_properties,
                if self.shared.private_caps.heap_create_not_zeroed {
                    Direct3D12::D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
                } else {
                    Direct3D12::D3D12_HEAP_FLAG_NONE
                },
                &raw_desc,
                Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                None, // clear value
                &mut resource,
            )
        }
        .into_device_result("Committed texture creation")?;

        resource.ok_or(crate::DeviceError::Unexpected)
    }

    pub(crate) fn create_committed_acceleration_structure(
        &self,
        _desc: &crate::AccelerationStructureDescriptor,
        raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
    ) -> Result<Direct3D12::ID3D12Resource, crate::DeviceError> {
        let heap_properties = Direct3D12::D3D12_HEAP_PROPERTIES {
            Type: Direct3D12::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: Direct3D12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: match self.shared.private_caps.memory_architecture {
                crate::dx12::MemoryArchitecture::NonUnified => Direct3D12::D3D12_MEMORY_POOL_L1,
                _ => Direct3D12::D3D12_MEMORY_POOL_L0,
            },
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let mut resource = None;

        unsafe {
            self.raw.CreateCommittedResource(
                &heap_properties,
                if self.shared.private_caps.heap_create_not_zeroed {
                    Direct3D12::D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
                } else {
                    Direct3D12::D3D12_HEAP_FLAG_NONE
                },
                &raw_desc,
                Direct3D12::D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                None,
                &mut resource,
            )
        }
        .into_device_result("Committed acceleration structure creation")?;

        resource.ok_or(crate::DeviceError::Unexpected)
    }
}

impl From<gpu_allocator::AllocationError> for crate::DeviceError {
    fn from(result: gpu_allocator::AllocationError) -> Self {
        match result {
            gpu_allocator::AllocationError::OutOfMemory => Self::OutOfMemory,
            gpu_allocator::AllocationError::FailedToMap(e) => {
                log::error!("DX12 gpu-allocator: Failed to map: {}", e);
                Self::Lost
            }
            gpu_allocator::AllocationError::NoCompatibleMemoryTypeFound => {
                log::error!("DX12 gpu-allocator: No Compatible Memory Type Found");
                Self::Lost
            }
            gpu_allocator::AllocationError::InvalidAllocationCreateDesc => {
                log::error!("DX12 gpu-allocator: Invalid Allocation Creation Description");
                Self::Lost
            }
            gpu_allocator::AllocationError::InvalidAllocatorCreateDesc(e) => {
                log::error!(
                    "DX12 gpu-allocator: Invalid Allocator Creation Description: {}",
                    e
                );
                Self::Lost
            }

            gpu_allocator::AllocationError::Internal(e) => {
                log::error!("DX12 gpu-allocator: Internal Error: {}", e);
                Self::Lost
            }
            gpu_allocator::AllocationError::BarrierLayoutNeedsDevice10
            | gpu_allocator::AllocationError::CastableFormatsRequiresEnhancedBarriers
            | gpu_allocator::AllocationError::CastableFormatsRequiresAtLeastDevice12 => {
                unreachable!()
            }
        }
    }
}
