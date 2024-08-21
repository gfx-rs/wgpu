use gpu_allocator::{d3d12::AllocationCreateDesc, MemoryLocation};
use parking_lot::Mutex;
use windows::Win32::Graphics::Direct3D12;

use crate::auxil::dxgi::result::HResult as _;

#[derive(Debug)]
pub(crate) struct GpuAllocatorWrapper {
    pub(crate) allocator: gpu_allocator::d3d12::Allocator,
}

#[derive(Debug)]
pub(crate) struct AllocationWrapper {
    pub(crate) allocation: gpu_allocator::d3d12::Allocation,
}

pub(crate) fn create_allocator_wrapper(
    raw: &Direct3D12::ID3D12Device,
    memory_hints: &wgt::MemoryHints,
) -> Result<Mutex<GpuAllocatorWrapper>, crate::DeviceError> {
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

    match gpu_allocator::d3d12::Allocator::new(&gpu_allocator::d3d12::AllocatorCreateDesc {
        device: gpu_allocator::d3d12::ID3D12DeviceVersion::Device(raw.clone()),
        debug_settings: Default::default(),
        allocation_sizes,
    }) {
        Ok(allocator) => Ok(Mutex::new(GpuAllocatorWrapper { allocator })),
        Err(e) => {
            log::error!("Failed to create d3d12 allocator, error: {}", e);
            Err(e)?
        }
    }
}

pub(crate) fn create_buffer_resource(
    device: &crate::dx12::Device,
    desc: &crate::BufferDescriptor,
    raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
) -> Result<(Direct3D12::ID3D12Resource, Option<AllocationWrapper>), crate::DeviceError> {
    let is_cpu_read = desc.usage.contains(crate::BufferUses::MAP_READ);
    let is_cpu_write = desc.usage.contains(crate::BufferUses::MAP_WRITE);

    // Workaround for Intel Xe drivers
    if !device.private_caps.suballocation_supported {
        return create_committed_buffer_resource(device, desc, raw_desc)
            .map(|resource| (resource, None));
    }

    let location = match (is_cpu_read, is_cpu_write) {
        (true, true) => MemoryLocation::CpuToGpu,
        (true, false) => MemoryLocation::GpuToCpu,
        (false, true) => MemoryLocation::CpuToGpu,
        (false, false) => MemoryLocation::GpuOnly,
    };

    let name = desc.label.unwrap_or("Unlabeled buffer");

    let mut allocator = device.mem_allocator.lock();

    let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
        allocator.allocator.device(),
        &raw_desc,
        name,
        location,
    );
    let allocation = allocator.allocator.allocate(&allocation_desc)?;
    let mut resource = None;

    unsafe {
        device.raw.CreatePlacedResource(
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

    device
        .counters
        .buffer_memory
        .add(allocation.size() as isize);

    Ok((resource, Some(AllocationWrapper { allocation })))
}

pub(crate) fn create_texture_resource(
    device: &crate::dx12::Device,
    desc: &crate::TextureDescriptor,
    raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
) -> Result<(Direct3D12::ID3D12Resource, Option<AllocationWrapper>), crate::DeviceError> {
    // Workaround for Intel Xe drivers
    if !device.private_caps.suballocation_supported {
        return create_committed_texture_resource(device, desc, raw_desc)
            .map(|resource| (resource, None));
    }

    let location = MemoryLocation::GpuOnly;

    let name = desc.label.unwrap_or("Unlabeled texture");

    let mut allocator = device.mem_allocator.lock();
    let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
        allocator.allocator.device(),
        &raw_desc,
        name,
        location,
    );
    let allocation = allocator.allocator.allocate(&allocation_desc)?;
    let mut resource = None;

    unsafe {
        device.raw.CreatePlacedResource(
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

    device
        .counters
        .texture_memory
        .add(allocation.size() as isize);

    Ok((resource, Some(AllocationWrapper { allocation })))
}

pub(crate) fn free_buffer_allocation(
    device: &crate::dx12::Device,
    allocation: AllocationWrapper,
    allocator: &Mutex<GpuAllocatorWrapper>,
) {
    device
        .counters
        .buffer_memory
        .sub(allocation.allocation.size() as isize);
    match allocator.lock().allocator.free(allocation.allocation) {
        Ok(_) => (),
        // TODO: Don't panic here
        Err(e) => panic!("Failed to destroy dx12 buffer, {e}"),
    };
}

pub(crate) fn free_texture_allocation(
    device: &crate::dx12::Device,
    allocation: AllocationWrapper,
    allocator: &Mutex<GpuAllocatorWrapper>,
) {
    device
        .counters
        .texture_memory
        .sub(allocation.allocation.size() as isize);
    match allocator.lock().allocator.free(allocation.allocation) {
        Ok(_) => (),
        // TODO: Don't panic here
        Err(e) => panic!("Failed to destroy dx12 texture, {e}"),
    };
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

pub(crate) fn create_committed_buffer_resource(
    device: &crate::dx12::Device,
    desc: &crate::BufferDescriptor,
    raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
) -> Result<Direct3D12::ID3D12Resource, crate::DeviceError> {
    let is_cpu_read = desc.usage.contains(crate::BufferUses::MAP_READ);
    let is_cpu_write = desc.usage.contains(crate::BufferUses::MAP_WRITE);

    let heap_properties = Direct3D12::D3D12_HEAP_PROPERTIES {
        Type: Direct3D12::D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: if is_cpu_read {
            Direct3D12::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
        } else if is_cpu_write {
            Direct3D12::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE
        } else {
            Direct3D12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE
        },
        MemoryPoolPreference: match device.private_caps.memory_architecture {
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
        device.raw.CreateCommittedResource(
            &heap_properties,
            if device.private_caps.heap_create_not_zeroed {
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

pub(crate) fn create_committed_texture_resource(
    device: &crate::dx12::Device,
    _desc: &crate::TextureDescriptor,
    raw_desc: Direct3D12::D3D12_RESOURCE_DESC,
) -> Result<Direct3D12::ID3D12Resource, crate::DeviceError> {
    let heap_properties = Direct3D12::D3D12_HEAP_PROPERTIES {
        Type: Direct3D12::D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: Direct3D12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
        MemoryPoolPreference: match device.private_caps.memory_architecture {
            crate::dx12::MemoryArchitecture::NonUnified => Direct3D12::D3D12_MEMORY_POOL_L1,
            crate::dx12::MemoryArchitecture::Unified { .. } => Direct3D12::D3D12_MEMORY_POOL_L0,
        },
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };

    let mut resource = None;

    unsafe {
        device.raw.CreateCommittedResource(
            &heap_properties,
            if device.private_caps.heap_create_not_zeroed {
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
