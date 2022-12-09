use native::WeakPtr;
use parking_lot::Mutex;
use std::ptr;
use winapi::{
    um::{
        d3d12::{self, ID3D12Resource},
        winnt::HRESULT,
    },
    Interface,
};

// This exists to work around https://github.com/gfx-rs/wgpu/issues/3207
// Currently this will work the older, slower way if the windows_rs feature is disabled,
// and will suballocate buffers using gpu_allocator if the windows_rs feature is enabled.

#[cfg(feature = "windows_rs")]
use gpu_allocator::{
    d3d12::{AllocationCreateDesc, ToWinapi, ToWindows},
    MemoryLocation,
};

// TODO: find the exact value
// TODO: figure out if this is even needed?
#[allow(unused)] // TODO: Exists until windows-rs is standard, then it can probably be removed?
pub(crate) const D3D12_HEAP_FLAG_CREATE_NOT_ZEROED: u32 = d3d12::D3D12_HEAP_FLAG_NONE;

#[derive(Debug)]
pub(crate) struct GpuAllocatorWrapper {
    #[cfg(feature = "windows_rs")]
    pub(crate) allocator: gpu_allocator::d3d12::Allocator,
    #[cfg(not(feature = "windows_rs"))]
    #[allow(unused)]
    pub(crate) allocator: (),
}

#[derive(Debug)]
pub(crate) struct AllocationWrapper {
    #[cfg(feature = "windows_rs")]
    pub(crate) allocation: gpu_allocator::d3d12::Allocation,
    #[cfg(not(feature = "windows_rs"))]
    #[allow(unused)]
    pub(crate) allocation: (),
}

#[cfg(feature = "windows_rs")]
pub(crate) fn create_allocator_wrapper(
    raw: &native::Device,
) -> Result<Option<Mutex<GpuAllocatorWrapper>>, crate::DeviceError> {
    use log::error;

    let device = raw.as_ptr();

    match gpu_allocator::d3d12::Allocator::new(&gpu_allocator::d3d12::AllocatorCreateDesc {
        device: device.as_windows().clone(),
        debug_settings: Default::default(),
    }) {
        Ok(allocator) => Ok(Some(Mutex::new(GpuAllocatorWrapper { allocator }))),
        Err(e) => {
            error!("Failed to create d3d12 allocator, error: {}", e);
            Err(e)?
        }
    }
}

#[cfg(not(feature = "windows_rs"))]
pub(crate) fn create_allocator_wrapper(
    _raw: &native::Device,
) -> Result<Option<Mutex<GpuAllocatorWrapper>>, crate::DeviceError> {
    Ok(None)
}

#[cfg(feature = "windows_rs")]
pub(crate) fn create_buffer_resource(
    device: &super::Device,
    desc: &crate::BufferDescriptor,
    raw_desc: d3d12::D3D12_RESOURCE_DESC,
    resource: &mut WeakPtr<ID3D12Resource>,
) -> Result<(HRESULT, Option<AllocationWrapper>), crate::DeviceError> {
    let is_cpu_read = desc.usage.contains(crate::BufferUses::MAP_READ);
    let is_cpu_write = desc.usage.contains(crate::BufferUses::MAP_WRITE);
    // TODO: These are probably wrong?
    let location = match (is_cpu_read, is_cpu_write) {
        (true, true) => MemoryLocation::CpuToGpu,
        (true, false) => MemoryLocation::GpuToCpu,
        (false, true) => MemoryLocation::CpuToGpu,
        (false, false) => MemoryLocation::GpuOnly,
    };

    let name = desc.label.unwrap_or("Unlabeled buffer");

    let mut allocator = device.mem_allocator.as_ref().unwrap().lock();
    let allocation_desc = AllocationCreateDesc::from_winapi_d3d12_resource_desc(
        allocator.allocator.device().as_winapi(),
        &raw_desc,
        name,
        location,
    );
    let allocation = allocator.allocator.allocate(&allocation_desc)?;

    let hr = unsafe {
        device.raw.CreatePlacedResource(
            allocation.heap().as_winapi() as *mut _,
            allocation.offset(),
            &raw_desc,
            d3d12::D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &d3d12::ID3D12Resource::uuidof(),
            resource.mut_void(),
        )
    };

    Ok((hr, Some(AllocationWrapper { allocation })))
}

#[cfg(not(feature = "windows_rs"))]
pub(crate) fn create_buffer_resource(
    device: &super::Device,
    desc: &crate::BufferDescriptor,
    raw_desc: d3d12::D3D12_RESOURCE_DESC,
    resource: &mut WeakPtr<ID3D12Resource>,
) -> Result<(HRESULT, Option<AllocationWrapper>), crate::DeviceError> {
    let is_cpu_read = desc.usage.contains(crate::BufferUses::MAP_READ);
    let is_cpu_write = desc.usage.contains(crate::BufferUses::MAP_WRITE);

    let heap_properties = d3d12::D3D12_HEAP_PROPERTIES {
        Type: d3d12::D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: if is_cpu_read {
            d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
        } else if is_cpu_write {
            d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE
        } else {
            d3d12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE
        },
        MemoryPoolPreference: match device.private_caps.memory_architecture {
            super::MemoryArchitecture::NonUnified if !is_cpu_read && !is_cpu_write => {
                d3d12::D3D12_MEMORY_POOL_L1
            }
            _ => d3d12::D3D12_MEMORY_POOL_L0,
        },
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };

    let hr = unsafe {
        device.raw.CreateCommittedResource(
            &heap_properties,
            if device.private_caps.heap_create_not_zeroed {
                D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
            } else {
                d3d12::D3D12_HEAP_FLAG_NONE
            },
            &raw_desc,
            d3d12::D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &d3d12::ID3D12Resource::uuidof(),
            resource.mut_void(),
        )
    };

    Ok((hr, None))
}

#[cfg(feature = "windows_rs")]
pub(crate) fn create_texture_resource(
    device: &super::Device,
    desc: &crate::TextureDescriptor,
    raw_desc: d3d12::D3D12_RESOURCE_DESC,
    resource: &mut WeakPtr<ID3D12Resource>,
) -> Result<(HRESULT, Option<AllocationWrapper>), crate::DeviceError> {
    let location = MemoryLocation::GpuOnly;

    let name = desc.label.unwrap_or("Unlabeled texture");

    let mut allocator = device.mem_allocator.as_ref().unwrap().lock();
    let allocation_desc = AllocationCreateDesc::from_winapi_d3d12_resource_desc(
        allocator.allocator.device().as_winapi(),
        &raw_desc,
        name,
        location,
    );
    let allocation = allocator.allocator.allocate(&allocation_desc)?;

    let hr = unsafe {
        device.raw.CreatePlacedResource(
            allocation.heap().as_winapi() as *mut _,
            allocation.offset(),
            &raw_desc,
            d3d12::D3D12_RESOURCE_STATE_COMMON,
            ptr::null(), // clear value
            &d3d12::ID3D12Resource::uuidof(),
            resource.mut_void(),
        )
    };

    Ok((hr, Some(AllocationWrapper { allocation })))
}

#[cfg(not(feature = "windows_rs"))]
pub(crate) fn create_texture_resource(
    device: &super::Device,
    _desc: &crate::TextureDescriptor,
    raw_desc: d3d12::D3D12_RESOURCE_DESC,
    resource: &mut WeakPtr<ID3D12Resource>,
) -> Result<(HRESULT, Option<AllocationWrapper>), crate::DeviceError> {
    let heap_properties = d3d12::D3D12_HEAP_PROPERTIES {
        Type: d3d12::D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: d3d12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
        MemoryPoolPreference: match device.private_caps.memory_architecture {
            super::MemoryArchitecture::NonUnified => d3d12::D3D12_MEMORY_POOL_L1,
            super::MemoryArchitecture::Unified { .. } => d3d12::D3D12_MEMORY_POOL_L0,
        },
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    };

    let hr = unsafe {
        device.raw.CreateCommittedResource(
            &heap_properties,
            if device.private_caps.heap_create_not_zeroed {
                D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
            } else {
                d3d12::D3D12_HEAP_FLAG_NONE
            },
            &raw_desc,
            d3d12::D3D12_RESOURCE_STATE_COMMON,
            ptr::null(), // clear value
            &d3d12::ID3D12Resource::uuidof(),
            resource.mut_void(),
        )
    };

    Ok((hr, None))
}

#[cfg(not(feature = "windows_rs"))]
pub(crate) fn free_buffer_allocation(
    _allocation: AllocationWrapper,
    _allocator: &Mutex<GpuAllocatorWrapper>,
) {
    // No-op when not using gpu-allocator
}

#[cfg(feature = "windows_rs")]
pub(crate) fn free_buffer_allocation(
    allocation: AllocationWrapper,
    allocator: &Mutex<GpuAllocatorWrapper>,
) {
    match allocator.lock().allocator.free(allocation.allocation) {
        Ok(_) => (),
        // TODO: Don't panic here
        Err(e) => panic!("Failed to destroy dx12 buffer, {}", e),
    };
}

#[cfg(not(feature = "windows_rs"))]
pub(crate) fn free_texture_allocation(
    _allocation: AllocationWrapper,
    _allocator: &Mutex<GpuAllocatorWrapper>,
) {
    // No-op when not using gpu-allocator
}

#[cfg(feature = "windows_rs")]
pub(crate) fn free_texture_allocation(
    allocation: AllocationWrapper,
    allocator: &Mutex<GpuAllocatorWrapper>,
) {
    match allocator.lock().allocator.free(allocation.allocation) {
        Ok(_) => (),
        // TODO: Don't panic here
        Err(e) => panic!("Failed to destroy dx12 texture, {}", e),
    };
}

#[cfg(feature = "windows_rs")]
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
        }
    }
}
