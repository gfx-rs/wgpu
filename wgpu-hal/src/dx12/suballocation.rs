pub(crate) use allocation::{
    create_allocator_wrapper, create_buffer_resource, create_texture_resource,
    free_buffer_allocation, free_texture_allocation, AllocationWrapper, GpuAllocatorWrapper,
};

#[cfg(not(feature = "windows_rs"))]
use committed as allocation;
#[cfg(feature = "windows_rs")]
use placed as allocation;

// This exists to work around https://github.com/gfx-rs/wgpu/issues/3207
// Currently this will work the older, slower way if the windows_rs feature is disabled,
// and will use the fast path of suballocating buffers and textures using gpu_allocator if
// the windows_rs feature is enabled.

// This is the fast path using gpu_allocator to suballocate buffers and textures.
#[cfg(feature = "windows_rs")]
mod placed {
    use d3d12::ComPtr;
    use parking_lot::Mutex;
    use std::ptr;
    use wgt::assertions::StrictAssertUnwrapExt;
    use winapi::{
        um::{
            d3d12::{self as d3d12_ty, ID3D12Resource},
            winnt::HRESULT,
        },
        Interface,
    };

    use gpu_allocator::{
        d3d12::{AllocationCreateDesc, ToWinapi, ToWindows},
        MemoryLocation,
    };

    #[derive(Debug)]
    pub(crate) struct GpuAllocatorWrapper {
        pub(crate) allocator: gpu_allocator::d3d12::Allocator,
    }

    #[derive(Debug)]
    pub(crate) struct AllocationWrapper {
        pub(crate) allocation: gpu_allocator::d3d12::Allocation,
    }

    pub(crate) fn create_allocator_wrapper(
        raw: &d3d12::Device,
    ) -> Result<Option<Mutex<GpuAllocatorWrapper>>, crate::DeviceError> {
        let device = raw.as_ptr();

        match gpu_allocator::d3d12::Allocator::new(&gpu_allocator::d3d12::AllocatorCreateDesc {
            device: gpu_allocator::d3d12::ID3D12DeviceVersion::Device(device.as_windows().clone()),
            debug_settings: Default::default(),
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        }) {
            Ok(allocator) => Ok(Some(Mutex::new(GpuAllocatorWrapper { allocator }))),
            Err(e) => {
                log::error!("Failed to create d3d12 allocator, error: {}", e);
                Err(e)?
            }
        }
    }

    pub(crate) fn create_buffer_resource(
        device: &crate::dx12::Device,
        desc: &crate::BufferDescriptor,
        raw_desc: d3d12_ty::D3D12_RESOURCE_DESC,
        resource: &mut ComPtr<ID3D12Resource>,
    ) -> Result<(HRESULT, Option<AllocationWrapper>), crate::DeviceError> {
        let is_cpu_read = desc.usage.contains(crate::BufferUses::MAP_READ);
        let is_cpu_write = desc.usage.contains(crate::BufferUses::MAP_WRITE);

        // It's a workaround for Intel Xe drivers.
        if !device.private_caps.suballocation_supported {
            return super::committed::create_buffer_resource(device, desc, raw_desc, resource)
                .map(|(hr, _)| (hr, None));
        }

        let location = match (is_cpu_read, is_cpu_write) {
            (true, true) => MemoryLocation::CpuToGpu,
            (true, false) => MemoryLocation::GpuToCpu,
            (false, true) => MemoryLocation::CpuToGpu,
            (false, false) => MemoryLocation::GpuOnly,
        };

        let name = desc.label.unwrap_or("Unlabeled buffer");

        // SAFETY: allocator exists when the windows_rs feature is enabled
        let mut allocator = unsafe {
            device
                .mem_allocator
                .as_ref()
                .strict_unwrap_unchecked()
                .lock()
        };

        // let mut allocator = unsafe { device.mem_allocator.as_ref().unwrap_unchecked().lock() };
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
                d3d12_ty::D3D12_RESOURCE_STATE_COMMON,
                ptr::null(),
                &d3d12_ty::ID3D12Resource::uuidof(),
                resource.mut_void(),
            )
        };

        Ok((hr, Some(AllocationWrapper { allocation })))
    }

    pub(crate) fn create_texture_resource(
        device: &crate::dx12::Device,
        desc: &crate::TextureDescriptor,
        raw_desc: d3d12_ty::D3D12_RESOURCE_DESC,
        resource: &mut ComPtr<ID3D12Resource>,
    ) -> Result<(HRESULT, Option<AllocationWrapper>), crate::DeviceError> {
        // It's a workaround for Intel Xe drivers.
        if !device.private_caps.suballocation_supported {
            return super::committed::create_texture_resource(device, desc, raw_desc, resource)
                .map(|(hr, _)| (hr, None));
        }

        let location = MemoryLocation::GpuOnly;

        let name = desc.label.unwrap_or("Unlabeled texture");

        // SAFETY: allocator exists when the windows_rs feature is enabled
        let mut allocator = unsafe {
            device
                .mem_allocator
                .as_ref()
                .strict_unwrap_unchecked()
                .lock()
        };
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
                d3d12_ty::D3D12_RESOURCE_STATE_COMMON,
                ptr::null(), // clear value
                &d3d12_ty::ID3D12Resource::uuidof(),
                resource.mut_void(),
            )
        };

        Ok((hr, Some(AllocationWrapper { allocation })))
    }

    pub(crate) fn free_buffer_allocation(
        allocation: AllocationWrapper,
        allocator: &Mutex<GpuAllocatorWrapper>,
    ) {
        match allocator.lock().allocator.free(allocation.allocation) {
            Ok(_) => (),
            // TODO: Don't panic here
            Err(e) => panic!("Failed to destroy dx12 buffer, {e}"),
        };
    }

    pub(crate) fn free_texture_allocation(
        allocation: AllocationWrapper,
        allocator: &Mutex<GpuAllocatorWrapper>,
    ) {
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
                gpu_allocator::AllocationError::BarrierLayoutNeedsDevice10 => todo!(),
            }
        }
    }
}

// This is the older, slower path where it doesn't suballocate buffers.
// Tracking issue for when it can be removed: https://github.com/gfx-rs/wgpu/issues/3207
mod committed {
    use d3d12::ComPtr;
    use parking_lot::Mutex;
    use std::ptr;
    use winapi::{
        um::{
            d3d12::{self as d3d12_ty, ID3D12Resource},
            winnt::HRESULT,
        },
        Interface,
    };

    // https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_flags
    const D3D12_HEAP_FLAG_CREATE_NOT_ZEROED: d3d12_ty::D3D12_HEAP_FLAGS = 0x1000;

    // Allocator isn't needed when not suballocating with gpu_allocator
    #[derive(Debug)]
    pub(crate) struct GpuAllocatorWrapper {}

    // Allocations aren't needed when not suballocating with gpu_allocator
    #[derive(Debug)]
    pub(crate) struct AllocationWrapper {}

    #[allow(unused)]
    pub(crate) fn create_allocator_wrapper(
        _raw: &d3d12::Device,
    ) -> Result<Option<Mutex<GpuAllocatorWrapper>>, crate::DeviceError> {
        Ok(None)
    }

    pub(crate) fn create_buffer_resource(
        device: &crate::dx12::Device,
        desc: &crate::BufferDescriptor,
        raw_desc: d3d12_ty::D3D12_RESOURCE_DESC,
        resource: &mut ComPtr<ID3D12Resource>,
    ) -> Result<(HRESULT, Option<AllocationWrapper>), crate::DeviceError> {
        let is_cpu_read = desc.usage.contains(crate::BufferUses::MAP_READ);
        let is_cpu_write = desc.usage.contains(crate::BufferUses::MAP_WRITE);

        let heap_properties = d3d12_ty::D3D12_HEAP_PROPERTIES {
            Type: d3d12_ty::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: if is_cpu_read {
                d3d12_ty::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
            } else if is_cpu_write {
                d3d12_ty::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE
            } else {
                d3d12_ty::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE
            },
            MemoryPoolPreference: match device.private_caps.memory_architecture {
                crate::dx12::MemoryArchitecture::NonUnified if !is_cpu_read && !is_cpu_write => {
                    d3d12_ty::D3D12_MEMORY_POOL_L1
                }
                _ => d3d12_ty::D3D12_MEMORY_POOL_L0,
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
                    d3d12_ty::D3D12_HEAP_FLAG_NONE
                },
                &raw_desc,
                d3d12_ty::D3D12_RESOURCE_STATE_COMMON,
                ptr::null(),
                &d3d12_ty::ID3D12Resource::uuidof(),
                resource.mut_void(),
            )
        };

        Ok((hr, None))
    }

    pub(crate) fn create_texture_resource(
        device: &crate::dx12::Device,
        _desc: &crate::TextureDescriptor,
        raw_desc: d3d12_ty::D3D12_RESOURCE_DESC,
        resource: &mut ComPtr<ID3D12Resource>,
    ) -> Result<(HRESULT, Option<AllocationWrapper>), crate::DeviceError> {
        let heap_properties = d3d12_ty::D3D12_HEAP_PROPERTIES {
            Type: d3d12_ty::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: d3d12_ty::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: match device.private_caps.memory_architecture {
                crate::dx12::MemoryArchitecture::NonUnified => d3d12_ty::D3D12_MEMORY_POOL_L1,
                crate::dx12::MemoryArchitecture::Unified { .. } => d3d12_ty::D3D12_MEMORY_POOL_L0,
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
                    d3d12_ty::D3D12_HEAP_FLAG_NONE
                },
                &raw_desc,
                d3d12_ty::D3D12_RESOURCE_STATE_COMMON,
                ptr::null(), // clear value
                &d3d12_ty::ID3D12Resource::uuidof(),
                resource.mut_void(),
            )
        };

        Ok((hr, None))
    }

    #[allow(unused)]
    pub(crate) fn free_buffer_allocation(
        _allocation: AllocationWrapper,
        _allocator: &Mutex<GpuAllocatorWrapper>,
    ) {
        // No-op when not using gpu-allocator
    }

    #[allow(unused)]
    pub(crate) fn free_texture_allocation(
        _allocation: AllocationWrapper,
        _allocator: &Mutex<GpuAllocatorWrapper>,
    ) {
        // No-op when not using gpu-allocator
    }
}
