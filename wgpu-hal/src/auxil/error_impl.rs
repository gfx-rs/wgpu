impl From<gpu_allocator::AllocationError> for crate::DeviceError {
    fn from(result: gpu_allocator::AllocationError) -> Self {
        match result {
            gpu_allocator::AllocationError::OutOfMemory => Self::OutOfMemory,
            gpu_allocator::AllocationError::FailedToMap(e) => {
                log::error!("gpu-allocator: Failed to map: {}", e);
                Self::Lost
            }
            gpu_allocator::AllocationError::NoCompatibleMemoryTypeFound => {
                log::error!("gpu-allocator: No Compatible Memory Type Found");
                Self::Lost
            }
            gpu_allocator::AllocationError::InvalidAllocationCreateDesc => {
                log::error!("gpu-allocator: Invalid Allocation Creation Description");
                Self::Lost
            }
            gpu_allocator::AllocationError::InvalidAllocatorCreateDesc(e) => {
                log::error!(
                    "gpu-allocator: Invalid Allocator Creation Description: {}",
                    e
                );
                Self::Lost
            }

            gpu_allocator::AllocationError::Internal(e) => {
                log::error!("gpu-allocator: Internal Error: {}", e);
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
