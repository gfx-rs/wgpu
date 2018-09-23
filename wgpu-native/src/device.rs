use hal::{self, Device as _Device, QueueGroup};
use {conv, memory, pipeline, resource};

use registry;
use {BufferId, CommandBufferId, DeviceId, ShaderModuleId};

#[repr(C)]
pub struct CommandBufferDescriptor {}

pub struct Device<B: hal::Backend> {
    device: B::Device,
    queue_group: QueueGroup<B, hal::General>,
    allocator: memory::SmartAllocator<B>,
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn new(
        device: B::Device,
        queue_group: QueueGroup<B, hal::General>,
        mem_props: hal::MemoryProperties,
    ) -> Self {
        Device {
            device,
            queue_group,
            allocator: memory::SmartAllocator::new(mem_props, 1, 1, 1, 1),
        }
    }
}

pub struct ShaderModule<B: hal::Backend> {
    pub raw: B::ShaderModule,
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_shader_module(
    device_id: DeviceId,
    desc: pipeline::ShaderModuleDescriptor,
) -> ShaderModuleId {
    let device_registry = registry::DEVICE_REGISTRY.lock().unwrap();
    let device = device_registry.get(device_id).unwrap();
    let shader = device.device.create_shader_module(desc.code).unwrap();
    registry::SHADER_MODULE_REGISTRY
        .lock()
        .unwrap()
        .register(ShaderModule { raw: shader })
}
