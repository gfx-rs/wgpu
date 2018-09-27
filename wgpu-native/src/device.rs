use hal::{self, Device as _Device, QueueGroup};
use {command, conv, memory, pipeline, resource};

use registry::{self, Registry};
use {BufferId, CommandBufferId, DeviceId, ShaderModuleId};

use std::slice;


pub struct Device<B: hal::Backend> {
    device: B::Device,
    queue_group: QueueGroup<B, hal::General>,
    mem_allocator: memory::SmartAllocator<B>,
    com_allocator: command::CommandAllocator<B>,
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn new(
        device: B::Device,
        queue_group: QueueGroup<B, hal::General>,
        mem_props: hal::MemoryProperties,
    ) -> Self {
        Device {
            device,
            mem_allocator: memory::SmartAllocator::new(mem_props, 1, 1, 1, 1),
            com_allocator: command::CommandAllocator::new(queue_group.family()),
            queue_group,
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
    let device = registry::DEVICE_REGISTRY.get_mut(device_id);
    let shader = device
        .device
        .create_shader_module(unsafe {
            slice::from_raw_parts(desc.code.bytes, desc.code.length)
        }).unwrap();
    registry::SHADER_MODULE_REGISTRY.register(ShaderModule { raw: shader })
}

#[no_mangle]
pub extern "C" fn wgpu_device_create_command_buffer(
    device_id: DeviceId,
    desc: command::CommandBufferDescriptor,
) -> CommandBufferId {
    let device = registry::DEVICE_REGISTRY.get_mut(device_id);
    let cmd_buf = device.com_allocator.allocate(&device.device);
    registry::COMMAND_BUFFER_REGISTRY.register(cmd_buf)
}
