use hal::{self, Device as _Device, QueueGroup};
use {conv, memory, pipeline, resource};

use {BufferHandle, CommandBufferHandle, DeviceHandle, ShaderModuleHandle};


#[repr(C)]
pub struct CommandBufferDescriptor {
}

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

pub extern "C"
fn device_create_buffer(
    device: DeviceHandle, desc: resource::BufferDescriptor
) -> BufferHandle {
    let (usage, memory_properties) = conv::map_buffer_usage(desc.usage);
    let buffer = device.device.create_buffer(desc.size as u64, usage).unwrap();
    BufferHandle::new(resource::Buffer {
        raw: buffer,
        memory_properties,
    })
}

pub struct ShaderModule<B: hal::Backend> {
    pub raw: B::ShaderModule,
}

pub extern "C"
fn device_create_shader_module(
    device: DeviceHandle, desc: pipeline::ShaderModuleDescriptor
) -> ShaderModuleHandle {
    let shader = device.device.create_shader_module(desc.code).unwrap();
    ShaderModuleHandle::new(ShaderModule {
        raw: shader,
    })
}

pub extern "C"
fn device_create_command_buffer(
    device: DeviceHandle, desc: CommandBufferDescriptor
) -> CommandBufferHandle {
    unimplemented!()
}
