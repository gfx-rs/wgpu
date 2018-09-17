use hal::{self, Device as _Device, QueueGroup};
use {memory, pipeline};

use {BufferHandle, CommandBufferHandle, DeviceHandle, ShaderModuleHandle};


pub type BufferUsage = hal::buffer::Usage;

#[repr(C)]
pub struct BufferDescriptor {
    pub size: u64,
    pub usage: BufferUsage,
}

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

pub struct Buffer<B: hal::Backend> {
    pub raw: B::UnboundBuffer,
}

pub extern "C"
fn device_create_buffer(
    device: DeviceHandle, desc: BufferDescriptor
) -> BufferHandle {
    let buffer = device.device.create_buffer(desc.size, desc.usage).unwrap();
    BufferHandle::new(Buffer {
        raw: buffer,
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
