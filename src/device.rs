use hal::{self, Device as _Device};
use memory;

use {BufferHandle, CommandBufferHandle, DeviceHandle};


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
    gpu: hal::Gpu<B>,
    allocator: memory::SmartAllocator<B>,
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn new(gpu: hal::Gpu<B>, mem_props: hal::MemoryProperties) -> Self {
        Device {
            gpu,
            allocator: memory::SmartAllocator::new(mem_props, 1, 1, 1, 1),
        }
    }
}

pub struct Buffer<B: hal::Backend> {
    pub raw: B::Buffer,
}

pub extern "C"
fn device_create_buffer(
    device: DeviceHandle, desc: BufferDescriptor
) -> BufferHandle {
    let buffer = device.gpu.device.create_buffer(desc.size, desc.usage).unwrap();
    BufferHandle::new(Buffer {
        raw: buffer,
    })
}

pub extern "C"
fn device_create_command_buffer(
    device: DeviceHandle, desc: CommandBufferDescriptor
) -> CommandBufferHandle {
    unimplemented!()
}
