use hal::{self, Device as _Device};

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
    pub gpu: hal::Gpu<B>,
}

pub struct Buffer<B: hal::Backend> {
    pub raw: B::Buffer,
}

pub extern "C"
fn device_create_buffer(
    device: DeviceHandle, desc: BufferDescriptor
) -> BufferHandle {
    //let unbound = device.raw.create_buffer(desc.size, desc.usage).unwrap();
    //let reqs = device.raw.get_buffer_requirements(&unbound);
    unimplemented!()
}

pub extern "C"
fn device_create_command_buffer(
    device: DeviceHandle, desc: CommandBufferDescriptor
) -> CommandBufferHandle {
    unimplemented!()
}
