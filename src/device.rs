use hal;


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
    pub raw: B::Device,
}

pub struct Buffer<B: hal::Backend> {
    pub raw: B::Buffer,
}
