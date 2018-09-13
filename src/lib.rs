extern crate gfx_hal as hal;
#[cfg(feature = "gfx-backend-vulkan")]
extern crate gfx_backend_vulkan as back;

mod command;
mod device;
mod handle;
mod instance;

use back::Backend as B;
use hal::Device;

use handle::Handle;


pub type InstanceHandle = Handle<back::Instance>;
pub type AdapterHandle = Handle<hal::Adapter<B>>;
pub type DeviceHandle = Handle<device::Device<B>>;
pub type BufferHandle = Handle<device::Buffer<B>>;
pub type CommandBufferHandle = Handle<command::CommandBuffer<B>>;
pub type RenderPassHandle = Handle<command::RenderPass<B>>;
pub type ComputePassHandle = Handle<command::ComputePass<B>>;

// Instance logic

pub extern "C"
fn create_instance() -> InstanceHandle {
    unimplemented!()
}

pub extern "C"
fn instance_get_adapter(
    instance: InstanceHandle, desc: instance::AdapterDescriptor
) -> AdapterHandle {
    unimplemented!()
}

pub extern "C"
fn adapter_create_device(
    adapter: AdapterHandle, desc: instance::DeviceDescriptor
) -> DeviceHandle {
    unimplemented!()
}

// Device logic

pub extern "C"
fn device_create_buffer(
    device: DeviceHandle, desc: device::BufferDescriptor
) -> BufferHandle {
    //let unbound = device.raw.create_buffer(desc.size, desc.usage).unwrap();
    //let reqs = device.raw.get_buffer_requirements(&unbound);
    unimplemented!()
}

pub extern "C"
fn device_create_command_buffer(
    device: DeviceHandle, desc: device::CommandBufferDescriptor
) -> CommandBufferHandle {
    unimplemented!()
}

// Command Buffer logic

pub extern "C"
fn command_buffer_begin_render_pass(
    command_buffer: CommandBufferHandle
) -> RenderPassHandle {
    unimplemented!()
}

pub extern "C"
fn command_buffer_begin_compute_pass(
) -> ComputePassHandle {
    unimplemented!()
}

// Render Pass logic

pub extern "C"
fn render_pass_draw(
    pass: RenderPassHandle, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32
) {
    unimplemented!()
}

pub extern "C"
fn render_pass_draw_indexed(
    pass: RenderPassHandle, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32
) {
    unimplemented!()
}

pub extern "C"
fn render_pass_end(pass: RenderPassHandle) -> CommandBufferHandle {
    unimplemented!()
}
