mod compute;
mod render;

pub use self::compute::*;
pub use self::render::*;

use hal;

use {CommandBufferHandle, ComputePassHandle, RenderPassHandle};


pub struct CommandBuffer<B: hal::Backend> {
    raw: B::CommandBuffer,
}

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
