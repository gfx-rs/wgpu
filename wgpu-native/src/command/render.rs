use hal;

use {CommandBuffer, CommandBufferHandle, RenderPassHandle};


pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
}

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
    match pass.unbox() {
        Some(pass) => CommandBufferHandle::new(CommandBuffer {
            raw: pass.raw,
        }),
        None => CommandBufferHandle::null(),
    }
}
