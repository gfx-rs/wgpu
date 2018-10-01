mod allocator;
mod compute;
mod render;

pub use self::allocator::*;
pub use self::compute::*;
pub use self::render::*;

use hal;

use {
    BufferId, Color, CommandBufferId, ComputePassId, Origin3d, RenderPassId, TextureId,
    TextureViewId,
};
use registry::{self, Items, Registry};

use std::thread::ThreadId;

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum LoadOp {
    Clear = 0,
    Load = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum StoreOp {
    Store = 0,
}

#[repr(C)]
pub struct RenderPassColorAttachmentDescriptor<T> {
    pub attachment: T,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: Color,
}

#[repr(C)]
pub struct RenderPassDepthStencilAttachmentDescriptor<T> {
    pub attachment: T,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub clear_depth: f32,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
    pub clear_stencil: u32,
}

#[repr(C)]
pub struct RenderPassDescriptor<'a, T: 'a> {
    pub color_attachments: &'a [RenderPassColorAttachmentDescriptor<T>],
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachmentDescriptor<T>>,
}

#[repr(C)]
pub struct BufferCopyView {
    pub buffer: BufferId,
    pub offset: u32,
    pub row_pitch: u32,
    pub image_height: u32,
}

#[repr(C)]
pub struct TextureCopyView {
    pub texture: TextureId,
    pub level: u32,
    pub slice: u32,
    pub origin: Origin3d,
    //TODO: pub aspect: TextureAspect,
}

pub struct CommandBuffer<B: hal::Backend> {
    pub(crate) raw: Option<B::CommandBuffer>,
    fence: B::Fence,
    recorded_thread_id: ThreadId,
}

#[repr(C)]
pub struct CommandBufferDescriptor {}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_begin_render_pass(
    command_buffer_id: CommandBufferId,
    _descriptor: RenderPassDescriptor<TextureViewId>,
) -> RenderPassId {
    let raw = registry::COMMAND_BUFFER_REGISTRY
        .lock()
        .get_mut(command_buffer_id)
        .raw
        .take()
        .unwrap();

    /*TODO:
    raw.begin_render_pass(
        render_pass: &B::RenderPass,
        framebuffer: &B::Framebuffer,
        render_area: pso::Rect,
        clear_values: T,
        hal::SubpassContents::Inline,
    );*/

    registry::RENDER_PASS_REGISTRY
        .lock()
        .register(RenderPass::new(raw, command_buffer_id))
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_begin_compute_pass() -> ComputePassId {
    unimplemented!()
}
