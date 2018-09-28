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
pub struct RenderPassColorAttachmentDescriptor {
    pub attachment: TextureViewId,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: Color,
}

#[repr(C)]
pub struct RenderPassDepthStencilAttachmentDescriptor {
    pub attachment: TextureViewId,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub clear_depth: f32,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
    pub clear_stencil: u32,
}

#[repr(C)]
pub struct RenderPassDescriptor<'a> {
    pub color_attachments: &'a [RenderPassColorAttachmentDescriptor],
    pub depth_stencil_attachment: RenderPassDepthStencilAttachmentDescriptor,
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
    pub(crate) raw: B::CommandBuffer,
    fence: B::Fence,
    recorded_thread_id: ThreadId,
}

#[repr(C)]
pub struct CommandBufferDescriptor {}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_begin_render_pass(
    _command_buffer: CommandBufferId,
) -> RenderPassId {
    unimplemented!()
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_begin_compute_pass() -> ComputePassId {
    unimplemented!()
}
