mod compute;
mod render;

pub use self::compute::*;
pub use self::render::*;

use hal;

use {BufferHandle, Color, CommandBufferHandle, ComputePassHandle, Origin3d,
    RenderPassHandle, TextureViewHandle, TextureHandle};


#[repr(C)]
pub enum LoadOp {
    Clear = 0,
    Load = 1,
}

#[repr(C)]
pub enum StoreOp {
    Store = 0,
}

#[repr(C)]
pub struct RenderPassColorAttachmentDescriptor {
    pub attachment: TextureViewHandle,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: Color,
}

#[repr(C)]
pub struct RenderPassDepthStencilAttachmentDescriptor {
    pub attachment: TextureViewHandle,
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
    pub buffer: BufferHandle,
    pub offset: u32,
    pub row_pitch: u32,
    pub image_height: u32,
}

#[repr(C)]
pub struct TextureCopyView {
    pub texture: TextureHandle,
    pub level: u32,
    pub slice: u32,
    pub origin: Origin3d,
    //TODO: pub aspect: TextureAspect,
}

pub struct CommandBuffer<B: hal::Backend> {
    raw: B::CommandBuffer,
}

#[repr(C)]
pub struct CommandBufferDescriptor;

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
