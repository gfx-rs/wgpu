mod allocator;
mod compute;
mod render;

pub use self::allocator::CommandAllocator;
pub use self::compute::*;
pub use self::render::*;

use hal;

use {
    Color, Origin3d, Stored,
    BufferId, CommandBufferId, ComputePassId, DeviceId, RenderPassId, TextureId, TextureViewId,
};
use registry::{HUB, Items, Registry};
use track::{BufferTracker, TextureTracker};

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
    pub(crate) raw: Vec<B::CommandBuffer>,
    fence: B::Fence,
    recorded_thread_id: ThreadId,
    device_id: Stored<DeviceId>,
    buffer_tracker: BufferTracker,
    texture_tracker: TextureTracker,
}

#[repr(C)]
pub struct CommandBufferDescriptor {}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_begin_render_pass(
    command_buffer_id: CommandBufferId,
    _descriptor: RenderPassDescriptor<TextureViewId>,
) -> RenderPassId {
    let mut cmb_guard = HUB.command_buffers.lock();
    let cmb = cmb_guard.get_mut(command_buffer_id);

    let device_guard = HUB.devices.lock();
    let device = device_guard.get(cmb.device_id.0);

    let current_comb = device.com_allocator.extend(cmb);

    //let render_pass = device.create_render_pass();
    //let framebuffer = device.create_framebuffer();

    /*TODO:
    raw.begin_render_pass(
        render_pass: &B::RenderPass,
        framebuffer: &B::Framebuffer,
        render_area: pso::Rect,
        clear_values: T,
        hal::SubpassContents::Inline,
    );*/

    HUB.render_passes
        .lock()
        .register(RenderPass::new(
            current_comb,
            command_buffer_id,
        ))
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_begin_compute_pass(
    command_buffer_id: CommandBufferId,
) -> ComputePassId {
    let mut cmb_guard = HUB.command_buffers.lock();
    let cmb = cmb_guard.get_mut(command_buffer_id);

    let raw = cmb.raw.pop().unwrap();

    HUB.compute_passes
        .lock()
        .register(ComputePass::new(raw, command_buffer_id))
}
