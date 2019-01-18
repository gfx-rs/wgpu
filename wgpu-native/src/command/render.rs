use crate::resource::BufferUsageFlags;
use crate::registry::{Items, HUB};
use crate::track::{BufferTracker, TextureTracker, TrackPermit};
use crate::{
    CommandBuffer, Stored,
    BufferId, CommandBufferId, RenderPassId,
};

use hal::command::RawCommandBuffer;


pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    buffer_tracker: BufferTracker,
    texture_tracker: TextureTracker,
}

impl<B: hal::Backend> RenderPass<B> {
    pub(crate) fn new(raw: B::CommandBuffer, cmb_id: Stored<CommandBufferId>) -> Self {
        RenderPass {
            raw,
            cmb_id,
            buffer_tracker: BufferTracker::new(),
            texture_tracker: TextureTracker::new(),
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_end_pass(pass_id: RenderPassId) -> CommandBufferId {
    let mut pass = HUB.render_passes.write().take(pass_id);
    unsafe {
        pass.raw.end_render_pass();
    }

    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = cmb_guard.get_mut(pass.cmb_id.value);

    if let Some(ref mut last) = cmb.raw.last_mut() {
        CommandBuffer::insert_barriers(
            last,
            cmb.buffer_tracker.consume(&pass.buffer_tracker),
            cmb.texture_tracker.consume(&pass.texture_tracker),
            &*HUB.buffers.read(),
            &*HUB.textures.read(),
        );
        unsafe { last.finish() };
    }

    cmb.raw.push(pass.raw);
    pass.cmb_id.value
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_index_buffer(
    pass_id: RenderPassId, buffer_id: BufferId, offset: u32
) {
    let mut pass_guard = HUB.render_passes.write();
    let buffer_guard = HUB.buffers.read();

    let pass = pass_guard.get_mut(pass_id);
    let (buffer, _) = pass.buffer_tracker
        .get_with_usage(
            &*buffer_guard,
            buffer_id,
            BufferUsageFlags::INDEX,
            TrackPermit::EXTEND,
        )
        .unwrap();
        buffer_guard.get(buffer_id);

    let view = hal::buffer::IndexBufferView {
        buffer: &buffer.raw,
        offset: offset as u64,
        index_type: hal::IndexType::U16, //TODO?
    };

    unsafe {
        pass.raw.bind_index_buffer(view);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_vertex_buffers(
    pass_id: RenderPassId, buffers: &[BufferId], offsets: &[u32]
) {
    let mut pass_guard = HUB.render_passes.write();
    let buffer_guard = HUB.buffers.read();

    let pass = pass_guard.get_mut(pass_id);
    for &id in buffers {
        pass.buffer_tracker
            .get_with_usage(
                &*buffer_guard,
                id,
                BufferUsageFlags::VERTEX,
                TrackPermit::EXTEND,
            )
            .unwrap();
    }

    assert_eq!(buffers.len(), offsets.len());
    let buffers = buffers
        .iter()
        .map(|&id| &buffer_guard.get(id).raw)
        .zip(offsets.iter().map(|&off| off as u64));

    unsafe {
        pass.raw.bind_vertex_buffers(0, buffers);
    }
}
