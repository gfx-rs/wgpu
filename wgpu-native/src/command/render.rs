use registry::{HUB, Items, Registry};
use track::{BufferTracker, TextureTracker};
use {
    CommandBuffer, Stored,
    CommandBufferId, RenderPassId,
};

use hal;
use hal::command::RawCommandBuffer;


pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    buffer_tracker: BufferTracker,
    texture_tracker: TextureTracker,
}

impl<B: hal::Backend> RenderPass<B> {
    pub(crate) fn new(
        raw: B::CommandBuffer,
        cmb_id: Stored<CommandBufferId>,
    ) -> Self {
        RenderPass {
            raw,
            cmb_id,
            buffer_tracker: BufferTracker::new(),
            texture_tracker: TextureTracker::new(),
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_end_pass(
    pass_id: RenderPassId,
) -> CommandBufferId {
    let mut pass = HUB.render_passes
        .lock()
        .take(pass_id);
    pass.raw.end_render_pass();

    let mut cmb_guard = HUB.command_buffers.lock();
    let cmb = cmb_guard.get_mut(pass.cmb_id.value);

    if let Some(ref mut last) = cmb.raw.last_mut() {
        CommandBuffer::insert_barriers(
            last,
            cmb.buffer_tracker.consume(&pass.buffer_tracker),
            cmb.texture_tracker.consume(&pass.texture_tracker),
        );
        last.finish();
    }

    cmb.raw.push(pass.raw);
    pass.cmb_id.value
}
