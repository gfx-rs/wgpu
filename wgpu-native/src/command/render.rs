use registry::{HUB, Items, Registry};
use track::{Tracker};
use {
    Stored,
    CommandBufferId, RenderPassId,
};

use hal;
use hal::command::RawCommandBuffer;

use std::iter;


pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
    parent: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    tracker: Tracker,
}

impl<B: hal::Backend> RenderPass<B> {
    pub fn new(
        raw: B::CommandBuffer,
        parent: B::CommandBuffer,
        cmb_id: CommandBufferId,
    ) -> Self {
        RenderPass {
            raw,
            parent,
            cmb_id: Stored(cmb_id),
            tracker: Tracker::default(),
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

    let combs = iter::once(pass.parent)
        .chain(iter::once(pass.raw));
    HUB.command_buffers
        .lock()
        .get_mut(pass.cmb_id.0)
        .raw
        .extend(combs);

    pass.cmb_id.0
}
