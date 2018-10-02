use hal;
use hal::command::RawCommandBuffer;

use registry::{self, Items, Registry};
use {
    Stored,
    CommandBufferId, RenderPassId,
};


pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
}

impl<B: hal::Backend> RenderPass<B> {
    pub fn new(raw: B::CommandBuffer, cmb_id: CommandBufferId) -> Self {
        RenderPass {
            raw,
            cmb_id: Stored(cmb_id),
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_end_pass(
    pass_id: RenderPassId,
) -> CommandBufferId {
    let mut pass = registry::RENDER_PASS_REGISTRY
        .lock()
        .take(pass_id);
    pass.raw.end_render_pass();

    registry::COMMAND_BUFFER_REGISTRY
        .lock()
        .get_mut(pass.cmb_id.0)
        .raw = Some(pass.raw);
    pass.cmb_id.0
}
