use hal;
use hal::command::RawCommandBuffer;

use registry::{self, Items, Registry};
use {CommandBufferId, RenderPassId};

pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: CommandBufferId,
}

// This is needed for `cmb_id` - would be great to remove.
#[cfg(not(feature = "remote"))]
unsafe impl<B: hal::Backend> Sync for RenderPass<B> {}

impl<B: hal::Backend> RenderPass<B> {
    pub fn new(raw: B::CommandBuffer, cmb_id: CommandBufferId) -> Self {
        RenderPass {
            raw,
            cmb_id,
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_end_pass(
    render_pass_id: RenderPassId,
) -> CommandBufferId {
    let mut rp = registry::RENDER_PASS_REGISTRY
        .lock()
        .take(render_pass_id);
    rp.raw.end_render_pass();

    registry::COMMAND_BUFFER_REGISTRY
        .lock()
        .get_mut(rp.cmb_id)
        .raw = Some(rp.raw);
    rp.cmb_id
}
