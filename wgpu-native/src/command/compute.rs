use hal;

use registry::{HUB, Items, Registry};
use {
    Stored,
    CommandBufferId, ComputePassId
};


pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
}

impl<B: hal::Backend> ComputePass<B> {
    pub fn new(raw: B::CommandBuffer, cmb_id: CommandBufferId) -> Self {
        ComputePass {
            raw,
            cmb_id: Stored(cmb_id),
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_end_pass(
    pass_id: ComputePassId,
) -> CommandBufferId {
    let pass = HUB.compute_passes
        .lock()
        .take(pass_id);

    HUB.command_buffers
        .lock()
        .get_mut(pass.cmb_id.0)
        .raw = Some(pass.raw);
    pass.cmb_id.0
}
