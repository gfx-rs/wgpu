use hal;

use registry::{self, Items, Registry};
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
    let pass = registry::COMPUTE_PASS_REGISTRY
        .lock()
        .take(pass_id);

    registry::COMMAND_BUFFER_REGISTRY
        .lock()
        .get_mut(pass.cmb_id.0)
        .raw = Some(pass.raw);
    pass.cmb_id.0
}
