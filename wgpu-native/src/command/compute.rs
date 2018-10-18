use registry::{HUB, Items, Registry};
use {
    Stored,
    BindGroupId, CommandBufferId, ComputePassId, ComputePipelineId,
};

use hal;
use hal::command::RawCommandBuffer;

use std::iter;


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
        .raw
        .push(pass.raw);
    pass.cmb_id.0
}

pub extern "C" fn wgpu_compute_pass_set_bind_group(
    pass_id: ComputePassId, index: u32, bind_group_id: BindGroupId,
) {
    let bind_group_guard = HUB.bind_groups.lock();
    let set = &bind_group_guard.get(bind_group_id).raw;
    let layout = unimplemented!();
    // see https://github.com/gpuweb/gpuweb/pull/93

    HUB.compute_passes
        .lock()
        .get_mut(pass_id)
        .raw
        .bind_compute_descriptor_sets(layout, index as usize, iter::once(set), &[]);
}

pub extern "C" fn wgpu_compute_pass_set_pipeline(
    pass_id: ComputePassId, pipeline_id: ComputePipelineId,
) {
    let pipeline_guard = HUB.compute_pipelines.lock();
    let pipeline = &pipeline_guard.get(pipeline_id).raw;

    HUB.compute_passes
        .lock()
        .get_mut(pass_id)
        .raw
        .bind_compute_pipeline(pipeline);
}

pub extern "C" fn wgpu_compute_pass_dispatch(
    pass_id: ComputePassId, x: u32, y: u32, z: u32,
) {
    HUB.compute_passes
        .lock()
        .get_mut(pass_id)
        .raw
        .dispatch([x, y, z]);
}
