use crate::registry::{Items, HUB};
use crate::{BindGroupId, CommandBufferId, ComputePassId, ComputePipelineId, Stored};

use hal::command::RawCommandBuffer;

use std::iter;


pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
}

impl<B: hal::Backend> ComputePass<B> {
    pub(crate) fn new(raw: B::CommandBuffer, cmb_id: Stored<CommandBufferId>) -> Self {
        ComputePass { raw, cmb_id }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_end_pass(pass_id: ComputePassId) -> CommandBufferId {
    let pass = HUB.compute_passes.write().take(pass_id);

    HUB.command_buffers
        .write()
        .get_mut(pass.cmb_id.value)
        .raw
        .push(pass.raw);
    pass.cmb_id.value
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_bind_group(
    pass_id: ComputePassId,
    index: u32,
    bind_group_id: BindGroupId,
) {
    let bind_group_guard = HUB.bind_groups.read();
    let set = &bind_group_guard.get(bind_group_id).raw;
    let layout = unimplemented!();
    // see https://github.com/gpuweb/gpuweb/pull/93

    unsafe {
        HUB.compute_passes
            .write()
            .get_mut(pass_id)
            .raw
            .bind_compute_descriptor_sets(layout, index as usize, iter::once(set), &[]);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_pipeline(
    pass_id: ComputePassId,
    pipeline_id: ComputePipelineId,
) {
    let pipeline_guard = HUB.compute_pipelines.read();
    let pipeline = &pipeline_guard.get(pipeline_id).raw;

    unsafe {
        HUB.compute_passes
            .write()
            .get_mut(pass_id)
            .raw
            .bind_compute_pipeline(pipeline);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_dispatch(pass_id: ComputePassId, x: u32, y: u32, z: u32) {
    unsafe {
        HUB.compute_passes
            .write()
            .get_mut(pass_id)
            .raw
            .dispatch([x, y, z]);
    }
}
