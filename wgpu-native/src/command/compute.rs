use crate::command::bind::Binder;
use crate::registry::{Items, HUB};
use crate::{
    Stored,
    BindGroupId, CommandBufferId, ComputePassId, ComputePipelineId,
};

use hal::command::RawCommandBuffer;

use std::iter;


pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    binder: Binder,
}

impl<B: hal::Backend> ComputePass<B> {
    pub(crate) fn new(raw: B::CommandBuffer, cmb_id: Stored<CommandBufferId>) -> Self {
        ComputePass {
            raw,
            cmb_id,
            binder: Binder::default(),
        }
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
pub extern "C" fn wgpu_compute_pass_dispatch(pass_id: ComputePassId, x: u32, y: u32, z: u32) {
    unsafe {
        HUB.compute_passes
            .write()
            .get_mut(pass_id)
            .raw
            .dispatch([x, y, z]);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_bind_group(
    pass_id: ComputePassId,
    index: u32,
    bind_group_id: BindGroupId,
) {
    let mut pass_guard = HUB.compute_passes.write();
    let ComputePass { ref mut raw, ref mut binder, .. } = *pass_guard.get_mut(pass_id);

    binder.bind_group(index as usize, bind_group_id, |pipeline_layout, desc_set| unsafe {
        raw.bind_compute_descriptor_sets(
            pipeline_layout,
            index as usize,
            iter::once(desc_set),
            &[],
        );
    });
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_pipeline(
    pass_id: ComputePassId,
    pipeline_id: ComputePipelineId,
) {
    let mut pass_guard = HUB.compute_passes.write();
    let ComputePass { ref mut raw, ref mut binder, .. } = *pass_guard.get_mut(pass_id);

    let pipeline_guard = HUB.compute_pipelines.read();
    let pipeline = pipeline_guard.get(pipeline_id);

    unsafe {
        raw.bind_compute_pipeline(&pipeline.raw);
    }
    binder.change_layout(pipeline.layout_id.0, |pipeline_layout, index, desc_set| unsafe {
        raw.bind_compute_descriptor_sets(
            pipeline_layout,
            index,
            iter::once(desc_set),
            &[],
        );
    });
}
