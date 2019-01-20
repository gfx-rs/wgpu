use crate::registry::{Items, HUB};
use crate::{
    Stored, WeaklyStored,
    BindGroupId, CommandBufferId, ComputePassId, ComputePipelineId, PipelineLayoutId,
};
use super::{BindGroupEntry};

use hal::command::RawCommandBuffer;

use std::iter;


pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    pipeline_layout_id: Option<WeaklyStored<PipelineLayoutId>>, //TODO: strongly `Stored`
    bind_groups: Vec<BindGroupEntry>,
}

impl<B: hal::Backend> ComputePass<B> {
    pub(crate) fn new(raw: B::CommandBuffer, cmb_id: Stored<CommandBufferId>) -> Self {
        ComputePass {
            raw,
            cmb_id,
            pipeline_layout_id: None,
            bind_groups: Vec::new(),
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
pub extern "C" fn wgpu_compute_pass_set_bind_group(
    pass_id: ComputePassId,
    index: u32,
    bind_group_id: BindGroupId,
) {
    let mut pass_guard = HUB.compute_passes.write();
    let pass = pass_guard.get_mut(pass_id);
    let bind_group_guard = HUB.bind_groups.read();
    let bind_group = bind_group_guard.get(bind_group_id);
    let pipeline_layout_guard = HUB.pipeline_layouts.read();

    while pass.bind_groups.len() <= index as usize {
        pass.bind_groups.push(BindGroupEntry::default());
    }
    *pass.bind_groups.get_mut(index as usize).unwrap() = BindGroupEntry {
        layout: Some(bind_group.layout_id.clone()),
        data: Some(Stored {
            value: bind_group_id,
            ref_count: bind_group.life_guard.ref_count.clone(),
        }),
    };

    if let Some(ref pipeline_layout_id) = pass.pipeline_layout_id {
        let pipeline_layout = pipeline_layout_guard.get(pipeline_layout_id.0);
        if pipeline_layout.bind_group_layout_ids[index as usize] == bind_group.layout_id {
            unsafe {
                pass.raw.bind_compute_descriptor_sets(
                    &pipeline_layout.raw,
                    index as usize,
                    iter::once(&bind_group.raw),
                    &[],
                );
            }
        }
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
