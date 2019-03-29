use crate::{
    command::bind::{Binder, Expectation},
    hub::HUB,
    track::{Stitch, TrackerSet},
    BindGroupId, CommandBuffer, CommandBufferId, ComputePassId, ComputePipelineId, Stored,
};

use hal::{
    self,
    command::RawCommandBuffer,
};

use std::iter;


pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    binder: Binder,
    trackers: TrackerSet,
}

impl<B: hal::Backend> ComputePass<B> {
    pub(crate) fn new(raw: B::CommandBuffer, cmb_id: Stored<CommandBufferId>) -> Self {
        ComputePass {
            raw,
            cmb_id,
            binder: Binder::default(),
            trackers: TrackerSet::new(),
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_end_pass(pass_id: ComputePassId) -> CommandBufferId {
    let pass = HUB.compute_passes.unregister(pass_id);

    //TODO: transitions?

    HUB.command_buffers.write()[pass.cmb_id.value]
        .raw
        .push(pass.raw);
    pass.cmb_id.value
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_dispatch(pass_id: ComputePassId, x: u32, y: u32, z: u32) {
    unsafe {
        HUB.compute_passes.write()[pass_id].raw.dispatch([x, y, z]);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_bind_group(
    pass_id: ComputePassId,
    index: u32,
    bind_group_id: BindGroupId,
) {
    let mut pass_guard = HUB.compute_passes.write();
    let pass = &mut pass_guard[pass_id];
    let bind_group_guard = HUB.bind_groups.read();
    let bind_group = &bind_group_guard[bind_group_id];

    //Note: currently, WebGPU compute passes have synchronization defined
    // at a dispatch granularity, so we insert the necessary barriers here.

    CommandBuffer::insert_barriers(
        &mut pass.raw,
        &mut pass.trackers,
        &bind_group.used,
        Stitch::Last,
        &*HUB.buffers.read(),
        &*HUB.textures.read(),
    );

    if let Some(pipeline_layout_id) =
        pass.binder
            .provide_entry(index as usize, bind_group_id, bind_group)
    {
        let pipeline_layout_guard = HUB.pipeline_layouts.read();
        unsafe {
            pass.raw.bind_compute_descriptor_sets(
                &pipeline_layout_guard[pipeline_layout_id].raw,
                index as usize,
                iter::once(&bind_group.raw),
                &[],
            );
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_pipeline(
    pass_id: ComputePassId,
    pipeline_id: ComputePipelineId,
) {
    let mut pass_guard = HUB.compute_passes.write();
    let pass = &mut pass_guard[pass_id];
    let pipeline_guard = HUB.compute_pipelines.read();
    let pipeline = &pipeline_guard[pipeline_id];

    unsafe {
        pass.raw.bind_compute_pipeline(&pipeline.raw);
    }

    if pass.binder.pipeline_layout_id == Some(pipeline.layout_id.clone()) {
        return;
    }

    let pipeline_layout_guard = HUB.pipeline_layouts.read();
    let pipeline_layout = &pipeline_layout_guard[pipeline.layout_id];
    let bing_group_guard = HUB.bind_groups.read();

    pass.binder.pipeline_layout_id = Some(pipeline.layout_id.clone());
    pass.binder
        .ensure_length(pipeline_layout.bind_group_layout_ids.len());

    for (index, (entry, &bgl_id)) in pass
        .binder
        .entries
        .iter_mut()
        .zip(&pipeline_layout.bind_group_layout_ids)
        .enumerate()
    {
        if let Expectation::Match(bg_id) = entry.expect_layout(bgl_id) {
            let desc_set = &bing_group_guard[bg_id].raw;
            unsafe {
                pass.raw.bind_compute_descriptor_sets(
                    &pipeline_layout.raw,
                    index,
                    iter::once(desc_set),
                    &[],
                );
            }
        }
    }
}
