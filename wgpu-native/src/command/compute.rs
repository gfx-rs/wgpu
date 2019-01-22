use crate::command::bind::{Binder};
use crate::registry::{Items, HUB};
use crate::track::{BufferTracker, TextureTracker};
use crate::{
    Stored, CommandBuffer,
    BindGroupId, CommandBufferId, ComputePassId, ComputePipelineId,
};

use hal;
use hal::command::RawCommandBuffer;

use std::iter;


pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    binder: Binder,
    buffer_tracker: BufferTracker,
    texture_tracker: TextureTracker,
}

impl<B: hal::Backend> ComputePass<B> {
    pub(crate) fn new(raw: B::CommandBuffer, cmb_id: Stored<CommandBufferId>) -> Self {
        ComputePass {
            raw,
            cmb_id,
            binder: Binder::default(),
            buffer_tracker: BufferTracker::new(),
            texture_tracker: TextureTracker::new(),
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
    let pass = pass_guard.get_mut(pass_id);
    let bind_group_guard = HUB.bind_groups.read();
    let bind_group = bind_group_guard.get(bind_group_id);

    CommandBuffer::insert_barriers(
        &mut pass.raw,
        pass.buffer_tracker.consume_by_replace(&bind_group.used_buffers),
        pass.texture_tracker.consume_by_replace(&bind_group.used_textures),
        &*HUB.buffers.read(),
        &*HUB.textures.read(),
    );

    if let Some(pipeline_layout_id) = pass.binder.provide_entry(index as usize, bind_group_id, bind_group) {
        let pipeline_layout_guard = HUB.pipeline_layouts.read();
        let pipeline_layout = pipeline_layout_guard.get(pipeline_layout_id);
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

#[no_mangle]
pub extern "C" fn wgpu_compute_pass_set_pipeline(
    pass_id: ComputePassId,
    pipeline_id: ComputePipelineId,
) {
    let mut pass_guard = HUB.compute_passes.write();
    let pass = pass_guard.get_mut(pass_id);
    let pipeline_guard = HUB.compute_pipelines.read();
    let pipeline = pipeline_guard.get(pipeline_id);

    unsafe {
        pass.raw.bind_compute_pipeline(&pipeline.raw);
    }

    if pass.binder.pipeline_layout_id == Some(pipeline.layout_id.clone()) {
        return
    }

    let pipeline_layout_guard = HUB.pipeline_layouts.read();
    let pipeline_layout = pipeline_layout_guard.get(pipeline.layout_id.0);
    let bing_group_guard = HUB.bind_groups.read();

    pass.binder.pipeline_layout_id = Some(pipeline.layout_id.clone());
    pass.binder.ensure_length(pipeline_layout.bind_group_layout_ids.len());

    for (index, (entry, bgl_id)) in pass.binder.entries
        .iter_mut()
        .zip(&pipeline_layout.bind_group_layout_ids)
        .enumerate()
    {
        if let Some(bg_id) = entry.expect_layout(bgl_id.0) {
            let bind_group = bing_group_guard.get(bg_id);
            unsafe {
                pass.raw.bind_compute_descriptor_sets(
                    &pipeline_layout.raw,
                    index,
                    iter::once(&bind_group.raw),
                    &[]
                );
            }
        }
    }
}
