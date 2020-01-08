/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    command::{
        bind::{Binder, LayoutChange},
        CommandBuffer,
        OffsetIndex,
    },
    device::{all_buffer_stages, BIND_BUFFER_ALIGNMENT},
    hub::{GfxBackend, Global, IdentityFilter, Token},
    id,
    resource::BufferUsage,
    track::TrackerSet,
    BufferAddress,
    Stored,
};

use hal::command::CommandBuffer as _;

use std::{iter, ops::Range};


#[non_exhaustive]
#[derive(Debug)]
pub enum ComputeCommand {
    SetBindGroup {
        index: u32,
        bind_group_id: id::BindGroupId,
        offset_indices: Range<OffsetIndex>,
    },
    SetPipeline(id::ComputePipelineId),
    Dispatch([u32; 3]),
    DispatchIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
    },
}

#[derive(Copy, Clone, Debug)]
pub struct StandaloneComputePass<'a> {
    pub commands: &'a [ComputeCommand],
    pub offsets: &'a [BufferAddress],
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor {
    pub todo: u32,
}

#[derive(Debug)]
pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<id::CommandBufferId>,
    binder: Binder,
    trackers: TrackerSet,
}

impl<B: hal::Backend> ComputePass<B> {
    pub(crate) fn new(
        raw: B::CommandBuffer,
        cmb_id: Stored<id::CommandBufferId>,
        trackers: TrackerSet,
        max_bind_groups: u32,
    ) -> Self {
        ComputePass {
            raw,
            cmb_id,
            binder: Binder::new(max_bind_groups),
            trackers,
        }
    }
}

// Common routines between render/compute

impl<F: IdentityFilter<id::ComputePassId>> Global<F> {
    pub fn compute_pass_end_pass<B: GfxBackend>(&self, pass_id: id::ComputePassId) {
        let mut token = Token::root();
        let hub = B::hub(self);
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let (pass, _) = hub.compute_passes.unregister(pass_id, &mut token);
        let cmb = &mut cmb_guard[pass.cmb_id.value];

        // There are no transitions to be made: we've already been inserting barriers
        // into the parent command buffer while recording this compute pass.
        log::debug!("Compute pass {:?} {:#?}", pass_id, pass.trackers);
        cmb.trackers = pass.trackers;
        cmb.raw.push(pass.raw);
    }
}

impl<F> Global<F> {
    pub fn command_encoder_run_compute_pass<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        pass: StandaloneComputePass,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[encoder_id];
        let raw = cmb.raw.last_mut().unwrap();
        let mut binder = Binder::new(cmb.features.max_bind_groups);

        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
        let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
        let (pipeline_guard, mut token) = hub.compute_pipelines.read(&mut token);
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);

        for command in pass.commands {
            match *command {
                ComputeCommand::SetBindGroup { index, bind_group_id, ref offset_indices } => {
                    let offsets = &pass.offsets[offset_indices.start as usize .. offset_indices.end as usize];
                    if cfg!(debug_assertions) {
                        for off in offsets {
                            assert_eq!(
                                *off % BIND_BUFFER_ALIGNMENT,
                                0,
                                "Misaligned dynamic buffer offset: {} does not align with {}",
                                off,
                                BIND_BUFFER_ALIGNMENT
                            );
                        }
                    }

                    let bind_group = cmb
                        .trackers
                        .bind_groups
                        .use_extend(&*bind_group_guard, bind_group_id, (), ())
                        .unwrap();
                    assert_eq!(bind_group.dynamic_count, offsets.len());

                    log::trace!(
                        "Encoding barriers on binding of {:?} to {:?}",
                        bind_group_id,
                        encoder_id
                    );
                    CommandBuffer::insert_barriers(
                        raw,
                        &mut cmb.trackers,
                        &bind_group.used,
                        &*buffer_guard,
                        &*texture_guard,
                    );

                    if let Some((pipeline_layout_id, follow_ups)) = binder
                        .provide_entry(index as usize, bind_group_id, bind_group, offsets)
                    {
                        let bind_groups = iter::once(bind_group.raw.raw())
                            .chain(follow_ups.clone().map(|(bg_id, _)| bind_group_guard[bg_id].raw.raw()));
                        unsafe {
                            raw.bind_compute_descriptor_sets(
                                &pipeline_layout_guard[pipeline_layout_id].raw,
                                index as usize,
                                bind_groups,
                                offsets
                                    .iter()
                                    .chain(follow_ups.flat_map(|(_, offsets)| offsets))
                                    .map(|&off| off as hal::command::DescriptorSetOffset),
                            );
                        }
                    }
                }
                ComputeCommand::SetPipeline(pipeline_id) => {
                    let pipeline = &pipeline_guard[pipeline_id];

                    unsafe {
                        raw.bind_compute_pipeline(&pipeline.raw);
                    }

                    // Rebind resources
                    if binder.pipeline_layout_id != Some(pipeline.layout_id) {
                        let pipeline_layout = &pipeline_layout_guard[pipeline.layout_id];
                        binder.pipeline_layout_id = Some(pipeline.layout_id);
                        binder
                            .reset_expectations(pipeline_layout.bind_group_layout_ids.len());
                        let mut is_compatible = true;

                        for (index, (entry, &bgl_id)) in binder
                            .entries
                            .iter_mut()
                            .zip(&pipeline_layout.bind_group_layout_ids)
                            .enumerate()
                        {
                            match entry.expect_layout(bgl_id) {
                                LayoutChange::Match(bg_id, offsets) if is_compatible => {
                                    let desc_set = bind_group_guard[bg_id].raw.raw();
                                    unsafe {
                                        raw.bind_compute_descriptor_sets(
                                            &pipeline_layout.raw,
                                            index,
                                            iter::once(desc_set),
                                            offsets.iter().map(|offset| *offset as u32),
                                        );
                                    }
                                }
                                LayoutChange::Match(..) | LayoutChange::Unchanged => {}
                                LayoutChange::Mismatch => {
                                    is_compatible = false;
                                }
                            }
                        }
                    }
                }
                ComputeCommand::Dispatch(groups) => {
                    unsafe {
                        raw.dispatch(groups);
                    }
                }
                ComputeCommand::DispatchIndirect { buffer_id, offset } => {
                    let (src_buffer, src_pending) = cmb.trackers.buffers.use_replace(
                        &*buffer_guard,
                        buffer_id,
                        (),
                        BufferUsage::INDIRECT,
                    );
                    assert!(src_buffer.usage.contains(BufferUsage::INDIRECT));

                    let barriers = src_pending.map(|pending| pending.into_hal(src_buffer));

                    unsafe {
                        raw.pipeline_barrier(
                            all_buffer_stages() .. all_buffer_stages(),
                            hal::memory::Dependencies::empty(),
                            barriers,
                        );
                        raw.dispatch_indirect(&src_buffer.raw, offset);
                    }
                }
            }
        }
    }

    pub fn compute_pass_set_bind_group<B: GfxBackend>(
        &self,
        pass_id: id::ComputePassId,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: &[BufferAddress],
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
        let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
        let (mut pass_guard, mut token) = hub.compute_passes.write(&mut token);
        let pass = &mut pass_guard[pass_id];

        let bind_group = pass
            .trackers
            .bind_groups
            .use_extend(&*bind_group_guard, bind_group_id, (), ())
            .unwrap();

        assert_eq!(bind_group.dynamic_count, offsets.len());

        if cfg!(debug_assertions) {
            for off in offsets {
                assert_eq!(
                    *off % BIND_BUFFER_ALIGNMENT,
                    0,
                    "Misaligned dynamic buffer offset: {} does not align with {}",
                    off,
                    BIND_BUFFER_ALIGNMENT
                );
            }
        }

        //Note: currently, WebGPU compute passes have synchronization defined
        // at a dispatch granularity, so we insert the necessary barriers here.
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);

        log::trace!(
            "Encoding barriers on binding of {:?} in pass {:?}",
            bind_group_id,
            pass_id
        );
        CommandBuffer::insert_barriers(
            &mut pass.raw,
            &mut pass.trackers,
            &bind_group.used,
            &*buffer_guard,
            &*texture_guard,
        );

        if let Some((pipeline_layout_id, follow_ups)) = pass
            .binder
            .provide_entry(index as usize, bind_group_id, bind_group, offsets)
        {
            let bind_groups = iter::once(bind_group.raw.raw())
                .chain(follow_ups.clone().map(|(bg_id, _)| bind_group_guard[bg_id].raw.raw()));
            unsafe {
                pass.raw.bind_compute_descriptor_sets(
                    &pipeline_layout_guard[pipeline_layout_id].raw,
                    index as usize,
                    bind_groups,
                    offsets
                        .iter()
                        .chain(follow_ups.flat_map(|(_, offsets)| offsets))
                        .map(|&off| off as hal::command::DescriptorSetOffset),
                );
            }
        };
    }

    // Compute-specific routines

    pub fn compute_pass_dispatch<B: GfxBackend>(
        &self,
        pass_id: id::ComputePassId,
        x: u32,
        y: u32,
        z: u32,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut pass_guard, _) = hub.compute_passes.write(&mut token);
        unsafe {
            pass_guard[pass_id].raw.dispatch([x, y, z]);
        }
    }

    pub fn compute_pass_dispatch_indirect<B: GfxBackend>(
        &self,
        pass_id: id::ComputePassId,
        indirect_buffer_id: id::BufferId,
        indirect_offset: BufferAddress,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut pass_guard, mut token) = hub.compute_passes.write(&mut token);
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        let pass = &mut pass_guard[pass_id];

        let (src_buffer, src_pending) = pass.trackers.buffers.use_replace(
            &*buffer_guard,
            indirect_buffer_id,
            (),
            BufferUsage::INDIRECT,
        );
        assert!(src_buffer.usage.contains(BufferUsage::INDIRECT));

        let barriers = src_pending.map(|pending| pending.into_hal(src_buffer));

        unsafe {
            pass.raw.pipeline_barrier(
                all_buffer_stages() .. all_buffer_stages(),
                hal::memory::Dependencies::empty(),
                barriers,
            );
            pass.raw.dispatch_indirect(&src_buffer.raw, indirect_offset);
        }
    }

    pub fn compute_pass_set_pipeline<B: GfxBackend>(
        &self,
        pass_id: id::ComputePassId,
        pipeline_id: id::ComputePipelineId,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
        let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
        let (mut pass_guard, mut token) = hub.compute_passes.write(&mut token);
        let pass = &mut pass_guard[pass_id];
        let (pipeline_guard, _) = hub.compute_pipelines.read(&mut token);
        let pipeline = &pipeline_guard[pipeline_id];

        unsafe {
            pass.raw.bind_compute_pipeline(&pipeline.raw);
        }

        // Rebind resources
        if pass.binder.pipeline_layout_id != Some(pipeline.layout_id) {
            let pipeline_layout = &pipeline_layout_guard[pipeline.layout_id];
            pass.binder.pipeline_layout_id = Some(pipeline.layout_id);
            pass.binder
                .reset_expectations(pipeline_layout.bind_group_layout_ids.len());
            let mut is_compatible = true;

            for (index, (entry, &bgl_id)) in pass
                .binder
                .entries
                .iter_mut()
                .zip(&pipeline_layout.bind_group_layout_ids)
                .enumerate()
            {
                match entry.expect_layout(bgl_id) {
                    LayoutChange::Match(bg_id, offsets) if is_compatible => {
                        let desc_set = bind_group_guard[bg_id].raw.raw();
                        unsafe {
                            pass.raw.bind_compute_descriptor_sets(
                                &pipeline_layout.raw,
                                index,
                                iter::once(desc_set),
                                offsets.iter().map(|offset| *offset as u32),
                            );
                        }
                    }
                    LayoutChange::Match(..) | LayoutChange::Unchanged => {}
                    LayoutChange::Mismatch => {
                        is_compatible = false;
                    }
                }
            }
        }
    }
}
