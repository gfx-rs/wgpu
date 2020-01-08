/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    command::{
        bind::{Binder, LayoutChange},
        CommandBuffer,
        RawPass,
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
use peek_poke::{Peek, PeekCopy, Poke};

use std::{convert::TryInto, iter, mem, ptr, slice};


#[derive(Clone, Copy, Debug, PeekCopy, Poke)]
pub enum ComputeCommand {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: u8,
        bind_group_id: id::BindGroupId,
    },
    SetPipeline(id::ComputePipelineId),
    Dispatch([u32; 3]),
    DispatchIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
    },
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
        raw_data: &[u8],
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

        let mut peeker = raw_data.as_ptr();
        let raw_data_end = unsafe {
            raw_data.as_ptr().add(raw_data.len())
        };
        let mut command = ComputeCommand::Dispatch([0; 3]); // dummy
        while unsafe { peeker.add(mem::size_of::<ComputeCommand>()) } <= raw_data_end {
            peeker = unsafe { command.peek_from(peeker) };
            match command {
                ComputeCommand::SetBindGroup { index, num_dynamic_offsets, bind_group_id } => {
                    debug_assert_eq!(peeker.align_offset(mem::align_of::<BufferAddress>()), 0);
                    let extra_size = (num_dynamic_offsets as usize) * mem::size_of::<BufferAddress>();
                    let end = unsafe { peeker.add(extra_size) };
                    assert!(end <= raw_data_end);
                    let offsets = unsafe {
                        slice::from_raw_parts(peeker as *const BufferAddress, num_dynamic_offsets as usize)
                    };
                    peeker = end;
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

        assert_eq!(peeker, raw_data_end);
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

impl RawPass {
    #[inline]
    unsafe fn encode(&mut self, command: &ComputeCommand) {
        self.ensure_extra_size(mem::size_of::<ComputeCommand>());
        self.data = command.poke_into(self.data);
    }

    #[inline]
    unsafe fn encode_with<T>(&mut self, command: &ComputeCommand, extra: &[T]) {
        let extra_size = extra.len() * mem::size_of::<T>();
        self.ensure_extra_size(mem::size_of::<ComputeCommand>() + extra_size);
        self.data = command.poke_into(self.data);
        debug_assert_eq!(self.data.align_offset(mem::align_of::<T>()), 0);
        ptr::copy_nonoverlapping(extra.as_ptr(), self.data as *mut T, extra.len());
        self.data = self.data.add(extra_size);
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_raw_compute_pass_set_bind_group(
        &mut self,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const BufferAddress,
        offset_length: usize,
    ) {
        self.encode_with(
            &ComputeCommand::SetBindGroup {
                index,
                num_dynamic_offsets: offset_length.try_into().unwrap(),
                bind_group_id,
            },
            slice::from_raw_parts(offsets, offset_length),
        );

        for offset in slice::from_raw_parts(offsets, offset_length) {
            self.data = offset.poke_into(self.data);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_standalone_compute_pass_set_pipeline(
        &mut self,
        pipeline_id: id::ComputePipelineId,
    ) {
        self.encode(&ComputeCommand::SetPipeline(pipeline_id));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_standalone_compute_pass_dispatch(
        &mut self,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) {
        self.encode(&ComputeCommand::Dispatch([groups_x, groups_y, groups_z]));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_standalone_compute_pass_dispatch_indirect(
        &mut self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        self.encode(&ComputeCommand::DispatchIndirect {
            buffer_id,
            offset,
        });
    }
}
