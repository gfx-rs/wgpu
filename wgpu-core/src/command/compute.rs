/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    command::{
        bind::{Binder, LayoutChange},
        CommandBuffer,
        PhantomSlice,
    },
    device::all_buffer_stages,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id,
};

use wgt::{BufferAddress, BufferUsage, DynamicOffset, BIND_BUFFER_ALIGNMENT};
use hal::command::CommandBuffer as _;
use peek_poke::{Peek, PeekPoke, Poke};

use std::iter;

#[derive(Debug, PartialEq)]
enum PipelineState {
    Required,
    Set,
}

#[derive(Clone, Copy, Debug, PeekPoke)]
enum ComputeCommand {
    SetBindGroup {
        index: u8,
        num_dynamic_offsets: u8,
        bind_group_id: id::BindGroupId,
        phantom_offsets: PhantomSlice<DynamicOffset>,
    },
    SetPipeline(id::ComputePipelineId),
    Dispatch([u32; 3]),
    DispatchIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
    },
    End,
}

impl Default for ComputeCommand {
    fn default() -> Self {
        ComputeCommand::End
    }
}

impl super::RawPass {
    pub unsafe fn new_compute(parent: id::CommandEncoderId) -> Self {
        Self::from_vec(Vec::<ComputeCommand>::with_capacity(1), parent)
    }

    pub unsafe fn finish_compute(mut self) -> (Vec<u8>, id::CommandEncoderId) {
        self.finish(ComputeCommand::End);
        self.into_vec()
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor {
    pub todo: u32,
}

// Common routines between render/compute

impl<G: GlobalIdentityHandlerFactory> Global<G> {
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

        let mut pipeline_state = PipelineState::Required;

        let mut peeker = raw_data.as_ptr();
        let raw_data_end = unsafe {
            raw_data.as_ptr().add(raw_data.len())
        };
        let mut command = ComputeCommand::Dispatch([0; 3]); // dummy
        loop {
            assert!(unsafe { peeker.add(ComputeCommand::max_size()) } <= raw_data_end);
            peeker = unsafe { ComputeCommand::peek_from(peeker, &mut command) };
            match command {
                ComputeCommand::SetBindGroup { index, num_dynamic_offsets, bind_group_id, phantom_offsets } => {
                    let (new_peeker, offsets) = unsafe {
                        phantom_offsets.decode_unaligned(peeker, num_dynamic_offsets as usize, raw_data_end)
                    };
                    peeker = new_peeker;

                    if cfg!(debug_assertions) {
                        for off in offsets {
                            assert_eq!(
                                *off as BufferAddress % BIND_BUFFER_ALIGNMENT,
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
                                    .cloned(),
                            );
                        }
                    }
                }
                ComputeCommand::SetPipeline(pipeline_id) => {
                    pipeline_state = PipelineState::Set;
                    let pipeline = cmb.trackers
                        .compute_pipes
                        .use_extend(&*pipeline_guard, pipeline_id, (), ())
                        .unwrap();

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
                                            offsets.iter().cloned(),
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
                    assert_eq!(pipeline_state, PipelineState::Set, "Dispatch error: Pipeline is missing");
                    unsafe {
                        raw.dispatch(groups);
                    }
                }
                ComputeCommand::DispatchIndirect { buffer_id, offset } => {
                    assert_eq!(pipeline_state, PipelineState::Set, "Dispatch error: Pipeline is missing");
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
                ComputeCommand::End => break,
            }
        }
    }
}

pub mod compute_ffi {
    use super::{
        ComputeCommand,
        super::{PhantomSlice, RawPass},
    };
    use crate::{
        id,
        RawString,
    };
use wgt::{BufferAddress, DynamicOffset};
    use std::{convert::TryInto, slice};

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `offset_length` elements.
    // TODO: There might be other safety issues, such as using the unsafe
    // `RawPass::encode` and `RawPass::encode_slice`.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_set_bind_group(
        pass: &mut RawPass,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const DynamicOffset,
        offset_length: usize,
    ) {
        pass.encode(&ComputeCommand::SetBindGroup {
            index: index.try_into().unwrap(),
            num_dynamic_offsets: offset_length.try_into().unwrap(),
            bind_group_id,
            phantom_offsets: PhantomSlice::default(),
        });
        pass.encode_slice(
            slice::from_raw_parts(offsets, offset_length),
        );
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_set_pipeline(
        pass: &mut RawPass,
        pipeline_id: id::ComputePipelineId,
    ) {
        pass.encode(&ComputeCommand::SetPipeline(pipeline_id));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_dispatch(
        pass: &mut RawPass,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) {
        pass.encode(&ComputeCommand::Dispatch([groups_x, groups_y, groups_z]));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_dispatch_indirect(
        pass: &mut RawPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        pass.encode(&ComputeCommand::DispatchIndirect {
            buffer_id,
            offset,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_push_debug_group(
        _pass: &mut RawPass,
        _label: RawString,
    ) {
        //TODO
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_pop_debug_group(
        _pass: &mut RawPass,
    ) {
        //TODO
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_insert_debug_marker(
        _pass: &mut RawPass,
        _label: RawString,
    ) {
        //TODO
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_finish(
        pass: &mut RawPass,
        length: &mut usize,
    ) -> *const u8 {
        pass.finish(ComputeCommand::End);
        *length = pass.size();
        pass.base
    }
}
