/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    command::{
        bind::{Binder, LayoutChange},
        CommandBuffer, PhantomSlice,
    },
    device::all_buffer_stages,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id,
    resource::BufferUse,
};

use hal::command::CommandBuffer as _;
use peek_poke::{Peek, PeekPoke, Poke};
use wgt::{BufferAddress, BufferUsage, DynamicOffset, BIND_BUFFER_ALIGNMENT};

use std::{iter, str};

#[derive(Debug, PartialEq)]
enum PipelineState {
    Required,
    Set,
}

#[derive(Clone, Copy, Debug, PeekPoke)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum ComputeCommand {
    SetBindGroup {
        index: u8,
        num_dynamic_offsets: u8,
        bind_group_id: id::BindGroupId,
        #[cfg_attr(any(feature = "trace", feature = "replay"), serde(skip))]
        phantom_offsets: PhantomSlice<DynamicOffset>,
    },
    SetPipeline(id::ComputePipelineId),
    Dispatch([u32; 3]),
    DispatchIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
    },
    PushDebugGroup {
        color: u32,
        len: usize,
        #[cfg_attr(any(feature = "trace", feature = "replay"), serde(skip))]
        phantom_marker: PhantomSlice<u8>,
    },
    PopDebugGroup,
    InsertDebugMarker {
        color: u32,
        len: usize,
        #[cfg_attr(any(feature = "trace", feature = "replay"), serde(skip))]
        phantom_marker: PhantomSlice<u8>,
    },
    End,
}

#[derive(Debug)]
struct State {
    binder: Binder,
    debug_scope_depth: u32,
}

impl Default for ComputeCommand {
    fn default() -> Self {
        ComputeCommand::End
    }
}

impl super::RawPass<id::CommandEncoderId> {
    pub unsafe fn new_compute(parent: id::CommandEncoderId) -> Self {
        Self::new::<ComputeCommand>(parent)
    }

    pub unsafe fn fill_compute_commands(
        &mut self,
        commands: &[ComputeCommand],
        mut offsets: &[DynamicOffset],
    ) {
        for com in commands {
            self.encode(com);
            if let ComputeCommand::SetBindGroup {
                num_dynamic_offsets,
                ..
            } = *com
            {
                self.encode_slice(&offsets[..num_dynamic_offsets as usize]);
                offsets = &offsets[num_dynamic_offsets as usize..];
            }
        }
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

        let (_, mut token) = hub.render_bundles.read(&mut token);
        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
        let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
        let (pipeline_guard, mut token) = hub.compute_pipelines.read(&mut token);
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);

        let mut pipeline_state = PipelineState::Required;

        let mut state = State {
            binder: Binder::new(cmb.limits.max_bind_groups),
            debug_scope_depth: 0,
        };

        let mut peeker = raw_data.as_ptr();
        let raw_data_end = unsafe { raw_data.as_ptr().add(raw_data.len()) };
        let mut command = ComputeCommand::Dispatch([0; 3]); // dummy
        loop {
            assert!(unsafe { peeker.add(ComputeCommand::max_size()) } <= raw_data_end);
            peeker = unsafe { ComputeCommand::peek_from(peeker, &mut command) };
            match command {
                ComputeCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                    phantom_offsets,
                } => {
                    let (new_peeker, offsets) = unsafe {
                        phantom_offsets.decode_unaligned(
                            peeker,
                            num_dynamic_offsets as usize,
                            raw_data_end,
                        )
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

                    if let Some((pipeline_layout_id, follow_ups)) = state.binder.provide_entry(
                        index as usize,
                        bind_group_id,
                        bind_group,
                        offsets,
                    ) {
                        let bind_groups = iter::once(bind_group.raw.raw()).chain(
                            follow_ups
                                .clone()
                                .map(|(bg_id, _)| bind_group_guard[bg_id].raw.raw()),
                        );
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
                    let pipeline = cmb
                        .trackers
                        .compute_pipes
                        .use_extend(&*pipeline_guard, pipeline_id, (), ())
                        .unwrap();

                    unsafe {
                        raw.bind_compute_pipeline(&pipeline.raw);
                    }

                    // Rebind resources
                    if state.binder.pipeline_layout_id != Some(pipeline.layout_id.value) {
                        let pipeline_layout = &pipeline_layout_guard[pipeline.layout_id.value];
                        state.binder.pipeline_layout_id = Some(pipeline.layout_id.value);
                        state
                            .binder
                            .reset_expectations(pipeline_layout.bind_group_layout_ids.len());
                        let mut is_compatible = true;

                        for (index, (entry, bgl_id)) in state
                            .binder
                            .entries
                            .iter_mut()
                            .zip(&pipeline_layout.bind_group_layout_ids)
                            .enumerate()
                        {
                            match entry.expect_layout(bgl_id.value) {
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
                    assert_eq!(
                        pipeline_state,
                        PipelineState::Set,
                        "Dispatch error: Pipeline is missing"
                    );
                    unsafe {
                        raw.dispatch(groups);
                    }
                }
                ComputeCommand::DispatchIndirect { buffer_id, offset } => {
                    assert_eq!(
                        pipeline_state,
                        PipelineState::Set,
                        "Dispatch error: Pipeline is missing"
                    );
                    let (src_buffer, src_pending) = cmb.trackers.buffers.use_replace(
                        &*buffer_guard,
                        buffer_id,
                        (),
                        BufferUse::INDIRECT,
                    );
                    assert!(src_buffer.usage.contains(BufferUsage::INDIRECT));

                    let barriers = src_pending.map(|pending| pending.into_hal(src_buffer));

                    unsafe {
                        raw.pipeline_barrier(
                            all_buffer_stages()..all_buffer_stages(),
                            hal::memory::Dependencies::empty(),
                            barriers,
                        );
                        raw.dispatch_indirect(&src_buffer.raw, offset);
                    }
                }
                ComputeCommand::PushDebugGroup {
                    color,
                    len,
                    phantom_marker,
                } => unsafe {
                    state.debug_scope_depth += 1;

                    let (new_peeker, label) =
                        { phantom_marker.decode_unaligned(peeker, len, raw_data_end) };
                    peeker = new_peeker;

                    raw.begin_debug_marker(str::from_utf8(label).unwrap(), color)
                },
                ComputeCommand::PopDebugGroup => unsafe {
                    assert_ne!(
                        state.debug_scope_depth, 0,
                        "Can't pop debug group, because number of pushed debug groups is zero!"
                    );
                    state.debug_scope_depth -= 1;

                    raw.end_debug_marker()
                },
                ComputeCommand::InsertDebugMarker {
                    color,
                    len,
                    phantom_marker,
                } => unsafe {
                    let (new_peeker, label) =
                        { phantom_marker.decode_unaligned(peeker, len, raw_data_end) };
                    peeker = new_peeker;

                    raw.insert_debug_marker(str::from_utf8(label).unwrap(), color)
                },
                ComputeCommand::End => break,
            }
        }

        #[cfg(feature = "trace")]
        match cmb.commands {
            Some(ref mut list) => {
                let mut pass_commands = Vec::new();
                let mut pass_dynamic_offsets = Vec::new();
                peeker = raw_data.as_ptr();
                loop {
                    peeker = unsafe { ComputeCommand::peek_from(peeker, &mut command) };
                    match command {
                        ComputeCommand::SetBindGroup {
                            num_dynamic_offsets,
                            phantom_offsets,
                            ..
                        } => {
                            let (new_peeker, offsets) = unsafe {
                                phantom_offsets.decode_unaligned(
                                    peeker,
                                    num_dynamic_offsets as usize,
                                    raw_data_end,
                                )
                            };
                            peeker = new_peeker;
                            pass_dynamic_offsets.extend_from_slice(offsets);
                        }
                        ComputeCommand::End => break,
                        _ => {}
                    }
                    pass_commands.push(command);
                }
                list.push(crate::device::trace::Command::RunComputePass {
                    commands: pass_commands,
                    dynamic_offsets: pass_dynamic_offsets,
                });
            }
            None => {}
        }
    }
}

pub mod compute_ffi {
    use super::{super::PhantomSlice, ComputeCommand};
    use crate::{id, RawString};
    use std::{convert::TryInto, ffi, slice};
    use wgt::{BufferAddress, DynamicOffset};

    type RawPass = super::super::RawPass<id::CommandEncoderId>;

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
        pass.encode_slice(slice::from_raw_parts(offsets, offset_length));
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
        pass.encode(&ComputeCommand::DispatchIndirect { buffer_id, offset });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_push_debug_group(
        pass: &mut RawPass,
        label: RawString,
        color: u32,
    ) {
        let bytes = ffi::CStr::from_ptr(label).to_bytes();

        pass.encode(&ComputeCommand::PushDebugGroup {
            color,
            len: bytes.len(),
            phantom_marker: PhantomSlice::default(),
        });
        pass.encode_slice(bytes);
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_pop_debug_group(pass: &mut RawPass) {
        pass.encode(&ComputeCommand::PopDebugGroup);
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_insert_debug_marker(
        pass: &mut RawPass,
        label: RawString,
        color: u32,
    ) {
        let bytes = ffi::CStr::from_ptr(label).to_bytes();

        pass.encode(&ComputeCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
            phantom_marker: PhantomSlice::default(),
        });
        pass.encode_slice(bytes);
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
