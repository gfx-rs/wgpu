/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model::{BindError, PushConstantUploadError},
    command::{
        bind::{Binder, LayoutChange},
        BasePass, BasePassRef, CommandBuffer,
    },
    device::all_buffer_stages,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id,
    resource::BufferUse,
    span,
};

use hal::command::CommandBuffer as _;
use wgt::{BufferAddress, BufferUsage};

use std::{fmt, iter, str};

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "trace"),
    derive(serde::Serialize)
)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "replay"),
    derive(serde::Deserialize)
)]
pub enum ComputeCommand {
    SetBindGroup {
        index: u8,
        num_dynamic_offsets: u8,
        bind_group_id: id::BindGroupId,
    },
    SetPipeline(id::ComputePipelineId),
    SetPushConstant {
        offset: u32,
        size_bytes: u32,
        values_offset: u32,
    },
    Dispatch([u32; 3]),
    DispatchIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
    },
    PushDebugGroup {
        color: u32,
        len: usize,
    },
    PopDebugGroup,
    InsertDebugMarker {
        color: u32,
        len: usize,
    },
}

#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
pub struct ComputePass {
    base: BasePass<ComputeCommand>,
    parent_id: id::CommandEncoderId,
}

impl ComputePass {
    pub fn new(parent_id: id::CommandEncoderId) -> Self {
        ComputePass {
            base: BasePass::new(),
            parent_id,
        }
    }

    pub fn parent_id(&self) -> id::CommandEncoderId {
        self.parent_id
    }
}

impl fmt::Debug for ComputePass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComputePass {{ encoder_id: {:?}, data: {:?} commands and {:?} dynamic offsets }}",
            self.parent_id,
            self.base.commands.len(),
            self.base.dynamic_offsets.len()
        )
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor {
    pub todo: u32,
}

#[derive(Debug, PartialEq)]
enum PipelineState {
    Required,
    Set,
}

#[derive(Debug)]
struct State {
    binder: Binder,
    pipeline: PipelineState,
    debug_scope_depth: u32,
}

#[derive(Clone, Debug)]
pub enum ComputePassError {
    BindGroupIndexOutOfRange {
        index: u8,
        max: u32,
    },
    UnboundPipeline,
    InvalidBufferUsage {
        actual: BufferUsage,
        expected: BufferUsage,
    },
    InvalidPopDebugGroup,
    Bind(BindError),
    PushConstants(PushConstantUploadError),
}

impl From<PushConstantUploadError> for ComputePassError {
    fn from(error: PushConstantUploadError) -> Self {
        Self::PushConstants(error)
    }
}

impl From<BindError> for ComputePassError {
    fn from(error: BindError) -> Self {
        Self::Bind(error)
    }
}

impl fmt::Display for ComputePassError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BindGroupIndexOutOfRange { index, max } => write!(
                f,
                "bind group index {} is greater than the device's requested `max_bind_group` limit {}",
                index,
                max,
            ),
            Self::UnboundPipeline => write!(f, "a compute pipeline must be bound"),
            Self::InvalidBufferUsage { actual, expected } => write!(
                f,
                "buffer usage is {:?} which does not contain required usage {:?}",
                actual,
                expected,
            ),
            Self::InvalidPopDebugGroup => write!(f, "cannot pop debug group, because number of pushed debug groups is zero"),
            Self::Bind(error) => write!(f, "{}", error),
            Self::PushConstants(error) => write!(f, "{}", error),
        }
    }
}

// Common routines between render/compute

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_run_compute_pass<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        pass: &ComputePass,
    ) -> Result<(), ComputePassError> {
        self.command_encoder_run_compute_pass_impl::<B>(encoder_id, pass.base.as_ref())
    }

    #[doc(hidden)]
    pub fn command_encoder_run_compute_pass_impl<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        mut base: BasePassRef<ComputeCommand>,
    ) -> Result<(), ComputePassError> {
        span!(_guard, INFO, "CommandEncoder::run_compute_pass");
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[encoder_id];
        let raw = cmb.raw.last_mut().unwrap();

        #[cfg(feature = "trace")]
        match cmb.commands {
            Some(ref mut list) => {
                list.push(crate::device::trace::Command::RunComputePass {
                    base: BasePass::from_ref(base),
                });
            }
            None => {}
        }

        let (_, mut token) = hub.render_bundles.read(&mut token);
        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
        let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
        let (pipeline_guard, mut token) = hub.compute_pipelines.read(&mut token);
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);

        let mut state = State {
            binder: Binder::new(cmb.limits.max_bind_groups),
            pipeline: PipelineState::Required,
            debug_scope_depth: 0,
        };

        for command in base.commands {
            match *command {
                ComputeCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let max_bind_groups = cmb.limits.max_bind_groups;
                    if (index as u32) >= max_bind_groups {
                        return Err(ComputePassError::BindGroupIndexOutOfRange {
                            index,
                            max: max_bind_groups,
                        });
                    }

                    let offsets = &base.dynamic_offsets[..num_dynamic_offsets as usize];
                    base.dynamic_offsets = &base.dynamic_offsets[num_dynamic_offsets as usize..];

                    let bind_group = cmb
                        .trackers
                        .bind_groups
                        .use_extend(&*bind_group_guard, bind_group_id, (), ())
                        .unwrap();
                    bind_group.validate_dynamic_bindings(offsets)?;

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
                    state.pipeline = PipelineState::Set;
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

                        // Clear push constant ranges
                        let non_overlapping = super::bind::compute_nonoverlapping_ranges(
                            &pipeline_layout.push_constant_ranges,
                        );
                        for range in non_overlapping {
                            let offset = range.range.start;
                            let size_bytes = range.range.end - offset;
                            super::push_constant_clear(
                                offset,
                                size_bytes,
                                |clear_offset, clear_data| unsafe {
                                    raw.push_compute_constants(
                                        &pipeline_layout.raw,
                                        clear_offset,
                                        clear_data,
                                    );
                                },
                            );
                        }
                    }
                }
                ComputeCommand::SetPushConstant {
                    offset,
                    size_bytes,
                    values_offset,
                } => {
                    let end_offset_bytes = offset + size_bytes;
                    let values_end_offset = (values_offset + size_bytes / 4) as usize;
                    let data_slice =
                        &base.push_constant_data[(values_offset as usize)..values_end_offset];

                    let pipeline_layout_id = state
                        .binder
                        .pipeline_layout_id
                        .ok_or(ComputePassError::UnboundPipeline)?;
                    let pipeline_layout = &pipeline_layout_guard[pipeline_layout_id];

                    pipeline_layout
                        .validate_push_constant_ranges(
                            wgt::ShaderStage::COMPUTE,
                            offset,
                            end_offset_bytes,
                        )
                        .map_err(ComputePassError::from)?;

                    unsafe { raw.push_compute_constants(&pipeline_layout.raw, offset, data_slice) }
                }
                ComputeCommand::Dispatch(groups) => {
                    if state.pipeline != PipelineState::Set {
                        return Err(ComputePassError::UnboundPipeline);
                    }
                    unsafe {
                        raw.dispatch(groups);
                    }
                }
                ComputeCommand::DispatchIndirect { buffer_id, offset } => {
                    if state.pipeline != PipelineState::Set {
                        return Err(ComputePassError::UnboundPipeline);
                    }
                    let (src_buffer, src_pending) = cmb.trackers.buffers.use_replace(
                        &*buffer_guard,
                        buffer_id,
                        (),
                        BufferUse::INDIRECT,
                    );
                    if !src_buffer.usage.contains(BufferUsage::INDIRECT) {
                        return Err(ComputePassError::InvalidBufferUsage {
                            actual: src_buffer.usage,
                            expected: BufferUsage::INDIRECT,
                        });
                    }

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
                ComputeCommand::PushDebugGroup { color, len } => {
                    state.debug_scope_depth += 1;

                    let label = str::from_utf8(&base.string_data[..len]).unwrap();
                    unsafe {
                        raw.begin_debug_marker(label, color);
                    }
                    base.string_data = &base.string_data[len..];
                }
                ComputeCommand::PopDebugGroup => {
                    if state.debug_scope_depth == 0 {
                        return Err(ComputePassError::InvalidPopDebugGroup);
                    }
                    state.debug_scope_depth -= 1;
                    unsafe {
                        raw.end_debug_marker();
                    }
                }
                ComputeCommand::InsertDebugMarker { color, len } => {
                    let label = str::from_utf8(&base.string_data[..len]).unwrap();
                    unsafe { raw.insert_debug_marker(label, color) }
                    base.string_data = &base.string_data[len..];
                }
            }
        }

        Ok(())
    }
}

pub mod compute_ffi {
    use super::{ComputeCommand, ComputePass};
    use crate::{id, span, RawString};
    use std::{convert::TryInto, ffi, slice};
    use wgt::{BufferAddress, DynamicOffset};

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `offset_length` elements.
    // TODO: There might be other safety issues, such as using the unsafe
    // `RawPass::encode` and `RawPass::encode_slice`.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_set_bind_group(
        pass: &mut ComputePass,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const DynamicOffset,
        offset_length: usize,
    ) {
        span!(_guard, DEBUG, "ComputePass::set_bind_group");
        pass.base.commands.push(ComputeCommand::SetBindGroup {
            index: index.try_into().unwrap(),
            num_dynamic_offsets: offset_length.try_into().unwrap(),
            bind_group_id,
        });
        pass.base
            .dynamic_offsets
            .extend_from_slice(slice::from_raw_parts(offsets, offset_length));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_set_pipeline(
        pass: &mut ComputePass,
        pipeline_id: id::ComputePipelineId,
    ) {
        span!(_guard, DEBUG, "ComputePass::set_pipeline");
        pass.base
            .commands
            .push(ComputeCommand::SetPipeline(pipeline_id));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_set_push_constant(
        pass: &mut ComputePass,
        offset: u32,
        size_bytes: u32,
        data: *const u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_push_constant");
        let data_slice = slice::from_raw_parts(data, (size_bytes / 4) as usize);
        let value_offset = pass.base.push_constant_data.len().try_into().expect(
            "Ran out of push constant space. Don't set 4gb of push constants per ComputePass.",
        );
        pass.base.push_constant_data.extend_from_slice(data_slice);
        pass.base.commands.push(ComputeCommand::SetPushConstant {
            offset,
            size_bytes,
            values_offset: value_offset,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_dispatch(
        pass: &mut ComputePass,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) {
        span!(_guard, DEBUG, "ComputePass::dispatch");
        pass.base
            .commands
            .push(ComputeCommand::Dispatch([groups_x, groups_y, groups_z]));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_dispatch_indirect(
        pass: &mut ComputePass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        span!(_guard, DEBUG, "ComputePass::dispatch_indirect");
        pass.base
            .commands
            .push(ComputeCommand::DispatchIndirect { buffer_id, offset });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_push_debug_group(
        pass: &mut ComputePass,
        label: RawString,
        color: u32,
    ) {
        span!(_guard, DEBUG, "ComputePass::push_debug_group");
        let bytes = ffi::CStr::from_ptr(label).to_bytes();
        pass.base.string_data.extend_from_slice(bytes);

        pass.base.commands.push(ComputeCommand::PushDebugGroup {
            color,
            len: bytes.len(),
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_pop_debug_group(pass: &mut ComputePass) {
        span!(_guard, DEBUG, "ComputePass::pop_debug_group");
        pass.base.commands.push(ComputeCommand::PopDebugGroup);
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_insert_debug_marker(
        pass: &mut ComputePass,
        label: RawString,
        color: u32,
    ) {
        span!(_guard, DEBUG, "ComputePass::insert_debug_marker");
        let bytes = ffi::CStr::from_ptr(label).to_bytes();
        pass.base.string_data.extend_from_slice(bytes);

        pass.base.commands.push(ComputeCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
        });
    }
}
