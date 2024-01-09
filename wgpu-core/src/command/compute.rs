use crate::device::DeviceError;
use crate::resource::Resource;
use crate::snatch::SnatchGuard;
use crate::{
    binding_model::{
        BindError, BindGroup, LateMinBufferBindingSizeMismatch, PushConstantUploadError,
    },
    command::{
        bind::Binder,
        end_pipeline_statistics_query,
        memory_init::{fixup_discarded_surfaces, SurfacesInDiscardState},
        BasePass, BasePassRef, BindGroupStateChange, CommandBuffer, CommandEncoderError,
        CommandEncoderStatus, MapPassErr, PassErrorScope, QueryUseError, StateChange,
    },
    device::{MissingDownlevelFlags, MissingFeatures},
    error::{ErrorFormatter, PrettyError},
    global::Global,
    hal_api::HalApi,
    hal_label, id,
    id::DeviceId,
    identity::GlobalIdentityHandlerFactory,
    init_tracker::MemoryInitKind,
    pipeline,
    resource::{self},
    storage::Storage,
    track::{Tracker, UsageConflict, UsageScope},
    validation::{check_buffer_usage, MissingBufferUsageError},
    Label,
};

use hal::CommandEncoder as _;
#[cfg(any(feature = "serial-pass", feature = "replay"))]
use serde::Deserialize;
#[cfg(any(feature = "serial-pass", feature = "trace"))]
use serde::Serialize;

use thiserror::Error;

use std::{fmt, mem, str};

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
        index: u32,
        num_dynamic_offsets: usize,
        bind_group_id: id::BindGroupId,
    },
    SetPipeline(id::ComputePipelineId),

    /// Set a range of push constants to values stored in [`BasePass::push_constant_data`].
    SetPushConstant {
        /// The byte offset within the push constant storage to write to. This
        /// must be a multiple of four.
        offset: u32,

        /// The number of bytes to write. This must be a multiple of four.
        size_bytes: u32,

        /// Index in [`BasePass::push_constant_data`] of the start of the data
        /// to be written.
        ///
        /// Note: this is not a byte offset like `offset`. Rather, it is the
        /// index of the first `u32` element in `push_constant_data` to read.
        values_offset: u32,
    },

    Dispatch([u32; 3]),
    DispatchIndirect {
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
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
    WriteTimestamp {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },
    BeginPipelineStatisticsQuery {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },
    EndPipelineStatisticsQuery,
}

#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
pub struct ComputePass {
    base: BasePass<ComputeCommand>,
    parent_id: id::CommandEncoderId,
    timestamp_writes: Option<ComputePassTimestampWrites>,

    // Resource binding dedupe state.
    #[cfg_attr(feature = "serial-pass", serde(skip))]
    current_bind_groups: BindGroupStateChange,
    #[cfg_attr(feature = "serial-pass", serde(skip))]
    current_pipeline: StateChange<id::ComputePipelineId>,
}

impl ComputePass {
    pub fn new(parent_id: id::CommandEncoderId, desc: &ComputePassDescriptor) -> Self {
        Self {
            base: BasePass::new(&desc.label),
            parent_id,
            timestamp_writes: desc.timestamp_writes.cloned(),

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    pub fn parent_id(&self) -> id::CommandEncoderId {
        self.parent_id
    }

    #[cfg(feature = "trace")]
    pub fn into_command(self) -> crate::device::trace::Command {
        crate::device::trace::Command::RunComputePass {
            base: self.base,
            timestamp_writes: self.timestamp_writes,
        }
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

/// Describes the writing of timestamp values in a compute pass.
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(any(feature = "serial-pass", feature = "trace"), derive(Serialize))]
#[cfg_attr(any(feature = "serial-pass", feature = "replay"), derive(Deserialize))]
pub struct ComputePassTimestampWrites {
    /// The query set to write the timestamps to.
    pub query_set: id::QuerySetId,
    /// The index of the query set at which a start timestamp of this pass is written, if any.
    pub beginning_of_pass_write_index: Option<u32>,
    /// The index of the query set at which an end timestamp of this pass is written, if any.
    pub end_of_pass_write_index: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor<'a> {
    pub label: Label<'a>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<&'a ComputePassTimestampWrites>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum DispatchError {
    #[error("Compute pipeline must be set")]
    MissingPipeline,
    #[error("Incompatible bind group at index {index} in the current compute pipeline")]
    IncompatibleBindGroup { index: u32, diff: Vec<String> },
    #[error(
        "Each current dispatch group size dimension ({current:?}) must be less or equal to {limit}"
    )]
    InvalidGroupSize { current: [u32; 3], limit: u32 },
    #[error(transparent)]
    BindingSizeTooSmall(#[from] LateMinBufferBindingSizeMismatch),
}

/// Error encountered when performing a compute pass.
#[derive(Clone, Debug, Error)]
pub enum ComputePassErrorInner {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),
    #[error("Bind group at index {0:?} is invalid")]
    InvalidBindGroup(usize),
    #[error("Device {0:?} is invalid")]
    InvalidDevice(DeviceId),
    #[error("Bind group index {index} is greater than the device's requested `max_bind_group` limit {max}")]
    BindGroupIndexOutOfRange { index: u32, max: u32 },
    #[error("Compute pipeline {0:?} is invalid")]
    InvalidPipeline(id::ComputePipelineId),
    #[error("QuerySet {0:?} is invalid")]
    InvalidQuerySet(id::QuerySetId),
    #[error("Indirect buffer {0:?} is invalid or destroyed")]
    InvalidIndirectBuffer(id::BufferId),
    #[error("Indirect buffer uses bytes {offset}..{end_offset} which overruns indirect buffer of size {buffer_size}")]
    IndirectBufferOverrun {
        offset: u64,
        end_offset: u64,
        buffer_size: u64,
    },
    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(id::BufferId),
    #[error(transparent)]
    ResourceUsageConflict(#[from] UsageConflict),
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error("Cannot pop debug group, because number of pushed debug groups is zero")]
    InvalidPopDebugGroup,
    #[error(transparent)]
    Dispatch(#[from] DispatchError),
    #[error(transparent)]
    Bind(#[from] BindError),
    #[error(transparent)]
    PushConstants(#[from] PushConstantUploadError),
    #[error(transparent)]
    QueryUse(#[from] QueryUseError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
}

impl PrettyError for ComputePassErrorInner {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
        match *self {
            Self::InvalidPipeline(id) => {
                fmt.compute_pipeline_label(&id);
            }
            Self::InvalidIndirectBuffer(id) => {
                fmt.buffer_label(&id);
            }
            Self::Dispatch(DispatchError::IncompatibleBindGroup { ref diff, .. }) => {
                for d in diff {
                    fmt.note(&d);
                }
            }
            _ => {}
        };
    }
}

/// Error encountered when performing a compute pass.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct ComputePassError {
    pub scope: PassErrorScope,
    #[source]
    inner: ComputePassErrorInner,
}
impl PrettyError for ComputePassError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        // This error is wrapper for the inner error,
        // but the scope has useful labels
        fmt.error(self);
        self.scope.fmt_pretty(fmt);
    }
}

impl<T, E> MapPassErr<T, ComputePassError> for Result<T, E>
where
    E: Into<ComputePassErrorInner>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> Result<T, ComputePassError> {
        self.map_err(|inner| ComputePassError {
            scope,
            inner: inner.into(),
        })
    }
}

struct State<A: HalApi> {
    binder: Binder<A>,
    pipeline: Option<id::ComputePipelineId>,
    scope: UsageScope<A>,
    debug_scope_depth: u32,
}

impl<A: HalApi> State<A> {
    fn is_ready(&self) -> Result<(), DispatchError> {
        let bind_mask = self.binder.invalid_mask();
        if bind_mask != 0 {
            //let (expected, provided) = self.binder.entries[index as usize].info();
            let index = bind_mask.trailing_zeros();

            return Err(DispatchError::IncompatibleBindGroup {
                index,
                diff: self.binder.bgl_diff(),
            });
        }
        if self.pipeline.is_none() {
            return Err(DispatchError::MissingPipeline);
        }
        self.binder.check_late_buffer_bindings()?;

        Ok(())
    }

    // `extra_buffer` is there to represent the indirect buffer that is also
    // part of the usage scope.
    fn flush_states(
        &mut self,
        raw_encoder: &mut A::CommandEncoder,
        base_trackers: &mut Tracker<A>,
        bind_group_guard: &Storage<BindGroup<A>, id::BindGroupId>,
        indirect_buffer: Option<id::BufferId>,
        snatch_guard: &SnatchGuard,
    ) -> Result<(), UsageConflict> {
        for id in self.binder.list_active() {
            unsafe { self.scope.merge_bind_group(&bind_group_guard[id].used)? };
            // Note: stateless trackers are not merged: the lifetime reference
            // is held to the bind group itself.
        }

        for id in self.binder.list_active() {
            unsafe {
                base_trackers.set_and_remove_from_usage_scope_sparse(
                    &mut self.scope,
                    &bind_group_guard[id].used,
                )
            }
        }

        // Add the state of the indirect buffer if it hasn't been hit before.
        unsafe {
            base_trackers
                .buffers
                .set_and_remove_from_usage_scope_sparse(&mut self.scope.buffers, indirect_buffer);
        }

        log::trace!("Encoding dispatch barriers");

        CommandBuffer::drain_barriers(raw_encoder, base_trackers, snatch_guard);
        Ok(())
    }
}

// Common routines between render/compute

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_run_compute_pass<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
        pass: &ComputePass,
    ) -> Result<(), ComputePassError> {
        self.command_encoder_run_compute_pass_impl::<A>(
            encoder_id,
            pass.base.as_ref(),
            pass.timestamp_writes.as_ref(),
        )
    }

    #[doc(hidden)]
    pub fn command_encoder_run_compute_pass_impl<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
        base: BasePassRef<ComputeCommand>,
        timestamp_writes: Option<&ComputePassTimestampWrites>,
    ) -> Result<(), ComputePassError> {
        profiling::scope!("CommandEncoder::run_compute_pass");
        let pass_scope = PassErrorScope::Pass(encoder_id);

        let hub = A::hub(self);

        let cmd_buf = CommandBuffer::get_encoder(hub, encoder_id).map_pass_err(pass_scope)?;
        let device = &cmd_buf.device;
        if !device.is_valid() {
            return Err(ComputePassErrorInner::InvalidDevice(
                cmd_buf.device.as_info().id(),
            ))
            .map_pass_err(pass_scope);
        }

        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(crate::device::trace::Command::RunComputePass {
                base: BasePass::from_ref(base),
                timestamp_writes: timestamp_writes.cloned(),
            });
        }

        let encoder = &mut cmd_buf_data.encoder;
        let status = &mut cmd_buf_data.status;
        let tracker = &mut cmd_buf_data.trackers;
        let buffer_memory_init_actions = &mut cmd_buf_data.buffer_memory_init_actions;
        let texture_memory_actions = &mut cmd_buf_data.texture_memory_actions;

        // We automatically keep extending command buffers over time, and because
        // we want to insert a command buffer _before_ what we're about to record,
        // we need to make sure to close the previous one.
        encoder.close().map_pass_err(pass_scope)?;
        // will be reset to true if recording is done without errors
        *status = CommandEncoderStatus::Error;
        let raw = encoder.open().map_pass_err(pass_scope)?;

        let bind_group_guard = hub.bind_groups.read();
        let pipeline_guard = hub.compute_pipelines.read();
        let query_set_guard = hub.query_sets.read();
        let buffer_guard = hub.buffers.read();
        let texture_guard = hub.textures.read();

        let mut state = State {
            binder: Binder::new(),
            pipeline: None,
            scope: UsageScope::new(&*buffer_guard, &*texture_guard),
            debug_scope_depth: 0,
        };
        let mut temp_offsets = Vec::new();
        let mut dynamic_offset_count = 0;
        let mut string_offset = 0;
        let mut active_query = None;

        let timestamp_writes = if let Some(tw) = timestamp_writes {
            let query_set: &resource::QuerySet<A> = tracker
                .query_sets
                .add_single(&*query_set_guard, tw.query_set)
                .ok_or(ComputePassErrorInner::InvalidQuerySet(tw.query_set))
                .map_pass_err(pass_scope)?;

            // Unlike in render passes we can't delay resetting the query sets since
            // there is no auxillary pass.
            let range = if let (Some(index_a), Some(index_b)) =
                (tw.beginning_of_pass_write_index, tw.end_of_pass_write_index)
            {
                Some(index_a.min(index_b)..index_a.max(index_b) + 1)
            } else {
                tw.beginning_of_pass_write_index
                    .or(tw.end_of_pass_write_index)
                    .map(|i| i..i + 1)
            };
            // Range should always be Some, both values being None should lead to a validation error.
            // But no point in erroring over that nuance here!
            if let Some(range) = range {
                unsafe {
                    raw.reset_queries(query_set.raw.as_ref().unwrap(), range);
                }
            }

            Some(hal::ComputePassTimestampWrites {
                query_set: query_set.raw.as_ref().unwrap(),
                beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                end_of_pass_write_index: tw.end_of_pass_write_index,
            })
        } else {
            None
        };

        let snatch_guard = device.snatchable_lock.read();

        tracker.set_size(
            Some(&*buffer_guard),
            Some(&*texture_guard),
            None,
            None,
            Some(&*bind_group_guard),
            Some(&*pipeline_guard),
            None,
            None,
            Some(&*query_set_guard),
        );

        let discard_hal_labels = self
            .instance
            .flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS);
        let hal_desc = hal::ComputePassDescriptor {
            label: hal_label(base.label, self.instance.flags),
            timestamp_writes,
        };

        unsafe {
            raw.begin_compute_pass(&hal_desc);
        }

        let mut intermediate_trackers = Tracker::<A>::new();

        // Immediate texture inits required because of prior discards. Need to
        // be inserted before texture reads.
        let mut pending_discard_init_fixups = SurfacesInDiscardState::new();

        for command in base.commands {
            match *command {
                ComputeCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let scope = PassErrorScope::SetBindGroup(bind_group_id);

                    let max_bind_groups = cmd_buf.limits.max_bind_groups;
                    if index >= max_bind_groups {
                        return Err(ComputePassErrorInner::BindGroupIndexOutOfRange {
                            index,
                            max: max_bind_groups,
                        })
                        .map_pass_err(scope);
                    }

                    temp_offsets.clear();
                    temp_offsets.extend_from_slice(
                        &base.dynamic_offsets
                            [dynamic_offset_count..dynamic_offset_count + num_dynamic_offsets],
                    );
                    dynamic_offset_count += num_dynamic_offsets;

                    let bind_group = tracker
                        .bind_groups
                        .add_single(&*bind_group_guard, bind_group_id)
                        .ok_or(ComputePassErrorInner::InvalidBindGroup(index as usize))
                        .map_pass_err(scope)?;
                    bind_group
                        .validate_dynamic_bindings(index, &temp_offsets, &cmd_buf.limits)
                        .map_pass_err(scope)?;

                    buffer_memory_init_actions.extend(
                        bind_group.used_buffer_ranges.iter().filter_map(|action| {
                            action
                                .buffer
                                .initialization_status
                                .read()
                                .check_action(action)
                        }),
                    );

                    for action in bind_group.used_texture_ranges.iter() {
                        pending_discard_init_fixups
                            .extend(texture_memory_actions.register_init_action(action));
                    }

                    let pipeline_layout = state.binder.pipeline_layout.clone();
                    let entries =
                        state
                            .binder
                            .assign_group(index as usize, bind_group, &temp_offsets);
                    if !entries.is_empty() && pipeline_layout.is_some() {
                        let pipeline_layout = pipeline_layout.as_ref().unwrap().raw();
                        for (i, e) in entries.iter().enumerate() {
                            if let Some(group) = e.group.as_ref() {
                                let raw_bg = group
                                    .raw(&snatch_guard)
                                    .ok_or(ComputePassErrorInner::InvalidBindGroup(i))
                                    .map_pass_err(scope)?;
                                unsafe {
                                    raw.set_bind_group(
                                        pipeline_layout,
                                        index + i as u32,
                                        raw_bg,
                                        &e.dynamic_offsets,
                                    );
                                }
                            }
                        }
                    }
                }
                ComputeCommand::SetPipeline(pipeline_id) => {
                    let scope = PassErrorScope::SetPipelineCompute(pipeline_id);

                    state.pipeline = Some(pipeline_id);

                    let pipeline: &pipeline::ComputePipeline<A> = tracker
                        .compute_pipelines
                        .add_single(&*pipeline_guard, pipeline_id)
                        .ok_or(ComputePassErrorInner::InvalidPipeline(pipeline_id))
                        .map_pass_err(scope)?;

                    unsafe {
                        raw.set_compute_pipeline(pipeline.raw());
                    }

                    // Rebind resources
                    if state.binder.pipeline_layout.is_none()
                        || !state
                            .binder
                            .pipeline_layout
                            .as_ref()
                            .unwrap()
                            .is_equal(&pipeline.layout)
                    {
                        let (start_index, entries) = state.binder.change_pipeline_layout(
                            &pipeline.layout,
                            &pipeline.late_sized_buffer_groups,
                        );
                        if !entries.is_empty() {
                            for (i, e) in entries.iter().enumerate() {
                                if let Some(group) = e.group.as_ref() {
                                    let raw_bg = group
                                        .raw(&snatch_guard)
                                        .ok_or(ComputePassErrorInner::InvalidBindGroup(i))
                                        .map_pass_err(scope)?;
                                    unsafe {
                                        raw.set_bind_group(
                                            pipeline.layout.raw(),
                                            start_index as u32 + i as u32,
                                            raw_bg,
                                            &e.dynamic_offsets,
                                        );
                                    }
                                }
                            }
                        }

                        // Clear push constant ranges
                        let non_overlapping = super::bind::compute_nonoverlapping_ranges(
                            &pipeline.layout.push_constant_ranges,
                        );
                        for range in non_overlapping {
                            let offset = range.range.start;
                            let size_bytes = range.range.end - offset;
                            super::push_constant_clear(
                                offset,
                                size_bytes,
                                |clear_offset, clear_data| unsafe {
                                    raw.set_push_constants(
                                        pipeline.layout.raw(),
                                        wgt::ShaderStages::COMPUTE,
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
                    let scope = PassErrorScope::SetPushConstant;

                    let end_offset_bytes = offset + size_bytes;
                    let values_end_offset =
                        (values_offset + size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
                    let data_slice =
                        &base.push_constant_data[(values_offset as usize)..values_end_offset];

                    let pipeline_layout = state
                        .binder
                        .pipeline_layout
                        .as_ref()
                        //TODO: don't error here, lazily update the push constants
                        .ok_or(ComputePassErrorInner::Dispatch(
                            DispatchError::MissingPipeline,
                        ))
                        .map_pass_err(scope)?;

                    pipeline_layout
                        .validate_push_constant_ranges(
                            wgt::ShaderStages::COMPUTE,
                            offset,
                            end_offset_bytes,
                        )
                        .map_pass_err(scope)?;

                    unsafe {
                        raw.set_push_constants(
                            pipeline_layout.raw(),
                            wgt::ShaderStages::COMPUTE,
                            offset,
                            data_slice,
                        );
                    }
                }
                ComputeCommand::Dispatch(groups) => {
                    let scope = PassErrorScope::Dispatch {
                        indirect: false,
                        pipeline: state.pipeline,
                    };
                    state.is_ready().map_pass_err(scope)?;

                    state
                        .flush_states(
                            raw,
                            &mut intermediate_trackers,
                            &*bind_group_guard,
                            None,
                            &snatch_guard,
                        )
                        .map_pass_err(scope)?;

                    let groups_size_limit = cmd_buf.limits.max_compute_workgroups_per_dimension;

                    if groups[0] > groups_size_limit
                        || groups[1] > groups_size_limit
                        || groups[2] > groups_size_limit
                    {
                        return Err(ComputePassErrorInner::Dispatch(
                            DispatchError::InvalidGroupSize {
                                current: groups,
                                limit: groups_size_limit,
                            },
                        ))
                        .map_pass_err(scope);
                    }

                    unsafe {
                        raw.dispatch(groups);
                    }
                }
                ComputeCommand::DispatchIndirect { buffer_id, offset } => {
                    let scope = PassErrorScope::Dispatch {
                        indirect: true,
                        pipeline: state.pipeline,
                    };

                    state.is_ready().map_pass_err(scope)?;

                    device
                        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)
                        .map_pass_err(scope)?;

                    let indirect_buffer = state
                        .scope
                        .buffers
                        .merge_single(&*buffer_guard, buffer_id, hal::BufferUses::INDIRECT)
                        .map_pass_err(scope)?;
                    check_buffer_usage(indirect_buffer.usage, wgt::BufferUsages::INDIRECT)
                        .map_pass_err(scope)?;

                    let end_offset = offset + mem::size_of::<wgt::DispatchIndirectArgs>() as u64;
                    if end_offset > indirect_buffer.size {
                        return Err(ComputePassErrorInner::IndirectBufferOverrun {
                            offset,
                            end_offset,
                            buffer_size: indirect_buffer.size,
                        })
                        .map_pass_err(scope);
                    }

                    let buf_raw = indirect_buffer
                        .raw
                        .get(&snatch_guard)
                        .ok_or(ComputePassErrorInner::InvalidIndirectBuffer(buffer_id))
                        .map_pass_err(scope)?;

                    let stride = 3 * 4; // 3 integers, x/y/z group size

                    buffer_memory_init_actions.extend(
                        indirect_buffer.initialization_status.read().create_action(
                            indirect_buffer,
                            offset..(offset + stride),
                            MemoryInitKind::NeedsInitializedMemory,
                        ),
                    );

                    state
                        .flush_states(
                            raw,
                            &mut intermediate_trackers,
                            &*bind_group_guard,
                            Some(buffer_id),
                            &snatch_guard,
                        )
                        .map_pass_err(scope)?;
                    unsafe {
                        raw.dispatch_indirect(buf_raw, offset);
                    }
                }
                ComputeCommand::PushDebugGroup { color: _, len } => {
                    state.debug_scope_depth += 1;
                    if !discard_hal_labels {
                        let label =
                            str::from_utf8(&base.string_data[string_offset..string_offset + len])
                                .unwrap();
                        unsafe {
                            raw.begin_debug_marker(label);
                        }
                    }
                    string_offset += len;
                }
                ComputeCommand::PopDebugGroup => {
                    let scope = PassErrorScope::PopDebugGroup;

                    if state.debug_scope_depth == 0 {
                        return Err(ComputePassErrorInner::InvalidPopDebugGroup)
                            .map_pass_err(scope);
                    }
                    state.debug_scope_depth -= 1;
                    if !discard_hal_labels {
                        unsafe {
                            raw.end_debug_marker();
                        }
                    }
                }
                ComputeCommand::InsertDebugMarker { color: _, len } => {
                    if !discard_hal_labels {
                        let label =
                            str::from_utf8(&base.string_data[string_offset..string_offset + len])
                                .unwrap();
                        unsafe { raw.insert_debug_marker(label) }
                    }
                    string_offset += len;
                }
                ComputeCommand::WriteTimestamp {
                    query_set_id,
                    query_index,
                } => {
                    let scope = PassErrorScope::WriteTimestamp;

                    device
                        .require_features(wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
                        .map_pass_err(scope)?;

                    let query_set: &resource::QuerySet<A> = tracker
                        .query_sets
                        .add_single(&*query_set_guard, query_set_id)
                        .ok_or(ComputePassErrorInner::InvalidQuerySet(query_set_id))
                        .map_pass_err(scope)?;

                    query_set
                        .validate_and_write_timestamp(raw, query_set_id, query_index, None)
                        .map_pass_err(scope)?;
                }
                ComputeCommand::BeginPipelineStatisticsQuery {
                    query_set_id,
                    query_index,
                } => {
                    let scope = PassErrorScope::BeginPipelineStatisticsQuery;

                    let query_set: &resource::QuerySet<A> = tracker
                        .query_sets
                        .add_single(&*query_set_guard, query_set_id)
                        .ok_or(ComputePassErrorInner::InvalidQuerySet(query_set_id))
                        .map_pass_err(scope)?;

                    query_set
                        .validate_and_begin_pipeline_statistics_query(
                            raw,
                            query_set_id,
                            query_index,
                            None,
                            &mut active_query,
                        )
                        .map_pass_err(scope)?;
                }
                ComputeCommand::EndPipelineStatisticsQuery => {
                    let scope = PassErrorScope::EndPipelineStatisticsQuery;

                    end_pipeline_statistics_query(raw, &*query_set_guard, &mut active_query)
                        .map_pass_err(scope)?;
                }
            }
        }

        unsafe {
            raw.end_compute_pass();
        }

        // We've successfully recorded the compute pass, bring the
        // command buffer out of the error state.
        *status = CommandEncoderStatus::Recording;

        // Stop the current command buffer.
        encoder.close().map_pass_err(pass_scope)?;

        // Create a new command buffer, which we will insert _before_ the body of the compute pass.
        //
        // Use that buffer to insert barriers and clear discarded images.
        let transit = encoder.open().map_pass_err(pass_scope)?;
        fixup_discarded_surfaces(
            pending_discard_init_fixups.into_iter(),
            transit,
            &mut tracker.textures,
            device,
        );
        CommandBuffer::insert_barriers_from_tracker(
            transit,
            tracker,
            &intermediate_trackers,
            &snatch_guard,
        );
        // Close the command buffer, and swap it with the previous.
        encoder.close_and_swap().map_pass_err(pass_scope)?;

        Ok(())
    }
}

pub mod compute_ffi {
    use super::{ComputeCommand, ComputePass};
    use crate::{id, RawString};
    use std::{convert::TryInto, ffi, slice};
    use wgt::{BufferAddress, DynamicOffset};

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `offset_length` elements.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_set_bind_group(
        pass: &mut ComputePass,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const DynamicOffset,
        offset_length: usize,
    ) {
        let redundant = unsafe {
            pass.current_bind_groups.set_and_check_redundant(
                bind_group_id,
                index,
                &mut pass.base.dynamic_offsets,
                offsets,
                offset_length,
            )
        };

        if redundant {
            return;
        }

        pass.base.commands.push(ComputeCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offset_length,
            bind_group_id,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_set_pipeline(
        pass: &mut ComputePass,
        pipeline_id: id::ComputePipelineId,
    ) {
        if pass.current_pipeline.set_and_check_redundant(pipeline_id) {
            return;
        }

        pass.base
            .commands
            .push(ComputeCommand::SetPipeline(pipeline_id));
    }

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `size_bytes` bytes.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_set_push_constant(
        pass: &mut ComputePass,
        offset: u32,
        size_bytes: u32,
        data: *const u8,
    ) {
        assert_eq!(
            offset & (wgt::PUSH_CONSTANT_ALIGNMENT - 1),
            0,
            "Push constant offset must be aligned to 4 bytes."
        );
        assert_eq!(
            size_bytes & (wgt::PUSH_CONSTANT_ALIGNMENT - 1),
            0,
            "Push constant size must be aligned to 4 bytes."
        );
        let data_slice = unsafe { slice::from_raw_parts(data, size_bytes as usize) };
        let value_offset = pass.base.push_constant_data.len().try_into().expect(
            "Ran out of push constant space. Don't set 4gb of push constants per ComputePass.",
        );

        pass.base.push_constant_data.extend(
            data_slice
                .chunks_exact(wgt::PUSH_CONSTANT_ALIGNMENT as usize)
                .map(|arr| u32::from_ne_bytes([arr[0], arr[1], arr[2], arr[3]])),
        );

        pass.base.commands.push(ComputeCommand::SetPushConstant {
            offset,
            size_bytes,
            values_offset: value_offset,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_dispatch_workgroups(
        pass: &mut ComputePass,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) {
        pass.base
            .commands
            .push(ComputeCommand::Dispatch([groups_x, groups_y, groups_z]));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_dispatch_workgroups_indirect(
        pass: &mut ComputePass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        pass.base
            .commands
            .push(ComputeCommand::DispatchIndirect { buffer_id, offset });
    }

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given `label`
    /// is a valid null-terminated string.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_push_debug_group(
        pass: &mut ComputePass,
        label: RawString,
        color: u32,
    ) {
        let bytes = unsafe { ffi::CStr::from_ptr(label) }.to_bytes();
        pass.base.string_data.extend_from_slice(bytes);

        pass.base.commands.push(ComputeCommand::PushDebugGroup {
            color,
            len: bytes.len(),
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_pop_debug_group(pass: &mut ComputePass) {
        pass.base.commands.push(ComputeCommand::PopDebugGroup);
    }

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given `label`
    /// is a valid null-terminated string.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_compute_pass_insert_debug_marker(
        pass: &mut ComputePass,
        label: RawString,
        color: u32,
    ) {
        let bytes = unsafe { ffi::CStr::from_ptr(label) }.to_bytes();
        pass.base.string_data.extend_from_slice(bytes);

        pass.base.commands.push(ComputeCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_write_timestamp(
        pass: &mut ComputePass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) {
        pass.base.commands.push(ComputeCommand::WriteTimestamp {
            query_set_id,
            query_index,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_begin_pipeline_statistics_query(
        pass: &mut ComputePass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) {
        pass.base
            .commands
            .push(ComputeCommand::BeginPipelineStatisticsQuery {
                query_set_id,
                query_index,
            });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_compute_pass_end_pipeline_statistics_query(pass: &mut ComputePass) {
        pass.base
            .commands
            .push(ComputeCommand::EndPipelineStatisticsQuery);
    }
}
