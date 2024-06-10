use crate::{
    binding_model::{BindError, LateMinBufferBindingSizeMismatch, PushConstantUploadError},
    command::{
        bind::Binder,
        compute_command::{ArcComputeCommand, ComputeCommand},
        end_pipeline_statistics_query,
        memory_init::{fixup_discarded_surfaces, SurfacesInDiscardState},
        validate_and_begin_pipeline_statistics_query, BasePass, BindGroupStateChange,
        CommandBuffer, CommandEncoderError, CommandEncoderStatus, MapPassErr, PassErrorScope,
        QueryUseError, StateChange,
    },
    device::{DeviceError, MissingDownlevelFlags, MissingFeatures},
    error::{ErrorFormatter, PrettyError},
    global::Global,
    hal_api::HalApi,
    hal_label, id,
    init_tracker::MemoryInitKind,
    resource::{self, Resource},
    snatch::SnatchGuard,
    track::{Tracker, TrackerIndex, UsageConflict, UsageScope},
    validation::{check_buffer_usage, MissingBufferUsageError},
    Label,
};

use hal::CommandEncoder as _;
#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")]
use serde::Serialize;

use thiserror::Error;
use wgt::{BufferAddress, DynamicOffset};

use std::sync::Arc;
use std::{fmt, mem, str};

use super::DynComputePass;

pub struct ComputePass<A: HalApi> {
    /// All pass data & records is stored here.
    ///
    /// If this is `None`, the pass is in the 'ended' state and can no longer be used.
    /// Any attempt to record more commands will result in a validation error.
    base: Option<BasePass<ArcComputeCommand<A>>>,

    /// Parent command buffer that this pass records commands into.
    ///
    /// If it is none, this pass is invalid and any operation on it will return an error.
    parent: Option<Arc<CommandBuffer<A>>>,

    timestamp_writes: Option<ArcComputePassTimestampWrites<A>>,

    // Resource binding dedupe state.
    current_bind_groups: BindGroupStateChange,
    current_pipeline: StateChange<id::ComputePipelineId>,

    /// The device that this pass is associated with.
    ///
    /// Used for quick validation during recording.
    device_id: id::DeviceId,
}

impl<A: HalApi> ComputePass<A> {
    /// If the parent command buffer is invalid, the returned pass will be invalid.
    fn new(parent: Option<Arc<CommandBuffer<A>>>, desc: ArcComputePassDescriptor<A>) -> Self {
        let ArcComputePassDescriptor {
            label,
            timestamp_writes,
        } = desc;

        let device_id = parent
            .as_ref()
            .map_or(id::DeviceId::dummy(0), |p| p.device.as_info().id());

        Self {
            base: Some(BasePass::new(label)),
            parent,
            timestamp_writes,

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),

            device_id,
        }
    }

    #[inline]
    pub fn parent_id(&self) -> Option<id::CommandBufferId> {
        self.parent.as_ref().map(|cmd_buf| cmd_buf.as_info().id())
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.base.as_ref().and_then(|base| base.label.as_deref())
    }

    fn base_mut<'a>(
        &'a mut self,
        scope: PassErrorScope,
    ) -> Result<&'a mut BasePass<ArcComputeCommand<A>>, ComputePassError> {
        self.base
            .as_mut()
            .ok_or(ComputePassErrorInner::PassEnded)
            .map_pass_err(scope)
    }
}

impl<A: HalApi> fmt::Debug for ComputePass<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComputePass {{ parent: {:?} }}", self.parent_id())
    }
}

/// Describes the writing of timestamp values in a compute pass.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ComputePassTimestampWrites {
    /// The query set to write the timestamps to.
    pub query_set: id::QuerySetId,
    /// The index of the query set at which a start timestamp of this pass is written, if any.
    pub beginning_of_pass_write_index: Option<u32>,
    /// The index of the query set at which an end timestamp of this pass is written, if any.
    pub end_of_pass_write_index: Option<u32>,
}

/// Describes the writing of timestamp values in a compute pass with the query set resolved.
struct ArcComputePassTimestampWrites<A: HalApi> {
    /// The query set to write the timestamps to.
    pub query_set: Arc<resource::QuerySet<A>>,
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

struct ArcComputePassDescriptor<'a, A: HalApi> {
    pub label: &'a Label<'a>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<ArcComputePassTimestampWrites<A>>,
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
    #[error("Parent encoder is invalid")]
    InvalidParentEncoder,
    #[error("Bind group at index {0:?} is invalid")]
    InvalidBindGroup(u32),
    #[error("Device {0:?} is invalid")]
    InvalidDevice(id::DeviceId),
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
    #[error("Push constant offset must be aligned to 4 bytes")]
    PushConstantOffsetAlignment,
    #[error("Push constant size must be aligned to 4 bytes")]
    PushConstantSizeAlignment,
    #[error("Ran out of push constant space. Don't set 4gb of push constants per ComputePass.")]
    PushConstantOutOfMemory,
    #[error(transparent)]
    QueryUse(#[from] QueryUseError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("The compute pass has already been ended and no further commands can be recorded")]
    PassEnded,
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
    pub(super) inner: ComputePassErrorInner,
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

struct State<'a, A: HalApi> {
    binder: Binder<A>,
    pipeline: Option<id::ComputePipelineId>,
    scope: UsageScope<'a, A>,
    debug_scope_depth: u32,
}

impl<'a, A: HalApi> State<'a, A> {
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
        indirect_buffer: Option<TrackerIndex>,
        snatch_guard: &SnatchGuard,
    ) -> Result<(), UsageConflict> {
        for bind_group in self.binder.list_active() {
            unsafe { self.scope.merge_bind_group(&bind_group.used)? };
            // Note: stateless trackers are not merged: the lifetime reference
            // is held to the bind group itself.
        }

        for bind_group in self.binder.list_active() {
            unsafe {
                base_trackers
                    .set_and_remove_from_usage_scope_sparse(&mut self.scope, &bind_group.used)
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

// Running the compute pass.

impl Global {
    /// Creates a compute pass.
    ///
    /// If creation fails, an invalid pass is returned.
    /// Any operation on an invalid pass will return an error.
    ///
    /// If successful, puts the encoder into the [`CommandEncoderStatus::Locked`] state.
    pub fn command_encoder_create_compute_pass<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor<'_>,
    ) -> (ComputePass<A>, Option<CommandEncoderError>) {
        let hub = A::hub(self);

        let mut arc_desc = ArcComputePassDescriptor {
            label: &desc.label,
            timestamp_writes: None, // Handle only once we resolved the encoder.
        };

        match CommandBuffer::lock_encoder(hub, encoder_id) {
            Ok(cmd_buf) => {
                arc_desc.timestamp_writes = if let Some(tw) = desc.timestamp_writes {
                    let Ok(query_set) = hub.query_sets.read().get_owned(tw.query_set) else {
                        return (
                            ComputePass::new(None, arc_desc),
                            Some(CommandEncoderError::InvalidTimestampWritesQuerySetId),
                        );
                    };

                    if query_set.device.as_info().id() != cmd_buf.device.as_info().id() {
                        return (
                            ComputePass::new(None, arc_desc),
                            Some(CommandEncoderError::WrongDeviceForTimestampWritesQuerySet),
                        );
                    }

                    Some(ArcComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                        end_of_pass_write_index: tw.end_of_pass_write_index,
                    })
                } else {
                    None
                };

                (ComputePass::new(Some(cmd_buf), arc_desc), None)
            }
            Err(err) => (ComputePass::new(None, arc_desc), Some(err)),
        }
    }

    /// Creates a type erased compute pass.
    ///
    /// If creation fails, an invalid pass is returned.
    /// Any operation on an invalid pass will return an error.
    pub fn command_encoder_create_compute_pass_dyn<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor,
    ) -> (Box<dyn DynComputePass>, Option<CommandEncoderError>) {
        let (pass, err) = self.command_encoder_create_compute_pass::<A>(encoder_id, desc);
        (Box::new(pass), err)
    }

    pub fn compute_pass_end<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::Pass(pass.parent_id());
        let Some(parent) = pass.parent.as_ref() else {
            return Err(ComputePassErrorInner::InvalidParentEncoder).map_pass_err(scope);
        };

        parent.unlock_encoder().map_pass_err(scope)?;

        let base = pass
            .base
            .take()
            .ok_or(ComputePassErrorInner::PassEnded)
            .map_pass_err(scope)?;
        self.compute_pass_end_impl(parent, base, pass.timestamp_writes.take())
    }

    #[doc(hidden)]
    pub fn compute_pass_end_with_unresolved_commands<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
        base: BasePass<ComputeCommand>,
        timestamp_writes: Option<&ComputePassTimestampWrites>,
    ) -> Result<(), ComputePassError> {
        let hub = A::hub(self);
        let scope = PassErrorScope::PassEncoder(encoder_id);

        let cmd_buf = CommandBuffer::get_encoder(hub, encoder_id).map_pass_err(scope)?;
        let commands = ComputeCommand::resolve_compute_command_ids(A::hub(self), &base.commands)?;

        let timestamp_writes = if let Some(tw) = timestamp_writes {
            Some(ArcComputePassTimestampWrites {
                query_set: hub
                    .query_sets
                    .read()
                    .get_owned(tw.query_set)
                    .map_err(|_| ComputePassErrorInner::InvalidQuerySet(tw.query_set))
                    .map_pass_err(scope)?,
                beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                end_of_pass_write_index: tw.end_of_pass_write_index,
            })
        } else {
            None
        };

        self.compute_pass_end_impl::<A>(
            &cmd_buf,
            BasePass {
                label: base.label,
                commands,
                dynamic_offsets: base.dynamic_offsets,
                string_data: base.string_data,
                push_constant_data: base.push_constant_data,
            },
            timestamp_writes,
        )
    }

    fn compute_pass_end_impl<A: HalApi>(
        &self,
        cmd_buf: &CommandBuffer<A>,
        base: BasePass<ArcComputeCommand<A>>,
        mut timestamp_writes: Option<ArcComputePassTimestampWrites<A>>,
    ) -> Result<(), ComputePassError> {
        profiling::scope!("CommandEncoder::run_compute_pass");
        let pass_scope = PassErrorScope::Pass(Some(cmd_buf.as_info().id()));

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
                base: BasePass {
                    label: base.label.clone(),
                    commands: base.commands.iter().map(Into::into).collect(),
                    dynamic_offsets: base.dynamic_offsets.to_vec(),
                    string_data: base.string_data.to_vec(),
                    push_constant_data: base.push_constant_data.to_vec(),
                },
                timestamp_writes: timestamp_writes
                    .as_ref()
                    .map(|tw| ComputePassTimestampWrites {
                        query_set: tw.query_set.as_info().id(),
                        beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                        end_of_pass_write_index: tw.end_of_pass_write_index,
                    }),
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

        let mut state = State {
            binder: Binder::new(),
            pipeline: None,
            scope: device.new_usage_scope(),
            debug_scope_depth: 0,
        };
        let mut temp_offsets = Vec::new();
        let mut dynamic_offset_count = 0;
        let mut string_offset = 0;
        let mut active_query = None;

        let snatch_guard = device.snatchable_lock.read();

        let indices = &device.tracker_indices;
        tracker.buffers.set_size(indices.buffers.size());
        tracker.textures.set_size(indices.textures.size());
        tracker.bind_groups.set_size(indices.bind_groups.size());
        tracker
            .compute_pipelines
            .set_size(indices.compute_pipelines.size());
        tracker.query_sets.set_size(indices.query_sets.size());

        let timestamp_writes = if let Some(tw) = timestamp_writes.take() {
            let query_set = tracker.query_sets.insert_single(tw.query_set);

            // Unlike in render passes we can't delay resetting the query sets since
            // there is no auxiliary pass.
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

        let discard_hal_labels = self
            .instance
            .flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS);
        let hal_desc = hal::ComputePassDescriptor {
            label: hal_label(base.label.as_deref(), self.instance.flags),
            timestamp_writes,
        };

        unsafe {
            raw.begin_compute_pass(&hal_desc);
        }

        let mut intermediate_trackers = Tracker::<A>::new();

        // Immediate texture inits required because of prior discards. Need to
        // be inserted before texture reads.
        let mut pending_discard_init_fixups = SurfacesInDiscardState::new();

        // TODO: We should be draining the commands here, avoiding extra copies in the process.
        //       (A command encoder can't be executed twice!)
        for command in base.commands {
            match command {
                ArcComputeCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group,
                } => {
                    let scope = PassErrorScope::SetBindGroup(bind_group.as_info().id());

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

                    let bind_group = tracker.bind_groups.insert_single(bind_group);
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
                                    .ok_or(ComputePassErrorInner::InvalidBindGroup(i as u32))
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
                ArcComputeCommand::SetPipeline(pipeline) => {
                    let pipeline_id = pipeline.as_info().id();
                    let scope = PassErrorScope::SetPipelineCompute(pipeline_id);

                    state.pipeline = Some(pipeline_id);

                    let pipeline = tracker.compute_pipelines.insert_single(pipeline);

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
                                        .ok_or(ComputePassErrorInner::InvalidBindGroup(i as u32))
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
                ArcComputeCommand::SetPushConstant {
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
                ArcComputeCommand::Dispatch(groups) => {
                    let scope = PassErrorScope::Dispatch {
                        indirect: false,
                        pipeline: state.pipeline,
                    };
                    state.is_ready().map_pass_err(scope)?;

                    state
                        .flush_states(raw, &mut intermediate_trackers, None, &snatch_guard)
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
                ArcComputeCommand::DispatchIndirect { buffer, offset } => {
                    let buffer_id = buffer.as_info().id();
                    let scope = PassErrorScope::Dispatch {
                        indirect: true,
                        pipeline: state.pipeline,
                    };

                    state.is_ready().map_pass_err(scope)?;

                    device
                        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)
                        .map_pass_err(scope)?;

                    state
                        .scope
                        .buffers
                        .insert_merge_single(buffer.clone(), hal::BufferUses::INDIRECT)
                        .map_pass_err(scope)?;
                    check_buffer_usage(buffer_id, buffer.usage, wgt::BufferUsages::INDIRECT)
                        .map_pass_err(scope)?;

                    let end_offset = offset + mem::size_of::<wgt::DispatchIndirectArgs>() as u64;
                    if end_offset > buffer.size {
                        return Err(ComputePassErrorInner::IndirectBufferOverrun {
                            offset,
                            end_offset,
                            buffer_size: buffer.size,
                        })
                        .map_pass_err(scope);
                    }

                    let buf_raw = buffer
                        .raw
                        .get(&snatch_guard)
                        .ok_or(ComputePassErrorInner::InvalidIndirectBuffer(buffer_id))
                        .map_pass_err(scope)?;

                    let stride = 3 * 4; // 3 integers, x/y/z group size

                    buffer_memory_init_actions.extend(
                        buffer.initialization_status.read().create_action(
                            &buffer,
                            offset..(offset + stride),
                            MemoryInitKind::NeedsInitializedMemory,
                        ),
                    );

                    state
                        .flush_states(
                            raw,
                            &mut intermediate_trackers,
                            Some(buffer.as_info().tracker_index()),
                            &snatch_guard,
                        )
                        .map_pass_err(scope)?;
                    unsafe {
                        raw.dispatch_indirect(buf_raw, offset);
                    }
                }
                ArcComputeCommand::PushDebugGroup { color: _, len } => {
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
                ArcComputeCommand::PopDebugGroup => {
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
                ArcComputeCommand::InsertDebugMarker { color: _, len } => {
                    if !discard_hal_labels {
                        let label =
                            str::from_utf8(&base.string_data[string_offset..string_offset + len])
                                .unwrap();
                        unsafe { raw.insert_debug_marker(label) }
                    }
                    string_offset += len;
                }
                ArcComputeCommand::WriteTimestamp {
                    query_set,
                    query_index,
                } => {
                    let scope = PassErrorScope::WriteTimestamp;

                    device
                        .require_features(wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
                        .map_pass_err(scope)?;

                    let query_set = tracker.query_sets.insert_single(query_set);

                    query_set
                        .validate_and_write_timestamp(raw, query_index, None)
                        .map_pass_err(scope)?;
                }
                ArcComputeCommand::BeginPipelineStatisticsQuery {
                    query_set,
                    query_index,
                } => {
                    let scope = PassErrorScope::BeginPipelineStatisticsQuery;

                    let query_set = tracker.query_sets.insert_single(query_set);

                    validate_and_begin_pipeline_statistics_query(
                        query_set.clone(),
                        raw,
                        query_index,
                        None,
                        &mut active_query,
                    )
                    .map_pass_err(scope)?;
                }
                ArcComputeCommand::EndPipelineStatisticsQuery => {
                    let scope = PassErrorScope::EndPipelineStatisticsQuery;
                    end_pipeline_statistics_query(raw, &mut active_query).map_pass_err(scope)?;
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
            &snatch_guard,
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

// Recording a compute pass.
impl Global {
    pub fn compute_pass_set_bind_group<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: &[DynamicOffset],
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::SetBindGroup(bind_group_id);
        let base = pass
            .base
            .as_mut()
            .ok_or(ComputePassErrorInner::PassEnded)
            .map_pass_err(scope)?; // Can't use base_mut() utility here because of borrow checker.

        let redundant = pass.current_bind_groups.set_and_check_redundant(
            bind_group_id,
            index,
            &mut base.dynamic_offsets,
            offsets,
        );

        if redundant {
            return Ok(());
        }

        let hub = A::hub(self);
        let bind_group = hub
            .bind_groups
            .read()
            .get_owned(bind_group_id)
            .map_err(|_| ComputePassErrorInner::InvalidBindGroup(index))
            .map_pass_err(scope)?;

        if bind_group.device.as_info().id() != pass.device_id {
            return Err(DeviceError::WrongDevice).map_pass_err(scope);
        }

        base.commands.push(ArcComputeCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offsets.len(),
            bind_group,
        });

        Ok(())
    }

    pub fn compute_pass_set_pipeline<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), ComputePassError> {
        let redundant = pass.current_pipeline.set_and_check_redundant(pipeline_id);

        let scope = PassErrorScope::SetPipelineCompute(pipeline_id);

        let device_id = pass.device_id;
        let base = pass.base_mut(scope)?;
        if redundant {
            // Do redundant early-out **after** checking whether the pass is ended or not.
            return Ok(());
        }

        let hub = A::hub(self);
        let pipeline = hub
            .compute_pipelines
            .read()
            .get_owned(pipeline_id)
            .map_err(|_| ComputePassErrorInner::InvalidPipeline(pipeline_id))
            .map_pass_err(scope)?;

        if pipeline.device.as_info().id() != device_id {
            return Err(DeviceError::WrongDevice).map_pass_err(scope);
        }

        base.commands.push(ArcComputeCommand::SetPipeline(pipeline));

        Ok(())
    }

    pub fn compute_pass_set_push_constant<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        offset: u32,
        data: &[u8],
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::SetPushConstant;
        let base = pass.base_mut(scope)?;

        if offset & (wgt::PUSH_CONSTANT_ALIGNMENT - 1) != 0 {
            return Err(ComputePassErrorInner::PushConstantOffsetAlignment).map_pass_err(scope);
        }

        if data.len() as u32 & (wgt::PUSH_CONSTANT_ALIGNMENT - 1) != 0 {
            return Err(ComputePassErrorInner::PushConstantSizeAlignment).map_pass_err(scope);
        }
        let value_offset = base
            .push_constant_data
            .len()
            .try_into()
            .map_err(|_| ComputePassErrorInner::PushConstantOutOfMemory)
            .map_pass_err(scope)?;

        base.push_constant_data.extend(
            data.chunks_exact(wgt::PUSH_CONSTANT_ALIGNMENT as usize)
                .map(|arr| u32::from_ne_bytes([arr[0], arr[1], arr[2], arr[3]])),
        );

        base.commands.push(ArcComputeCommand::<A>::SetPushConstant {
            offset,
            size_bytes: data.len() as u32,
            values_offset: value_offset,
        });

        Ok(())
    }

    pub fn compute_pass_dispatch_workgroups<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::Dispatch {
            indirect: false,
            pipeline: pass.current_pipeline.last_state,
        };

        let base = pass.base_mut(scope)?;
        base.commands.push(ArcComputeCommand::<A>::Dispatch([
            groups_x, groups_y, groups_z,
        ]));

        Ok(())
    }

    pub fn compute_pass_dispatch_workgroups_indirect<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), ComputePassError> {
        let hub = A::hub(self);
        let scope = PassErrorScope::Dispatch {
            indirect: true,
            pipeline: pass.current_pipeline.last_state,
        };
        let device_id = pass.device_id;
        let base = pass.base_mut(scope)?;

        let buffer = hub
            .buffers
            .read()
            .get_owned(buffer_id)
            .map_err(|_| ComputePassErrorInner::InvalidBuffer(buffer_id))
            .map_pass_err(scope)?;

        if buffer.device.as_info().id() != device_id {
            return Err(DeviceError::WrongDevice).map_pass_err(scope);
        }

        base.commands
            .push(ArcComputeCommand::<A>::DispatchIndirect { buffer, offset });

        Ok(())
    }

    pub fn compute_pass_push_debug_group<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        label: &str,
        color: u32,
    ) -> Result<(), ComputePassError> {
        let base = pass.base_mut(PassErrorScope::PushDebugGroup)?;

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcComputeCommand::<A>::PushDebugGroup {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn compute_pass_pop_debug_group<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
    ) -> Result<(), ComputePassError> {
        let base = pass.base_mut(PassErrorScope::PopDebugGroup)?;

        base.commands.push(ArcComputeCommand::<A>::PopDebugGroup);

        Ok(())
    }

    pub fn compute_pass_insert_debug_marker<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        label: &str,
        color: u32,
    ) -> Result<(), ComputePassError> {
        let base = pass.base_mut(PassErrorScope::InsertDebugMarker)?;

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands
            .push(ArcComputeCommand::<A>::InsertDebugMarker {
                color,
                len: bytes.len(),
            });

        Ok(())
    }

    pub fn compute_pass_write_timestamp<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::WriteTimestamp;
        let device_id = pass.device_id;
        let base = pass.base_mut(scope)?;

        let hub = A::hub(self);
        let query_set = hub
            .query_sets
            .read()
            .get_owned(query_set_id)
            .map_err(|_| ComputePassErrorInner::InvalidQuerySet(query_set_id))
            .map_pass_err(scope)?;

        if query_set.device.as_info().id() != device_id {
            return Err(DeviceError::WrongDevice).map_pass_err(scope);
        }

        base.commands.push(ArcComputeCommand::WriteTimestamp {
            query_set,
            query_index,
        });

        Ok(())
    }

    pub fn compute_pass_begin_pipeline_statistics_query<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::BeginPipelineStatisticsQuery;
        let device_id = pass.device_id;
        let base = pass.base_mut(scope)?;

        let hub = A::hub(self);
        let query_set = hub
            .query_sets
            .read()
            .get_owned(query_set_id)
            .map_err(|_| ComputePassErrorInner::InvalidQuerySet(query_set_id))
            .map_pass_err(scope)?;

        if query_set.device.as_info().id() != device_id {
            return Err(DeviceError::WrongDevice).map_pass_err(scope);
        }

        base.commands
            .push(ArcComputeCommand::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            });

        Ok(())
    }

    pub fn compute_pass_end_pipeline_statistics_query<A: HalApi>(
        &self,
        pass: &mut ComputePass<A>,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::EndPipelineStatisticsQuery;
        let base = pass.base_mut(scope)?;
        base.commands
            .push(ArcComputeCommand::<A>::EndPipelineStatisticsQuery);

        Ok(())
    }
}
