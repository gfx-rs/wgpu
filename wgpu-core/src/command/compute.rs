use crate::{
    binding_model::{
        BindError, BindGroup, LateMinBufferBindingSizeMismatch, PushConstantUploadError,
    },
    command::{
        bind::Binder,
        compute_command::ArcComputeCommand,
        end_pipeline_statistics_query,
        memory_init::{fixup_discarded_surfaces, SurfacesInDiscardState},
        validate_and_begin_pipeline_statistics_query, ArcPassTimestampWrites, BasePass,
        BindGroupStateChange, CommandBuffer, CommandEncoderError, MapPassErr, PassErrorScope,
        PassTimestampWrites, QueryUseError, StateChange,
    },
    device::{Device, DeviceError, MissingDownlevelFlags, MissingFeatures},
    global::Global,
    hal_label, id,
    init_tracker::{BufferInitTrackerAction, MemoryInitKind},
    pipeline::ComputePipeline,
    resource::{
        self, Buffer, DestroyedResourceError, InvalidResourceError, Labeled,
        MissingBufferUsageError, ParentDevice,
    },
    snatch::SnatchGuard,
    track::{ResourceUsageCompatibilityError, Tracker, TrackerIndex, UsageScope},
    Label,
};

use thiserror::Error;
use wgt::{BufferAddress, DynamicOffset};

use super::{bind::BinderError, memory_init::CommandBufferTextureMemoryActions};
use crate::ray_tracing::TlasAction;
use std::{fmt, mem::size_of, str, sync::Arc};

pub struct ComputePass {
    /// All pass data & records is stored here.
    ///
    /// If this is `None`, the pass is in the 'ended' state and can no longer be used.
    /// Any attempt to record more commands will result in a validation error.
    base: Option<BasePass<ArcComputeCommand>>,

    /// Parent command buffer that this pass records commands into.
    ///
    /// If it is none, this pass is invalid and any operation on it will return an error.
    parent: Option<Arc<CommandBuffer>>,

    timestamp_writes: Option<ArcPassTimestampWrites>,

    // Resource binding dedupe state.
    current_bind_groups: BindGroupStateChange,
    current_pipeline: StateChange<id::ComputePipelineId>,
}

impl ComputePass {
    /// If the parent command buffer is invalid, the returned pass will be invalid.
    fn new(parent: Option<Arc<CommandBuffer>>, desc: ArcComputePassDescriptor) -> Self {
        let ArcComputePassDescriptor {
            label,
            timestamp_writes,
        } = desc;

        Self {
            base: Some(BasePass::new(label)),
            parent,
            timestamp_writes,

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.base.as_ref().and_then(|base| base.label.as_deref())
    }

    fn base_mut<'a>(
        &'a mut self,
        scope: PassErrorScope,
    ) -> Result<&'a mut BasePass<ArcComputeCommand>, ComputePassError> {
        self.base
            .as_mut()
            .ok_or(ComputePassErrorInner::PassEnded)
            .map_pass_err(scope)
    }
}

impl fmt::Debug for ComputePass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.parent {
            Some(ref cmd_buf) => write!(f, "ComputePass {{ parent: {} }}", cmd_buf.error_ident()),
            None => write!(f, "ComputePass {{ parent: None }}"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor<'a> {
    pub label: Label<'a>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<&'a PassTimestampWrites>,
}

struct ArcComputePassDescriptor<'a> {
    pub label: &'a Label<'a>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<ArcPassTimestampWrites>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DispatchError {
    #[error("Compute pipeline must be set")]
    MissingPipeline,
    #[error(transparent)]
    IncompatibleBindGroup(#[from] Box<BinderError>),
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
    #[error("Bind group index {index} is greater than the device's requested `max_bind_group` limit {max}")]
    BindGroupIndexOutOfRange { index: u32, max: u32 },
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error("Indirect buffer offset {0:?} is not a multiple of 4")]
    UnalignedIndirectBufferOffset(BufferAddress),
    #[error("Indirect buffer uses bytes {offset}..{end_offset} which overruns indirect buffer of size {buffer_size}")]
    IndirectBufferOverrun {
        offset: u64,
        end_offset: u64,
        buffer_size: u64,
    },
    #[error(transparent)]
    ResourceUsageCompatibility(#[from] ResourceUsageCompatibilityError),
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
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

/// Error encountered when performing a compute pass.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct ComputePassError {
    pub scope: PassErrorScope,
    #[source]
    pub(super) inner: ComputePassErrorInner,
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

struct State<'scope, 'snatch_guard, 'cmd_buf, 'raw_encoder> {
    binder: Binder,
    pipeline: Option<Arc<ComputePipeline>>,
    scope: UsageScope<'scope>,
    debug_scope_depth: u32,

    snatch_guard: SnatchGuard<'snatch_guard>,

    device: &'cmd_buf Arc<Device>,

    raw_encoder: &'raw_encoder mut dyn hal::DynCommandEncoder,

    tracker: &'cmd_buf mut Tracker,
    buffer_memory_init_actions: &'cmd_buf mut Vec<BufferInitTrackerAction>,
    texture_memory_actions: &'cmd_buf mut CommandBufferTextureMemoryActions,
    tlas_actions: &'cmd_buf mut Vec<TlasAction>,

    temp_offsets: Vec<u32>,
    dynamic_offset_count: usize,
    string_offset: usize,
    active_query: Option<(Arc<resource::QuerySet>, u32)>,

    push_constants: Vec<u32>,

    intermediate_trackers: Tracker,

    /// Immediate texture inits required because of prior discards. Need to
    /// be inserted before texture reads.
    pending_discard_init_fixups: SurfacesInDiscardState,
}

impl<'scope, 'snatch_guard, 'cmd_buf, 'raw_encoder>
    State<'scope, 'snatch_guard, 'cmd_buf, 'raw_encoder>
{
    fn is_ready(&self) -> Result<(), DispatchError> {
        if let Some(pipeline) = self.pipeline.as_ref() {
            self.binder.check_compatibility(pipeline.as_ref())?;
            self.binder.check_late_buffer_bindings()?;
            Ok(())
        } else {
            Err(DispatchError::MissingPipeline)
        }
    }

    // `extra_buffer` is there to represent the indirect buffer that is also
    // part of the usage scope.
    fn flush_states(
        &mut self,
        indirect_buffer: Option<TrackerIndex>,
    ) -> Result<(), ResourceUsageCompatibilityError> {
        for bind_group in self.binder.list_active() {
            unsafe { self.scope.merge_bind_group(&bind_group.used)? };
            // Note: stateless trackers are not merged: the lifetime reference
            // is held to the bind group itself.
        }

        for bind_group in self.binder.list_active() {
            unsafe {
                self.intermediate_trackers
                    .set_and_remove_from_usage_scope_sparse(&mut self.scope, &bind_group.used)
            }
        }

        // Add the state of the indirect buffer if it hasn't been hit before.
        unsafe {
            self.intermediate_trackers
                .buffers
                .set_and_remove_from_usage_scope_sparse(&mut self.scope.buffers, indirect_buffer);
        }

        CommandBuffer::drain_barriers(
            self.raw_encoder,
            &mut self.intermediate_trackers,
            &self.snatch_guard,
        );
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
    /// If successful, puts the encoder into the [`Locked`] state.
    ///
    /// [`Locked`]: crate::command::CommandEncoderStatus::Locked
    pub fn command_encoder_create_compute_pass(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor<'_>,
    ) -> (ComputePass, Option<CommandEncoderError>) {
        let hub = &self.hub;

        let mut arc_desc = ArcComputePassDescriptor {
            label: &desc.label,
            timestamp_writes: None, // Handle only once we resolved the encoder.
        };

        let make_err = |e, arc_desc| (ComputePass::new(None, arc_desc), Some(e));

        let cmd_buf = hub.command_buffers.get(encoder_id.into_command_buffer_id());

        match cmd_buf.data.lock().lock_encoder() {
            Ok(_) => {}
            Err(e) => return make_err(e, arc_desc),
        };

        arc_desc.timestamp_writes = match desc
            .timestamp_writes
            .map(|tw| {
                Self::validate_pass_timestamp_writes(&cmd_buf.device, &hub.query_sets.read(), tw)
            })
            .transpose()
        {
            Ok(ok) => ok,
            Err(e) => return make_err(e, arc_desc),
        };

        (ComputePass::new(Some(cmd_buf), arc_desc), None)
    }

    /// Note that this differs from [`Self::compute_pass_end`], it will
    /// create a new pass, replay the commands and end the pass.
    #[doc(hidden)]
    #[cfg(any(feature = "serde", feature = "replay"))]
    pub fn compute_pass_end_with_unresolved_commands(
        &self,
        encoder_id: id::CommandEncoderId,
        base: BasePass<super::ComputeCommand>,
        timestamp_writes: Option<&PassTimestampWrites>,
    ) -> Result<(), ComputePassError> {
        let pass_scope = PassErrorScope::Pass;

        #[cfg(feature = "trace")]
        {
            let cmd_buf = self
                .hub
                .command_buffers
                .get(encoder_id.into_command_buffer_id());
            let mut cmd_buf_data = cmd_buf.data.lock();
            let cmd_buf_data = cmd_buf_data.get_inner().map_pass_err(pass_scope)?;

            if let Some(ref mut list) = cmd_buf_data.commands {
                list.push(crate::device::trace::Command::RunComputePass {
                    base: BasePass {
                        label: base.label.clone(),
                        commands: base.commands.clone(),
                        dynamic_offsets: base.dynamic_offsets.clone(),
                        string_data: base.string_data.clone(),
                        push_constant_data: base.push_constant_data.clone(),
                    },
                    timestamp_writes: timestamp_writes.cloned(),
                });
            }
        }

        let BasePass {
            label,
            commands,
            dynamic_offsets,
            string_data,
            push_constant_data,
        } = base;

        let (mut compute_pass, encoder_error) = self.command_encoder_create_compute_pass(
            encoder_id,
            &ComputePassDescriptor {
                label: label.as_deref().map(std::borrow::Cow::Borrowed),
                timestamp_writes,
            },
        );
        if let Some(err) = encoder_error {
            return Err(ComputePassError {
                scope: pass_scope,
                inner: err.into(),
            });
        };

        compute_pass.base = Some(BasePass {
            label,
            commands: super::ComputeCommand::resolve_compute_command_ids(&self.hub, &commands)?,
            dynamic_offsets,
            string_data,
            push_constant_data,
        });

        self.compute_pass_end(&mut compute_pass)
    }

    pub fn compute_pass_end(&self, pass: &mut ComputePass) -> Result<(), ComputePassError> {
        profiling::scope!("CommandEncoder::run_compute_pass");
        let pass_scope = PassErrorScope::Pass;

        let cmd_buf = pass
            .parent
            .as_ref()
            .ok_or(ComputePassErrorInner::InvalidParentEncoder)
            .map_pass_err(pass_scope)?;

        let base = pass
            .base
            .take()
            .ok_or(ComputePassErrorInner::PassEnded)
            .map_pass_err(pass_scope)?;

        let device = &cmd_buf.device;
        device.check_is_valid().map_pass_err(pass_scope)?;

        let mut cmd_buf_data = cmd_buf.data.lock();
        let mut cmd_buf_data_guard = cmd_buf_data.unlock_encoder().map_pass_err(pass_scope)?;
        let cmd_buf_data = &mut *cmd_buf_data_guard;

        let encoder = &mut cmd_buf_data.encoder;

        // We automatically keep extending command buffers over time, and because
        // we want to insert a command buffer _before_ what we're about to record,
        // we need to make sure to close the previous one.
        encoder.close_if_open().map_pass_err(pass_scope)?;
        let raw_encoder = encoder
            .open_pass(base.label.as_deref())
            .map_pass_err(pass_scope)?;

        let mut state = State {
            binder: Binder::new(),
            pipeline: None,
            scope: device.new_usage_scope(),
            debug_scope_depth: 0,

            snatch_guard: device.snatchable_lock.read(),

            device,
            raw_encoder,
            tracker: &mut cmd_buf_data.trackers,
            buffer_memory_init_actions: &mut cmd_buf_data.buffer_memory_init_actions,
            texture_memory_actions: &mut cmd_buf_data.texture_memory_actions,
            tlas_actions: &mut cmd_buf_data.tlas_actions,

            temp_offsets: Vec::new(),
            dynamic_offset_count: 0,
            string_offset: 0,
            active_query: None,

            push_constants: Vec::new(),

            intermediate_trackers: Tracker::new(),

            pending_discard_init_fixups: SurfacesInDiscardState::new(),
        };

        let indices = &state.device.tracker_indices;
        state.tracker.buffers.set_size(indices.buffers.size());
        state.tracker.textures.set_size(indices.textures.size());

        let timestamp_writes: Option<hal::PassTimestampWrites<'_, dyn hal::DynQuerySet>> =
            if let Some(tw) = pass.timestamp_writes.take() {
                tw.query_set
                    .same_device_as(cmd_buf.as_ref())
                    .map_pass_err(pass_scope)?;

                let query_set = state.tracker.query_sets.insert_single(tw.query_set);

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
                        state.raw_encoder.reset_queries(query_set.raw(), range);
                    }
                }

                Some(hal::PassTimestampWrites {
                    query_set: query_set.raw(),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                })
            } else {
                None
            };

        let hal_desc = hal::ComputePassDescriptor {
            label: hal_label(base.label.as_deref(), device.instance_flags),
            timestamp_writes,
        };

        unsafe {
            state.raw_encoder.begin_compute_pass(&hal_desc);
        }

        for command in base.commands {
            match command {
                ArcComputeCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group,
                } => {
                    let scope = PassErrorScope::SetBindGroup;
                    set_bind_group(
                        &mut state,
                        cmd_buf,
                        &base.dynamic_offsets,
                        index,
                        num_dynamic_offsets,
                        bind_group,
                    )
                    .map_pass_err(scope)?;
                }
                ArcComputeCommand::SetPipeline(pipeline) => {
                    let scope = PassErrorScope::SetPipelineCompute;
                    set_pipeline(&mut state, cmd_buf, pipeline).map_pass_err(scope)?;
                }
                ArcComputeCommand::SetPushConstant {
                    offset,
                    size_bytes,
                    values_offset,
                } => {
                    let scope = PassErrorScope::SetPushConstant;
                    set_push_constant(
                        &mut state,
                        &base.push_constant_data,
                        offset,
                        size_bytes,
                        values_offset,
                    )
                    .map_pass_err(scope)?;
                }
                ArcComputeCommand::Dispatch(groups) => {
                    let scope = PassErrorScope::Dispatch { indirect: false };
                    dispatch(&mut state, groups).map_pass_err(scope)?;
                }
                ArcComputeCommand::DispatchIndirect { buffer, offset } => {
                    let scope = PassErrorScope::Dispatch { indirect: true };
                    dispatch_indirect(&mut state, cmd_buf, buffer, offset).map_pass_err(scope)?;
                }
                ArcComputeCommand::PushDebugGroup { color: _, len } => {
                    push_debug_group(&mut state, &base.string_data, len);
                }
                ArcComputeCommand::PopDebugGroup => {
                    let scope = PassErrorScope::PopDebugGroup;
                    pop_debug_group(&mut state).map_pass_err(scope)?;
                }
                ArcComputeCommand::InsertDebugMarker { color: _, len } => {
                    insert_debug_marker(&mut state, &base.string_data, len);
                }
                ArcComputeCommand::WriteTimestamp {
                    query_set,
                    query_index,
                } => {
                    let scope = PassErrorScope::WriteTimestamp;
                    write_timestamp(&mut state, cmd_buf, query_set, query_index)
                        .map_pass_err(scope)?;
                }
                ArcComputeCommand::BeginPipelineStatisticsQuery {
                    query_set,
                    query_index,
                } => {
                    let scope = PassErrorScope::BeginPipelineStatisticsQuery;
                    validate_and_begin_pipeline_statistics_query(
                        query_set,
                        state.raw_encoder,
                        &mut state.tracker.query_sets,
                        cmd_buf,
                        query_index,
                        None,
                        &mut state.active_query,
                    )
                    .map_pass_err(scope)?;
                }
                ArcComputeCommand::EndPipelineStatisticsQuery => {
                    let scope = PassErrorScope::EndPipelineStatisticsQuery;
                    end_pipeline_statistics_query(state.raw_encoder, &mut state.active_query)
                        .map_pass_err(scope)?;
                }
            }
        }

        unsafe {
            state.raw_encoder.end_compute_pass();
        }

        let State {
            snatch_guard,
            tracker,
            intermediate_trackers,
            pending_discard_init_fixups,
            ..
        } = state;

        // Stop the current command buffer.
        encoder.close().map_pass_err(pass_scope)?;

        // Create a new command buffer, which we will insert _before_ the body of the compute pass.
        //
        // Use that buffer to insert barriers and clear discarded images.
        let transit = encoder
            .open_pass(Some("(wgpu internal) Pre Pass"))
            .map_pass_err(pass_scope)?;
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
        cmd_buf_data_guard.mark_successful();

        Ok(())
    }
}

fn set_bind_group(
    state: &mut State,
    cmd_buf: &CommandBuffer,
    dynamic_offsets: &[DynamicOffset],
    index: u32,
    num_dynamic_offsets: usize,
    bind_group: Option<Arc<BindGroup>>,
) -> Result<(), ComputePassErrorInner> {
    let max_bind_groups = state.device.limits.max_bind_groups;
    if index >= max_bind_groups {
        return Err(ComputePassErrorInner::BindGroupIndexOutOfRange {
            index,
            max: max_bind_groups,
        });
    }

    state.temp_offsets.clear();
    state.temp_offsets.extend_from_slice(
        &dynamic_offsets
            [state.dynamic_offset_count..state.dynamic_offset_count + num_dynamic_offsets],
    );
    state.dynamic_offset_count += num_dynamic_offsets;

    if bind_group.is_none() {
        // TODO: Handle bind_group None.
        return Ok(());
    }

    let bind_group = bind_group.unwrap();
    let bind_group = state.tracker.bind_groups.insert_single(bind_group);

    bind_group.same_device_as(cmd_buf)?;

    bind_group.validate_dynamic_bindings(index, &state.temp_offsets)?;

    state
        .buffer_memory_init_actions
        .extend(bind_group.used_buffer_ranges.iter().filter_map(|action| {
            action
                .buffer
                .initialization_status
                .read()
                .check_action(action)
        }));

    for action in bind_group.used_texture_ranges.iter() {
        state
            .pending_discard_init_fixups
            .extend(state.texture_memory_actions.register_init_action(action));
    }

    let used_resource = bind_group
        .used
        .acceleration_structures
        .into_iter()
        .map(|tlas| TlasAction {
            tlas: tlas.clone(),
            kind: crate::ray_tracing::TlasActionKind::Use,
        });

    state.tlas_actions.extend(used_resource);

    let pipeline_layout = state.binder.pipeline_layout.clone();
    let entries = state
        .binder
        .assign_group(index as usize, bind_group, &state.temp_offsets);
    if !entries.is_empty() && pipeline_layout.is_some() {
        let pipeline_layout = pipeline_layout.as_ref().unwrap().raw();
        for (i, e) in entries.iter().enumerate() {
            if let Some(group) = e.group.as_ref() {
                let raw_bg = group.try_raw(&state.snatch_guard)?;
                unsafe {
                    state.raw_encoder.set_bind_group(
                        pipeline_layout,
                        index + i as u32,
                        Some(raw_bg),
                        &e.dynamic_offsets,
                    );
                }
            }
        }
    }
    Ok(())
}

fn set_pipeline(
    state: &mut State,
    cmd_buf: &CommandBuffer,
    pipeline: Arc<ComputePipeline>,
) -> Result<(), ComputePassErrorInner> {
    pipeline.same_device_as(cmd_buf)?;

    state.pipeline = Some(pipeline.clone());

    let pipeline = state.tracker.compute_pipelines.insert_single(pipeline);

    unsafe {
        state.raw_encoder.set_compute_pipeline(pipeline.raw());
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
        let (start_index, entries) = state
            .binder
            .change_pipeline_layout(&pipeline.layout, &pipeline.late_sized_buffer_groups);
        if !entries.is_empty() {
            for (i, e) in entries.iter().enumerate() {
                if let Some(group) = e.group.as_ref() {
                    let raw_bg = group.try_raw(&state.snatch_guard)?;
                    unsafe {
                        state.raw_encoder.set_bind_group(
                            pipeline.layout.raw(),
                            start_index as u32 + i as u32,
                            Some(raw_bg),
                            &e.dynamic_offsets,
                        );
                    }
                }
            }
        }

        // TODO: integrate this in the code below once we simplify push constants
        state.push_constants.clear();
        // Note that can only be one range for each stage. See the `MoreThanOnePushConstantRangePerStage` error.
        if let Some(push_constant_range) =
            pipeline.layout.push_constant_ranges.iter().find_map(|pcr| {
                pcr.stages
                    .contains(wgt::ShaderStages::COMPUTE)
                    .then_some(pcr.range.clone())
            })
        {
            // Note that non-0 range start doesn't work anyway https://github.com/gfx-rs/wgpu/issues/4502
            let len = push_constant_range.len() / wgt::PUSH_CONSTANT_ALIGNMENT as usize;
            state.push_constants.extend(core::iter::repeat(0).take(len));
        }

        // Clear push constant ranges
        let non_overlapping =
            super::bind::compute_nonoverlapping_ranges(&pipeline.layout.push_constant_ranges);
        for range in non_overlapping {
            let offset = range.range.start;
            let size_bytes = range.range.end - offset;
            super::push_constant_clear(offset, size_bytes, |clear_offset, clear_data| unsafe {
                state.raw_encoder.set_push_constants(
                    pipeline.layout.raw(),
                    wgt::ShaderStages::COMPUTE,
                    clear_offset,
                    clear_data,
                );
            });
        }
    }
    Ok(())
}

fn set_push_constant(
    state: &mut State,
    push_constant_data: &[u32],
    offset: u32,
    size_bytes: u32,
    values_offset: u32,
) -> Result<(), ComputePassErrorInner> {
    let end_offset_bytes = offset + size_bytes;
    let values_end_offset = (values_offset + size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
    let data_slice = &push_constant_data[(values_offset as usize)..values_end_offset];

    let pipeline_layout = state
        .binder
        .pipeline_layout
        .as_ref()
        // TODO: don't error here, lazily update the push constants using `state.push_constants`
        .ok_or(ComputePassErrorInner::Dispatch(
            DispatchError::MissingPipeline,
        ))?;

    pipeline_layout.validate_push_constant_ranges(
        wgt::ShaderStages::COMPUTE,
        offset,
        end_offset_bytes,
    )?;

    let offset_in_elements = (offset / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
    let size_in_elements = (size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
    state.push_constants[offset_in_elements..][..size_in_elements].copy_from_slice(data_slice);

    unsafe {
        state.raw_encoder.set_push_constants(
            pipeline_layout.raw(),
            wgt::ShaderStages::COMPUTE,
            offset,
            data_slice,
        );
    }
    Ok(())
}

fn dispatch(state: &mut State, groups: [u32; 3]) -> Result<(), ComputePassErrorInner> {
    state.is_ready()?;

    state.flush_states(None)?;

    let groups_size_limit = state.device.limits.max_compute_workgroups_per_dimension;

    if groups[0] > groups_size_limit
        || groups[1] > groups_size_limit
        || groups[2] > groups_size_limit
    {
        return Err(ComputePassErrorInner::Dispatch(
            DispatchError::InvalidGroupSize {
                current: groups,
                limit: groups_size_limit,
            },
        ));
    }

    unsafe {
        state.raw_encoder.dispatch(groups);
    }
    Ok(())
}

fn dispatch_indirect(
    state: &mut State,
    cmd_buf: &CommandBuffer,
    buffer: Arc<Buffer>,
    offset: u64,
) -> Result<(), ComputePassErrorInner> {
    buffer.same_device_as(cmd_buf)?;

    state.is_ready()?;

    state
        .device
        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)?;

    buffer.check_usage(wgt::BufferUsages::INDIRECT)?;

    if offset % 4 != 0 {
        return Err(ComputePassErrorInner::UnalignedIndirectBufferOffset(offset));
    }

    let end_offset = offset + size_of::<wgt::DispatchIndirectArgs>() as u64;
    if end_offset > buffer.size {
        return Err(ComputePassErrorInner::IndirectBufferOverrun {
            offset,
            end_offset,
            buffer_size: buffer.size,
        });
    }

    let stride = 3 * 4; // 3 integers, x/y/z group size
    state
        .buffer_memory_init_actions
        .extend(buffer.initialization_status.read().create_action(
            &buffer,
            offset..(offset + stride),
            MemoryInitKind::NeedsInitializedMemory,
        ));

    #[cfg(feature = "indirect-validation")]
    {
        let params = state.device.indirect_validation.as_ref().unwrap().params(
            &state.device.limits,
            offset,
            buffer.size,
        );

        unsafe {
            state.raw_encoder.set_compute_pipeline(params.pipeline);
        }

        unsafe {
            state.raw_encoder.set_push_constants(
                params.pipeline_layout,
                wgt::ShaderStages::COMPUTE,
                0,
                &[params.offset_remainder as u32 / 4],
            );
        }

        unsafe {
            state.raw_encoder.set_bind_group(
                params.pipeline_layout,
                0,
                Some(params.dst_bind_group),
                &[],
            );
        }
        unsafe {
            state.raw_encoder.set_bind_group(
                params.pipeline_layout,
                1,
                Some(
                    buffer
                        .raw_indirect_validation_bind_group
                        .get(&state.snatch_guard)
                        .unwrap()
                        .as_ref(),
                ),
                &[params.aligned_offset as u32],
            );
        }

        let src_transition = state
            .intermediate_trackers
            .buffers
            .set_single(&buffer, hal::BufferUses::STORAGE_READ_ONLY);
        let src_barrier =
            src_transition.map(|transition| transition.into_hal(&buffer, &state.snatch_guard));
        unsafe {
            state.raw_encoder.transition_buffers(src_barrier.as_slice());
        }

        unsafe {
            state.raw_encoder.transition_buffers(&[hal::BufferBarrier {
                buffer: params.dst_buffer,
                usage: hal::StateTransition {
                    from: hal::BufferUses::INDIRECT,
                    to: hal::BufferUses::STORAGE_READ_WRITE,
                },
            }]);
        }

        unsafe {
            state.raw_encoder.dispatch([1, 1, 1]);
        }

        // reset state
        {
            let pipeline = state.pipeline.as_ref().unwrap();

            unsafe {
                state.raw_encoder.set_compute_pipeline(pipeline.raw());
            }

            if !state.push_constants.is_empty() {
                unsafe {
                    state.raw_encoder.set_push_constants(
                        pipeline.layout.raw(),
                        wgt::ShaderStages::COMPUTE,
                        0,
                        &state.push_constants,
                    );
                }
            }

            for (i, e) in state.binder.list_valid() {
                let group = e.group.as_ref().unwrap();
                let raw_bg = group.try_raw(&state.snatch_guard)?;
                unsafe {
                    state.raw_encoder.set_bind_group(
                        pipeline.layout.raw(),
                        i as u32,
                        Some(raw_bg),
                        &e.dynamic_offsets,
                    );
                }
            }
        }

        unsafe {
            state.raw_encoder.transition_buffers(&[hal::BufferBarrier {
                buffer: params.dst_buffer,
                usage: hal::StateTransition {
                    from: hal::BufferUses::STORAGE_READ_WRITE,
                    to: hal::BufferUses::INDIRECT,
                },
            }]);
        }

        state.flush_states(None)?;
        unsafe {
            state.raw_encoder.dispatch_indirect(params.dst_buffer, 0);
        }
    };
    #[cfg(not(feature = "indirect-validation"))]
    {
        state
            .scope
            .buffers
            .merge_single(&buffer, hal::BufferUses::INDIRECT)?;

        use crate::resource::Trackable;
        state.flush_states(Some(buffer.tracker_index()))?;

        let buf_raw = buffer.try_raw(&state.snatch_guard)?;
        unsafe {
            state.raw_encoder.dispatch_indirect(buf_raw, offset);
        }
    }

    Ok(())
}

fn push_debug_group(state: &mut State, string_data: &[u8], len: usize) {
    state.debug_scope_depth += 1;
    if !state
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        let label =
            str::from_utf8(&string_data[state.string_offset..state.string_offset + len]).unwrap();
        unsafe {
            state.raw_encoder.begin_debug_marker(label);
        }
    }
    state.string_offset += len;
}

fn pop_debug_group(state: &mut State) -> Result<(), ComputePassErrorInner> {
    if state.debug_scope_depth == 0 {
        return Err(ComputePassErrorInner::InvalidPopDebugGroup);
    }
    state.debug_scope_depth -= 1;
    if !state
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        unsafe {
            state.raw_encoder.end_debug_marker();
        }
    }
    Ok(())
}

fn insert_debug_marker(state: &mut State, string_data: &[u8], len: usize) {
    if !state
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        let label =
            str::from_utf8(&string_data[state.string_offset..state.string_offset + len]).unwrap();
        unsafe { state.raw_encoder.insert_debug_marker(label) }
    }
    state.string_offset += len;
}

fn write_timestamp(
    state: &mut State,
    cmd_buf: &CommandBuffer,
    query_set: Arc<resource::QuerySet>,
    query_index: u32,
) -> Result<(), ComputePassErrorInner> {
    query_set.same_device_as(cmd_buf)?;

    state
        .device
        .require_features(wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES)?;

    let query_set = state.tracker.query_sets.insert_single(query_set);

    query_set.validate_and_write_timestamp(state.raw_encoder, query_index, None)?;
    Ok(())
}

// Recording a compute pass.
impl Global {
    pub fn compute_pass_set_bind_group(
        &self,
        pass: &mut ComputePass,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::SetBindGroup;
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

        let mut bind_group = None;
        if bind_group_id.is_some() {
            let bind_group_id = bind_group_id.unwrap();

            let hub = &self.hub;
            let bg = hub
                .bind_groups
                .get(bind_group_id)
                .get()
                .map_pass_err(scope)?;
            bind_group = Some(bg);
        }

        base.commands.push(ArcComputeCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offsets.len(),
            bind_group,
        });

        Ok(())
    }

    pub fn compute_pass_set_pipeline(
        &self,
        pass: &mut ComputePass,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), ComputePassError> {
        let redundant = pass.current_pipeline.set_and_check_redundant(pipeline_id);

        let scope = PassErrorScope::SetPipelineCompute;

        let base = pass.base_mut(scope)?;
        if redundant {
            // Do redundant early-out **after** checking whether the pass is ended or not.
            return Ok(());
        }

        let hub = &self.hub;
        let pipeline = hub
            .compute_pipelines
            .get(pipeline_id)
            .get()
            .map_pass_err(scope)?;

        base.commands.push(ArcComputeCommand::SetPipeline(pipeline));

        Ok(())
    }

    pub fn compute_pass_set_push_constants(
        &self,
        pass: &mut ComputePass,
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

        base.commands.push(ArcComputeCommand::SetPushConstant {
            offset,
            size_bytes: data.len() as u32,
            values_offset: value_offset,
        });

        Ok(())
    }

    pub fn compute_pass_dispatch_workgroups(
        &self,
        pass: &mut ComputePass,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::Dispatch { indirect: false };

        let base = pass.base_mut(scope)?;
        base.commands
            .push(ArcComputeCommand::Dispatch([groups_x, groups_y, groups_z]));

        Ok(())
    }

    pub fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut ComputePass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), ComputePassError> {
        let hub = &self.hub;
        let scope = PassErrorScope::Dispatch { indirect: true };
        let base = pass.base_mut(scope)?;

        let buffer = hub.buffers.get(buffer_id).get().map_pass_err(scope)?;

        base.commands
            .push(ArcComputeCommand::DispatchIndirect { buffer, offset });

        Ok(())
    }

    pub fn compute_pass_push_debug_group(
        &self,
        pass: &mut ComputePass,
        label: &str,
        color: u32,
    ) -> Result<(), ComputePassError> {
        let base = pass.base_mut(PassErrorScope::PushDebugGroup)?;

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcComputeCommand::PushDebugGroup {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn compute_pass_pop_debug_group(
        &self,
        pass: &mut ComputePass,
    ) -> Result<(), ComputePassError> {
        let base = pass.base_mut(PassErrorScope::PopDebugGroup)?;

        base.commands.push(ArcComputeCommand::PopDebugGroup);

        Ok(())
    }

    pub fn compute_pass_insert_debug_marker(
        &self,
        pass: &mut ComputePass,
        label: &str,
        color: u32,
    ) -> Result<(), ComputePassError> {
        let base = pass.base_mut(PassErrorScope::InsertDebugMarker)?;

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcComputeCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn compute_pass_write_timestamp(
        &self,
        pass: &mut ComputePass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::WriteTimestamp;
        let base = pass.base_mut(scope)?;

        let hub = &self.hub;
        let query_set = hub.query_sets.get(query_set_id).get().map_pass_err(scope)?;

        base.commands.push(ArcComputeCommand::WriteTimestamp {
            query_set,
            query_index,
        });

        Ok(())
    }

    pub fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut ComputePass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::BeginPipelineStatisticsQuery;
        let base = pass.base_mut(scope)?;

        let hub = &self.hub;
        let query_set = hub.query_sets.get(query_set_id).get().map_pass_err(scope)?;

        base.commands
            .push(ArcComputeCommand::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            });

        Ok(())
    }

    pub fn compute_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut ComputePass,
    ) -> Result<(), ComputePassError> {
        let scope = PassErrorScope::EndPipelineStatisticsQuery;
        let base = pass.base_mut(scope)?;
        base.commands
            .push(ArcComputeCommand::EndPipelineStatisticsQuery);

        Ok(())
    }
}
