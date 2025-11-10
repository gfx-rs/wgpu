use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    BufferAddress, DynamicOffset,
};

use alloc::{borrow::Cow, boxed::Box, sync::Arc, vec::Vec};
use core::{convert::Infallible, fmt, str};

use crate::{
    api_log,
    binding_model::BindError,
    command::pass::flush_bindings_helper,
    resource::{RawResourceAccess, Trackable},
};
use crate::{
    binding_model::{LateMinBufferBindingSizeMismatch, PushConstantUploadError},
    command::{
        bind::{Binder, BinderError},
        compute_command::ArcComputeCommand,
        end_pipeline_statistics_query,
        memory_init::{fixup_discarded_surfaces, SurfacesInDiscardState},
        pass_base, pass_try, validate_and_begin_pipeline_statistics_query, ArcPassTimestampWrites,
        BasePass, BindGroupStateChange, CommandEncoderError, MapPassErr, PassErrorScope,
        PassTimestampWrites, QueryUseError, StateChange,
    },
    device::{DeviceError, MissingDownlevelFlags, MissingFeatures},
    global::Global,
    hal_label, id,
    init_tracker::MemoryInitKind,
    pipeline::ComputePipeline,
    resource::{
        self, Buffer, InvalidResourceError, Labeled, MissingBufferUsageError, ParentDevice,
    },
    track::{ResourceUsageCompatibilityError, Tracker},
    Label,
};
use crate::{command::InnerCommandEncoder, resource::DestroyedResourceError};
use crate::{
    command::{
        encoder::EncodingState, pass, ArcCommand, CommandEncoder, DebugGroupError,
        EncoderStateError, PassStateError, TimestampWritesError,
    },
    device::Device,
};

pub type ComputeBasePass = BasePass<ArcComputeCommand, ComputePassError>;

/// A pass's [encoder state](https://www.w3.org/TR/webgpu/#encoder-state) and
/// its validity are two distinct conditions, i.e., the full matrix of
/// (open, ended) x (valid, invalid) is possible.
///
/// The presence or absence of the `parent` `Option` indicates the pass's state.
/// The presence or absence of an error in `base.error` indicates the pass's
/// validity.
pub struct ComputePass {
    /// All pass data & records is stored here.
    base: ComputeBasePass,

    /// Parent command encoder that this pass records commands into.
    ///
    /// If this is `Some`, then the pass is in WebGPU's "open" state. If it is
    /// `None`, then the pass is in the "ended" state.
    /// See <https://www.w3.org/TR/webgpu/#encoder-state>
    parent: Option<Arc<CommandEncoder>>,

    timestamp_writes: Option<ArcPassTimestampWrites>,

    // Resource binding dedupe state.
    current_bind_groups: BindGroupStateChange,
    current_pipeline: StateChange<id::ComputePipelineId>,
}

impl ComputePass {
    /// If the parent command encoder is invalid, the returned pass will be invalid.
    fn new(parent: Arc<CommandEncoder>, desc: ArcComputePassDescriptor) -> Self {
        let ArcComputePassDescriptor {
            label,
            timestamp_writes,
        } = desc;

        Self {
            base: BasePass::new(&label),
            parent: Some(parent),
            timestamp_writes,

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    fn new_invalid(parent: Arc<CommandEncoder>, label: &Label, err: ComputePassError) -> Self {
        Self {
            base: BasePass::new_invalid(label, err),
            parent: Some(parent),
            timestamp_writes: None,
            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.base.label.as_deref()
    }
}

impl fmt::Debug for ComputePass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.parent {
            Some(ref cmd_enc) => write!(f, "ComputePass {{ parent: {} }}", cmd_enc.error_ident()),
            None => write!(f, "ComputePass {{ parent: None }}"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor<'a, PTW = PassTimestampWrites> {
    pub label: Label<'a>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<PTW>,
}

/// cbindgen:ignore
type ArcComputePassDescriptor<'a> = ComputePassDescriptor<'a, ArcPassTimestampWrites>;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DispatchError {
    #[error("Compute pipeline must be set")]
    MissingPipeline(pass::MissingPipeline),
    #[error(transparent)]
    IncompatibleBindGroup(#[from] Box<BinderError>),
    #[error(
        "Each current dispatch group size dimension ({current:?}) must be less or equal to {limit}"
    )]
    InvalidGroupSize { current: [u32; 3], limit: u32 },
    #[error(transparent)]
    BindingSizeTooSmall(#[from] LateMinBufferBindingSizeMismatch),
}

impl WebGpuError for DispatchError {
    fn webgpu_error_type(&self) -> ErrorType {
        ErrorType::Validation
    }
}

/// Error encountered when performing a compute pass.
#[derive(Clone, Debug, Error)]
pub enum ComputePassErrorInner {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    EncoderState(#[from] EncoderStateError),
    #[error("Parent encoder is invalid")]
    InvalidParentEncoder,
    #[error(transparent)]
    DebugGroupError(#[from] DebugGroupError),
    #[error(transparent)]
    BindGroupIndexOutOfRange(#[from] pass::BindGroupIndexOutOfRange),
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
    #[error(transparent)]
    TimestampWrites(#[from] TimestampWritesError),
    // This one is unreachable, but required for generic pass support
    #[error(transparent)]
    InvalidValuesOffset(#[from] pass::InvalidValuesOffset),
}

/// Error encountered when performing a compute pass, stored for later reporting
/// when encoding ends.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct ComputePassError {
    pub scope: PassErrorScope,
    #[source]
    pub(super) inner: ComputePassErrorInner,
}

impl From<pass::MissingPipeline> for ComputePassErrorInner {
    fn from(value: pass::MissingPipeline) -> Self {
        Self::Dispatch(DispatchError::MissingPipeline(value))
    }
}

impl<E> MapPassErr<ComputePassError> for E
where
    E: Into<ComputePassErrorInner>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> ComputePassError {
        ComputePassError {
            scope,
            inner: self.into(),
        }
    }
}

impl WebGpuError for ComputePassError {
    fn webgpu_error_type(&self) -> ErrorType {
        let Self { scope: _, inner } = self;
        let e: &dyn WebGpuError = match inner {
            ComputePassErrorInner::Device(e) => e,
            ComputePassErrorInner::EncoderState(e) => e,
            ComputePassErrorInner::DebugGroupError(e) => e,
            ComputePassErrorInner::DestroyedResource(e) => e,
            ComputePassErrorInner::ResourceUsageCompatibility(e) => e,
            ComputePassErrorInner::MissingBufferUsage(e) => e,
            ComputePassErrorInner::Dispatch(e) => e,
            ComputePassErrorInner::Bind(e) => e,
            ComputePassErrorInner::PushConstants(e) => e,
            ComputePassErrorInner::QueryUse(e) => e,
            ComputePassErrorInner::MissingFeatures(e) => e,
            ComputePassErrorInner::MissingDownlevelFlags(e) => e,
            ComputePassErrorInner::InvalidResource(e) => e,
            ComputePassErrorInner::TimestampWrites(e) => e,
            ComputePassErrorInner::InvalidValuesOffset(e) => e,

            ComputePassErrorInner::InvalidParentEncoder
            | ComputePassErrorInner::BindGroupIndexOutOfRange { .. }
            | ComputePassErrorInner::UnalignedIndirectBufferOffset(_)
            | ComputePassErrorInner::IndirectBufferOverrun { .. }
            | ComputePassErrorInner::PushConstantOffsetAlignment
            | ComputePassErrorInner::PushConstantSizeAlignment
            | ComputePassErrorInner::PushConstantOutOfMemory
            | ComputePassErrorInner::PassEnded => return ErrorType::Validation,
        };
        e.webgpu_error_type()
    }
}

struct State<'scope, 'snatch_guard, 'cmd_enc> {
    pipeline: Option<Arc<ComputePipeline>>,

    pass: pass::PassState<'scope, 'snatch_guard, 'cmd_enc>,

    active_query: Option<(Arc<resource::QuerySet>, u32)>,

    push_constants: Vec<u32>,

    intermediate_trackers: Tracker,
}

impl<'scope, 'snatch_guard, 'cmd_enc> State<'scope, 'snatch_guard, 'cmd_enc> {
    fn is_ready(&self) -> Result<(), DispatchError> {
        if let Some(pipeline) = self.pipeline.as_ref() {
            self.pass.binder.check_compatibility(pipeline.as_ref())?;
            self.pass.binder.check_late_buffer_bindings()?;
            Ok(())
        } else {
            Err(DispatchError::MissingPipeline(pass::MissingPipeline))
        }
    }

    /// Flush binding state in preparation for a dispatch.
    ///
    /// # Differences between render and compute passes
    ///
    /// There are differences between the `flush_bindings` implementations for
    /// render and compute passes, because render passes have a single usage
    /// scope for the entire pass, and compute passes have a separate usage
    /// scope for each dispatch.
    ///
    /// For compute passes, bind groups are merged into a fresh usage scope
    /// here, not into the pass usage scope within calls to `set_bind_group`. As
    /// specified by WebGPU, for compute passes, we merge only the bind groups
    /// that are actually used by the pipeline, unlike render passes, which
    /// merge every bind group that is ever set, even if it is not ultimately
    /// used by the pipeline.
    ///
    /// For compute passes, we call `drain_barriers` here, because barriers may
    /// be needed before each dispatch if a previous dispatch had a conflicting
    /// usage. For render passes, barriers are emitted once at the start of the
    /// render pass.
    ///
    /// # Indirect buffer handling
    ///
    /// The `indirect_buffer` argument should be passed for any indirect
    /// dispatch (with or without validation). It will be checked for
    /// conflicting usages according to WebGPU rules. For the purpose of
    /// these rules, the fact that we have actually processed the buffer in
    /// the validation pass is an implementation detail.
    ///
    /// The `track_indirect_buffer` argument should be set when doing indirect
    /// dispatch *without* validation. In this case, the indirect buffer will
    /// be added to the tracker in order to generate any necessary transitions
    /// for that usage.
    ///
    /// When doing indirect dispatch *with* validation, the indirect buffer is
    /// processed by the validation pass and is not used by the actual dispatch.
    /// The indirect validation code handles transitions for the validation
    /// pass.
    fn flush_bindings(
        &mut self,
        indirect_buffer: Option<&Arc<Buffer>>,
        track_indirect_buffer: bool,
    ) -> Result<(), ComputePassErrorInner> {
        for bind_group in self.pass.binder.list_active() {
            unsafe { self.pass.scope.merge_bind_group(&bind_group.used)? };
        }

        // Add the indirect buffer. Because usage scopes are per-dispatch, this
        // is the only place where INDIRECT usage could be added, and it is safe
        // for us to remove it below.
        if let Some(buffer) = indirect_buffer {
            self.pass
                .scope
                .buffers
                .merge_single(buffer, wgt::BufferUses::INDIRECT)?;
        }

        // For compute, usage scopes are associated with each dispatch and not
        // with the pass as a whole. However, because the cost of creating and
        // dropping `UsageScope`s is significant (even with the pool), we
        // add and then remove usage from a single usage scope.

        for bind_group in self.pass.binder.list_active() {
            self.intermediate_trackers
                .set_and_remove_from_usage_scope_sparse(&mut self.pass.scope, &bind_group.used);
        }

        if track_indirect_buffer {
            self.intermediate_trackers
                .buffers
                .set_and_remove_from_usage_scope_sparse(
                    &mut self.pass.scope.buffers,
                    indirect_buffer.map(|buf| buf.tracker_index()),
                );
        } else if let Some(buffer) = indirect_buffer {
            self.pass
                .scope
                .buffers
                .remove_usage(buffer, wgt::BufferUses::INDIRECT);
        }

        flush_bindings_helper(&mut self.pass)?;

        CommandEncoder::drain_barriers(
            self.pass.base.raw_encoder,
            &mut self.intermediate_trackers,
            self.pass.base.snatch_guard,
        );
        Ok(())
    }
}

// Running the compute pass.

impl Global {
    /// Creates a compute pass.
    ///
    /// If creation fails, an invalid pass is returned. Attempting to record
    /// commands into an invalid pass is permitted, but a validation error will
    /// ultimately be generated when the parent encoder is finished, and it is
    /// not possible to run any commands from the invalid pass.
    ///
    /// If successful, puts the encoder into the [`Locked`] state.
    ///
    /// [`Locked`]: crate::command::CommandEncoderStatus::Locked
    pub fn command_encoder_begin_compute_pass(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor<'_>,
    ) -> (ComputePass, Option<CommandEncoderError>) {
        use EncoderStateError as SErr;

        let scope = PassErrorScope::Pass;
        let hub = &self.hub;

        let label = desc.label.as_deref().map(Cow::Borrowed);

        let cmd_enc = hub.command_encoders.get(encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();

        match cmd_buf_data.lock_encoder() {
            Ok(()) => {
                drop(cmd_buf_data);
                if let Err(err) = cmd_enc.device.check_is_valid() {
                    return (
                        ComputePass::new_invalid(cmd_enc, &label, err.map_pass_err(scope)),
                        None,
                    );
                }

                match desc
                    .timestamp_writes
                    .as_ref()
                    .map(|tw| {
                        Self::validate_pass_timestamp_writes::<ComputePassErrorInner>(
                            &cmd_enc.device,
                            &hub.query_sets.read(),
                            tw,
                        )
                    })
                    .transpose()
                {
                    Ok(timestamp_writes) => {
                        let arc_desc = ArcComputePassDescriptor {
                            label,
                            timestamp_writes,
                        };
                        (ComputePass::new(cmd_enc, arc_desc), None)
                    }
                    Err(err) => (
                        ComputePass::new_invalid(cmd_enc, &label, err.map_pass_err(scope)),
                        None,
                    ),
                }
            }
            Err(err @ SErr::Locked) => {
                // Attempting to open a new pass while the encoder is locked
                // invalidates the encoder, but does not generate a validation
                // error.
                cmd_buf_data.invalidate(err.clone());
                drop(cmd_buf_data);
                (
                    ComputePass::new_invalid(cmd_enc, &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(err @ (SErr::Ended | SErr::Submitted)) => {
                // Attempting to open a new pass after the encode has ended
                // generates an immediate validation error.
                drop(cmd_buf_data);
                (
                    ComputePass::new_invalid(cmd_enc, &label, err.clone().map_pass_err(scope)),
                    Some(err.into()),
                )
            }
            Err(err @ SErr::Invalid) => {
                // Passes can be opened even on an invalid encoder. Such passes
                // are even valid, but since there's no visible side-effect of
                // the pass being valid and there's no point in storing recorded
                // commands that will ultimately be discarded, we open an
                // invalid pass to save that work.
                drop(cmd_buf_data);
                (
                    ComputePass::new_invalid(cmd_enc, &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(SErr::Unlocked) => {
                unreachable!("lock_encoder cannot fail due to the encoder being unlocked")
            }
        }
    }

    /// Note that this differs from [`Self::compute_pass_end`], it will
    /// create a new pass, replay the commands and end the pass.
    ///
    /// # Panics
    /// On any error.
    #[doc(hidden)]
    #[cfg(any(feature = "serde", feature = "replay"))]
    pub fn compute_pass_end_with_unresolved_commands(
        &self,
        encoder_id: id::CommandEncoderId,
        base: BasePass<super::ComputeCommand, Infallible>,
        timestamp_writes: Option<&PassTimestampWrites>,
    ) {
        #[cfg(feature = "trace")]
        {
            let cmd_enc = self.hub.command_encoders.get(encoder_id);
            let mut cmd_buf_data = cmd_enc.data.lock();
            let cmd_buf_data = cmd_buf_data.get_inner();

            if let Some(ref mut list) = cmd_buf_data.trace_commands {
                list.push(crate::command::Command::RunComputePass {
                    base: BasePass {
                        label: base.label.clone(),
                        error: None,
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
            error: _,
            commands,
            dynamic_offsets,
            string_data,
            push_constant_data,
        } = base;

        let (mut compute_pass, encoder_error) = self.command_encoder_begin_compute_pass(
            encoder_id,
            &ComputePassDescriptor {
                label: label.as_deref().map(Cow::Borrowed),
                timestamp_writes: timestamp_writes.cloned(),
            },
        );
        if let Some(err) = encoder_error {
            panic!("{:?}", err);
        };

        compute_pass.base = BasePass {
            label,
            error: None,
            commands: super::ComputeCommand::resolve_compute_command_ids(&self.hub, &commands)
                .unwrap(),
            dynamic_offsets,
            string_data,
            push_constant_data,
        };

        self.compute_pass_end(&mut compute_pass).unwrap();
    }

    pub fn compute_pass_end(&self, pass: &mut ComputePass) -> Result<(), EncoderStateError> {
        profiling::scope!(
            "CommandEncoder::run_compute_pass {}",
            pass.base.label.as_deref().unwrap_or("")
        );

        let cmd_enc = pass.parent.take().ok_or(EncoderStateError::Ended)?;
        let mut cmd_buf_data = cmd_enc.data.lock();

        cmd_buf_data.unlock_encoder()?;

        let base = pass.base.take();

        if matches!(
            base,
            Err(ComputePassError {
                inner: ComputePassErrorInner::EncoderState(EncoderStateError::Ended),
                scope: _,
            })
        ) {
            // If the encoder was already finished at time of pass creation,
            // then it was not put in the locked state, so we need to
            // generate a validation error here and now due to the encoder not
            // being locked. The encoder already holds an error from when the
            // pass was opened, or earlier.
            //
            // All other errors are propagated to the encoder within `push_with`,
            // and will be reported later.
            return Err(EncoderStateError::Ended);
        }

        cmd_buf_data.push_with(|| -> Result<_, ComputePassError> {
            Ok(ArcCommand::RunComputePass {
                pass: base?,
                timestamp_writes: pass.timestamp_writes.take(),
            })
        })
    }
}

pub(super) fn encode_compute_pass(
    parent_state: &mut EncodingState<InnerCommandEncoder>,
    mut base: BasePass<ArcComputeCommand, Infallible>,
    mut timestamp_writes: Option<ArcPassTimestampWrites>,
) -> Result<(), ComputePassError> {
    let pass_scope = PassErrorScope::Pass;

    let device = parent_state.device;

    // We automatically keep extending command buffers over time, and because
    // we want to insert a command buffer _before_ what we're about to record,
    // we need to make sure to close the previous one.
    parent_state
        .raw_encoder
        .close_if_open()
        .map_pass_err(pass_scope)?;
    let raw_encoder = parent_state
        .raw_encoder
        .open_pass(base.label.as_deref())
        .map_pass_err(pass_scope)?;

    let mut debug_scope_depth = 0;

    let mut state = State {
        pipeline: None,

        pass: pass::PassState {
            base: EncodingState {
                device,
                raw_encoder,
                tracker: parent_state.tracker,
                buffer_memory_init_actions: parent_state.buffer_memory_init_actions,
                texture_memory_actions: parent_state.texture_memory_actions,
                as_actions: parent_state.as_actions,
                temp_resources: parent_state.temp_resources,
                indirect_draw_validation_resources: parent_state.indirect_draw_validation_resources,
                snatch_guard: parent_state.snatch_guard,
                debug_scope_depth: &mut debug_scope_depth,
            },
            binder: Binder::new(),
            temp_offsets: Vec::new(),
            dynamic_offset_count: 0,
            pending_discard_init_fixups: SurfacesInDiscardState::new(),
            scope: device.new_usage_scope(),
            string_offset: 0,
        },
        active_query: None,

        push_constants: Vec::new(),

        intermediate_trackers: Tracker::new(),
    };

    let indices = &device.tracker_indices;
    state
        .pass
        .base
        .tracker
        .buffers
        .set_size(indices.buffers.size());
    state
        .pass
        .base
        .tracker
        .textures
        .set_size(indices.textures.size());

    let timestamp_writes: Option<hal::PassTimestampWrites<'_, dyn hal::DynQuerySet>> =
        if let Some(tw) = timestamp_writes.take() {
            tw.query_set.same_device(device).map_pass_err(pass_scope)?;

            let query_set = state
                .pass
                .base
                .tracker
                .query_sets
                .insert_single(tw.query_set);

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
                    state
                        .pass
                        .base
                        .raw_encoder
                        .reset_queries(query_set.raw(), range);
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
        state.pass.base.raw_encoder.begin_compute_pass(&hal_desc);
    }

    for command in base.commands.drain(..) {
        match command {
            ArcComputeCommand::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => {
                let scope = PassErrorScope::SetBindGroup;
                pass::set_bind_group::<ComputePassErrorInner>(
                    &mut state.pass,
                    device,
                    &base.dynamic_offsets,
                    index,
                    num_dynamic_offsets,
                    bind_group,
                    false,
                )
                .map_pass_err(scope)?;
            }
            ArcComputeCommand::SetPipeline(pipeline) => {
                let scope = PassErrorScope::SetPipelineCompute;
                set_pipeline(&mut state, device, pipeline).map_pass_err(scope)?;
            }
            ArcComputeCommand::SetPushConstant {
                offset,
                size_bytes,
                values_offset,
            } => {
                let scope = PassErrorScope::SetPushConstant;
                pass::set_push_constant::<ComputePassErrorInner, _>(
                    &mut state.pass,
                    &base.push_constant_data,
                    wgt::ShaderStages::COMPUTE,
                    offset,
                    size_bytes,
                    Some(values_offset),
                    |data_slice| {
                        let offset_in_elements = (offset / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
                        let size_in_elements = (size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
                        state.push_constants[offset_in_elements..][..size_in_elements]
                            .copy_from_slice(data_slice);
                    },
                )
                .map_pass_err(scope)?;
            }
            ArcComputeCommand::Dispatch(groups) => {
                let scope = PassErrorScope::Dispatch { indirect: false };
                dispatch(&mut state, groups).map_pass_err(scope)?;
            }
            ArcComputeCommand::DispatchIndirect { buffer, offset } => {
                let scope = PassErrorScope::Dispatch { indirect: true };
                dispatch_indirect(&mut state, device, buffer, offset).map_pass_err(scope)?;
            }
            ArcComputeCommand::PushDebugGroup { color: _, len } => {
                pass::push_debug_group(&mut state.pass, &base.string_data, len);
            }
            ArcComputeCommand::PopDebugGroup => {
                let scope = PassErrorScope::PopDebugGroup;
                pass::pop_debug_group::<ComputePassErrorInner>(&mut state.pass)
                    .map_pass_err(scope)?;
            }
            ArcComputeCommand::InsertDebugMarker { color: _, len } => {
                pass::insert_debug_marker(&mut state.pass, &base.string_data, len);
            }
            ArcComputeCommand::WriteTimestamp {
                query_set,
                query_index,
            } => {
                let scope = PassErrorScope::WriteTimestamp;
                pass::write_timestamp::<ComputePassErrorInner>(
                    &mut state.pass,
                    device,
                    None, // compute passes do not attempt to coalesce query resets
                    query_set,
                    query_index,
                )
                .map_pass_err(scope)?;
            }
            ArcComputeCommand::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => {
                let scope = PassErrorScope::BeginPipelineStatisticsQuery;
                validate_and_begin_pipeline_statistics_query(
                    query_set,
                    state.pass.base.raw_encoder,
                    &mut state.pass.base.tracker.query_sets,
                    device,
                    query_index,
                    None,
                    &mut state.active_query,
                )
                .map_pass_err(scope)?;
            }
            ArcComputeCommand::EndPipelineStatisticsQuery => {
                let scope = PassErrorScope::EndPipelineStatisticsQuery;
                end_pipeline_statistics_query(state.pass.base.raw_encoder, &mut state.active_query)
                    .map_pass_err(scope)?;
            }
        }
    }

    if *state.pass.base.debug_scope_depth > 0 {
        Err(
            ComputePassErrorInner::DebugGroupError(DebugGroupError::MissingPop)
                .map_pass_err(pass_scope),
        )?;
    }

    unsafe {
        state.pass.base.raw_encoder.end_compute_pass();
    }

    let State {
        pass: pass::PassState {
            pending_discard_init_fixups,
            ..
        },
        intermediate_trackers,
        ..
    } = state;

    // Stop the current command encoder.
    parent_state.raw_encoder.close().map_pass_err(pass_scope)?;

    // Create a new command encoder, which we will insert _before_ the body of the compute pass.
    //
    // Use that buffer to insert barriers and clear discarded images.
    let transit = parent_state
        .raw_encoder
        .open_pass(hal_label(
            Some("(wgpu internal) Pre Pass"),
            device.instance_flags,
        ))
        .map_pass_err(pass_scope)?;
    fixup_discarded_surfaces(
        pending_discard_init_fixups.into_iter(),
        transit,
        &mut parent_state.tracker.textures,
        device,
        parent_state.snatch_guard,
    );
    CommandEncoder::insert_barriers_from_tracker(
        transit,
        parent_state.tracker,
        &intermediate_trackers,
        parent_state.snatch_guard,
    );
    // Close the command encoder, and swap it with the previous.
    parent_state
        .raw_encoder
        .close_and_swap()
        .map_pass_err(pass_scope)?;

    Ok(())
}

fn set_pipeline(
    state: &mut State,
    device: &Arc<Device>,
    pipeline: Arc<ComputePipeline>,
) -> Result<(), ComputePassErrorInner> {
    pipeline.same_device(device)?;

    state.pipeline = Some(pipeline.clone());

    let pipeline = state
        .pass
        .base
        .tracker
        .compute_pipelines
        .insert_single(pipeline)
        .clone();

    unsafe {
        state
            .pass
            .base
            .raw_encoder
            .set_compute_pipeline(pipeline.raw());
    }

    // Rebind resources
    pass::change_pipeline_layout::<ComputePassErrorInner, _>(
        &mut state.pass,
        &pipeline.layout,
        &pipeline.late_sized_buffer_groups,
        || {
            // This only needs to be here for compute pipelines because they use push constants for
            // validating indirect draws.
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
                state.push_constants.extend(core::iter::repeat_n(0, len));
            }
        },
    )
}

fn dispatch(state: &mut State, groups: [u32; 3]) -> Result<(), ComputePassErrorInner> {
    api_log!("ComputePass::dispatch {groups:?}");

    state.is_ready()?;

    state.flush_bindings(None, false)?;

    let groups_size_limit = state
        .pass
        .base
        .device
        .limits
        .max_compute_workgroups_per_dimension;

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
        state.pass.base.raw_encoder.dispatch(groups);
    }
    Ok(())
}

fn dispatch_indirect(
    state: &mut State,
    device: &Arc<Device>,
    buffer: Arc<Buffer>,
    offset: u64,
) -> Result<(), ComputePassErrorInner> {
    api_log!("ComputePass::dispatch_indirect");

    buffer.same_device(device)?;

    state.is_ready()?;

    state
        .pass
        .base
        .device
        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)?;

    buffer.check_usage(wgt::BufferUsages::INDIRECT)?;
    buffer.check_destroyed(state.pass.base.snatch_guard)?;

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
    state.pass.base.buffer_memory_init_actions.extend(
        buffer.initialization_status.read().create_action(
            &buffer,
            offset..(offset + stride),
            MemoryInitKind::NeedsInitializedMemory,
        ),
    );

    if let Some(ref indirect_validation) = state.pass.base.device.indirect_validation {
        let params = indirect_validation.dispatch.params(
            &state.pass.base.device.limits,
            offset,
            buffer.size,
        );

        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .set_compute_pipeline(params.pipeline);
        }

        unsafe {
            state.pass.base.raw_encoder.set_push_constants(
                params.pipeline_layout,
                wgt::ShaderStages::COMPUTE,
                0,
                &[params.offset_remainder as u32 / 4],
            );
        }

        unsafe {
            state.pass.base.raw_encoder.set_bind_group(
                params.pipeline_layout,
                0,
                Some(params.dst_bind_group),
                &[],
            );
        }
        unsafe {
            state.pass.base.raw_encoder.set_bind_group(
                params.pipeline_layout,
                1,
                Some(
                    buffer
                        .indirect_validation_bind_groups
                        .get(state.pass.base.snatch_guard)
                        .unwrap()
                        .dispatch
                        .as_ref(),
                ),
                &[params.aligned_offset as u32],
            );
        }

        let src_transition = state
            .intermediate_trackers
            .buffers
            .set_single(&buffer, wgt::BufferUses::STORAGE_READ_ONLY);
        let src_barrier = src_transition
            .map(|transition| transition.into_hal(&buffer, state.pass.base.snatch_guard));
        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .transition_buffers(src_barrier.as_slice());
        }

        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .transition_buffers(&[hal::BufferBarrier {
                    buffer: params.dst_buffer,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::INDIRECT,
                        to: wgt::BufferUses::STORAGE_READ_WRITE,
                    },
                }]);
        }

        unsafe {
            state.pass.base.raw_encoder.dispatch([1, 1, 1]);
        }

        // reset state
        {
            let pipeline = state.pipeline.as_ref().unwrap();

            unsafe {
                state
                    .pass
                    .base
                    .raw_encoder
                    .set_compute_pipeline(pipeline.raw());
            }

            if !state.push_constants.is_empty() {
                unsafe {
                    state.pass.base.raw_encoder.set_push_constants(
                        pipeline.layout.raw(),
                        wgt::ShaderStages::COMPUTE,
                        0,
                        &state.push_constants,
                    );
                }
            }

            for (i, e) in state.pass.binder.list_valid() {
                let group = e.group.as_ref().unwrap();
                let raw_bg = group.try_raw(state.pass.base.snatch_guard)?;
                unsafe {
                    state.pass.base.raw_encoder.set_bind_group(
                        pipeline.layout.raw(),
                        i as u32,
                        Some(raw_bg),
                        &e.dynamic_offsets,
                    );
                }
            }
        }

        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .transition_buffers(&[hal::BufferBarrier {
                    buffer: params.dst_buffer,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::STORAGE_READ_WRITE,
                        to: wgt::BufferUses::INDIRECT,
                    },
                }]);
        }

        state.flush_bindings(Some(&buffer), false)?;
        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .dispatch_indirect(params.dst_buffer, 0);
        }
    } else {
        state.flush_bindings(Some(&buffer), true)?;

        let buf_raw = buffer.try_raw(state.pass.base.snatch_guard)?;
        unsafe {
            state
                .pass
                .base
                .raw_encoder
                .dispatch_indirect(buf_raw, offset);
        }
    }

    Ok(())
}

// Recording a compute pass.
//
// The only error that should be returned from these methods is
// `EncoderStateError::Ended`, when the pass has already ended and an immediate
// validation error is raised.
//
// All other errors should be stored in the pass for later reporting when
// `CommandEncoder.finish()` is called.
//
// The `pass_try!` macro should be used to handle errors appropriately. Note
// that the `pass_try!` and `pass_base!` macros may return early from the
// function that invokes them, like the `?` operator.
impl Global {
    pub fn compute_pass_set_bind_group(
        &self,
        pass: &mut ComputePass,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetBindGroup;

        // This statement will return an error if the pass is ended. It's
        // important the error check comes before the early-out for
        // `set_and_check_redundant`.
        let base = pass_base!(pass, scope);

        if pass.current_bind_groups.set_and_check_redundant(
            bind_group_id,
            index,
            &mut base.dynamic_offsets,
            offsets,
        ) {
            return Ok(());
        }

        let mut bind_group = None;
        if bind_group_id.is_some() {
            let bind_group_id = bind_group_id.unwrap();

            let hub = &self.hub;
            bind_group = Some(pass_try!(
                base,
                scope,
                hub.bind_groups.get(bind_group_id).get(),
            ));
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
    ) -> Result<(), PassStateError> {
        let redundant = pass.current_pipeline.set_and_check_redundant(pipeline_id);

        let scope = PassErrorScope::SetPipelineCompute;

        // This statement will return an error if the pass is ended.
        // Its important the error check comes before the early-out for `redundant`.
        let base = pass_base!(pass, scope);

        if redundant {
            return Ok(());
        }

        let hub = &self.hub;
        let pipeline = pass_try!(base, scope, hub.compute_pipelines.get(pipeline_id).get());

        base.commands.push(ArcComputeCommand::SetPipeline(pipeline));

        Ok(())
    }

    pub fn compute_pass_set_push_constants(
        &self,
        pass: &mut ComputePass,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetPushConstant;
        let base = pass_base!(pass, scope);

        if offset & (wgt::PUSH_CONSTANT_ALIGNMENT - 1) != 0 {
            pass_try!(
                base,
                scope,
                Err(ComputePassErrorInner::PushConstantOffsetAlignment),
            );
        }

        if data.len() as u32 & (wgt::PUSH_CONSTANT_ALIGNMENT - 1) != 0 {
            pass_try!(
                base,
                scope,
                Err(ComputePassErrorInner::PushConstantSizeAlignment),
            )
        }
        let value_offset = pass_try!(
            base,
            scope,
            base.push_constant_data
                .len()
                .try_into()
                .map_err(|_| ComputePassErrorInner::PushConstantOutOfMemory)
        );

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
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::Dispatch { indirect: false };

        pass_base!(pass, scope)
            .commands
            .push(ArcComputeCommand::Dispatch([groups_x, groups_y, groups_z]));

        Ok(())
    }

    pub fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut ComputePass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        let hub = &self.hub;
        let scope = PassErrorScope::Dispatch { indirect: true };
        let base = pass_base!(pass, scope);

        let buffer = pass_try!(base, scope, hub.buffers.get(buffer_id).get());

        base.commands
            .push(ArcComputeCommand::DispatchIndirect { buffer, offset });

        Ok(())
    }

    pub fn compute_pass_push_debug_group(
        &self,
        pass: &mut ComputePass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::PushDebugGroup);

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
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::PopDebugGroup);

        base.commands.push(ArcComputeCommand::PopDebugGroup);

        Ok(())
    }

    pub fn compute_pass_insert_debug_marker(
        &self,
        pass: &mut ComputePass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::InsertDebugMarker);

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
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::WriteTimestamp;
        let base = pass_base!(pass, scope);

        let hub = &self.hub;
        let query_set = pass_try!(base, scope, hub.query_sets.get(query_set_id).get());

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
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::BeginPipelineStatisticsQuery;
        let base = pass_base!(pass, scope);

        let hub = &self.hub;
        let query_set = pass_try!(base, scope, hub.query_sets.get(query_set_id).get());

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
    ) -> Result<(), PassStateError> {
        pass_base!(pass, PassErrorScope::EndPipelineStatisticsQuery)
            .commands
            .push(ArcComputeCommand::EndPipelineStatisticsQuery);

        Ok(())
    }
}
