use core::{convert::Infallible, fmt};

use alloc::{borrow::Cow, boxed::Box, sync::Arc, vec::Vec};

use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    BufferAddress, DynamicOffset,
};

use crate::{
    api_log,
    binding_model::{BindError, ImmediateUploadError, LateMinBufferBindingSizeMismatch},
    command::{
        bind::{Binder, BinderError},
        memory_init::SurfacesInDiscardState,
        pass::{self, flush_bindings_helper},
        pass_base, pass_try,
        ray_tracing_pass_commands::ArcRayTracingCommand,
        ArcCommand, BasePass, BindGroupStateChange, CommandEncoder, CommandEncoderError,
        DebugGroupError, EncoderStateError, EncodingState, InnerCommandEncoder, MapPassErr,
        PassErrorScope, PassStateError, StateChange,
    },
    device::{Device, DeviceError, MissingDownlevelFlags, MissingFeatures},
    global::Global,
    hal_label, id,
    pipeline::RayTracingPipeline,
    resource::{
        DestroyedResourceError, InvalidResourceError, Labeled, MissingBufferUsageError,
        ParentDevice,
    },
    track::{ResourceUsageCompatibilityError, Tracker},
    Label,
};

pub type RayTracingBasePass = BasePass<ArcRayTracingCommand, RayTracingPassError>;

/// Very similar to [`super::compute::ComputePass`]
pub struct RayTracingPass {
    /// All pass data & records is stored here.
    base: RayTracingBasePass,

    /// Parent command encoder that this pass records commands into.
    ///
    /// Implications are the same as [`super::compute::ComputePass::parent`]
    parent: Option<Arc<CommandEncoder>>,

    current_bind_groups: BindGroupStateChange,
    current_pipeline: StateChange<id::RayTracingPipelineId>,
}

impl fmt::Debug for RayTracingPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.parent {
            Some(ref cmd_enc) => {
                write!(f, "RayTracingPass {{ parent: {} }}", cmd_enc.error_ident())
            }
            None => write!(f, "RayTracingPass {{ parent: None }}"),
        }
    }
}

impl RayTracingPass {
    /// If the parent command encoder is invalid, the returned pass will be invalid.
    fn new(parent: Arc<CommandEncoder>, desc: RayTracingPassDescriptor) -> Self {
        let RayTracingPassDescriptor { label } = desc;

        Self {
            base: BasePass::new(&label),
            parent: Some(parent),

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    fn new_invalid(parent: Arc<CommandEncoder>, label: &Label, err: RayTracingPassError) -> Self {
        Self {
            base: BasePass::new_invalid(label, err),
            parent: Some(parent),
            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.base.label.as_deref()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RayTracingPassDescriptor<'a> {
    pub label: Label<'a>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TraceRayError {
    #[error("Ray tracing pipeline must be set")]
    MissingPipeline(pass::MissingPipeline),
    #[error(transparent)]
    IncompatibleBindGroup(#[from] Box<BinderError>),
    #[error("Trace ray size ({current:?}) must be less or equal to {limit:?}")]
    InvalidDimensionSize { current: [u32; 3], limit: [u32; 3] },
    #[error("The total count of rays invocations ({current:?}) must be less or equal to {limit}")]
    TooManyTotal { current: u32, limit: u32 },
    #[error(transparent)]
    BindingSizeTooSmall(#[from] LateMinBufferBindingSizeMismatch),
}

impl WebGpuError for TraceRayError {
    fn webgpu_error_type(&self) -> ErrorType {
        ErrorType::Validation
    }
}

/// Error encountered when performing a ray tracing pass.
#[derive(Clone, Debug, Error)]
pub enum RayTracingPassErrorInner {
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
    TraceRay(#[from] TraceRayError),
    #[error(transparent)]
    Bind(#[from] BindError),
    #[error(transparent)]
    ImmediateData(#[from] ImmediateUploadError),
    #[error("Immediate data offset must be aligned to 4 bytes")]
    ImmediateOffsetAlignment,
    #[error("Immediate data size must be aligned to 4 bytes")]
    ImmediateDataizeAlignment,
    #[error("Ran out of immediate data space. Don't set 4gb of immediates per RayTracingPass.")]
    ImmediateOutOfMemory,
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("The ray tracing pass has already been ended and no further commands can be recorded")]
    PassEnded,
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    // This one is unreachable, but required for generic pass support
    #[error(transparent)]
    InvalidValuesOffset(#[from] pass::InvalidValuesOffset),
}

/// Error encountered when performing a ray tracing pass, stored for later reporting
/// when encoding ends.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct RayTracingPassError {
    pub scope: PassErrorScope,
    #[source]
    pub(super) inner: RayTracingPassErrorInner,
}

impl From<pass::MissingPipeline> for RayTracingPassErrorInner {
    fn from(value: pass::MissingPipeline) -> Self {
        Self::TraceRay(TraceRayError::MissingPipeline(value))
    }
}

impl<E> MapPassErr<RayTracingPassError> for E
where
    E: Into<RayTracingPassErrorInner>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> RayTracingPassError {
        RayTracingPassError {
            scope,
            inner: self.into(),
        }
    }
}

impl WebGpuError for RayTracingPassError {
    fn webgpu_error_type(&self) -> ErrorType {
        let Self { scope: _, inner } = self;
        match inner {
            RayTracingPassErrorInner::Device(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::EncoderState(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::DebugGroupError(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::DestroyedResource(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::ResourceUsageCompatibility(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::MissingBufferUsage(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::TraceRay(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::Bind(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::ImmediateData(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::MissingFeatures(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::MissingDownlevelFlags(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::InvalidResource(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::InvalidValuesOffset(e) => e.webgpu_error_type(),

            RayTracingPassErrorInner::InvalidParentEncoder
            | RayTracingPassErrorInner::BindGroupIndexOutOfRange { .. }
            | RayTracingPassErrorInner::UnalignedIndirectBufferOffset(_)
            | RayTracingPassErrorInner::IndirectBufferOverrun { .. }
            | RayTracingPassErrorInner::ImmediateOffsetAlignment
            | RayTracingPassErrorInner::ImmediateDataizeAlignment
            | RayTracingPassErrorInner::ImmediateOutOfMemory
            | RayTracingPassErrorInner::PassEnded => ErrorType::Validation,
        }
    }
}

struct State<'scope, 'snatch_guard, 'cmd_enc> {
    pipeline: Option<Arc<RayTracingPipeline>>,

    pass: pass::PassState<'scope, 'snatch_guard, 'cmd_enc>,

    intermediate_trackers: Tracker,
}

impl<'scope, 'snatch_guard, 'cmd_enc> State<'scope, 'snatch_guard, 'cmd_enc> {
    fn is_ready(&self) -> Result<(), TraceRayError> {
        if let Some(pipeline) = self.pipeline.as_ref() {
            self.pass.binder.check_compatibility(pipeline.as_ref())?;
            self.pass.binder.check_late_buffer_bindings()?;
            Ok(())
        } else {
            Err(TraceRayError::MissingPipeline(pass::MissingPipeline))
        }
    }

    /// Flush binding state in preparation for a trace rays call.
    ///
    /// # Differences between render and compute (from which ray tracing passes inherit functionality) passes
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
    fn flush_bindings(&mut self) -> Result<(), RayTracingPassErrorInner> {
        for bind_group in self.pass.binder.list_active() {
            unsafe { self.pass.scope.merge_bind_group(&bind_group.used)? };
        }
        // For compute, usage scopes are associated with each dispatch and not
        // with the pass as a whole. However, because the cost of creating and
        // dropping `UsageScope`s is significant (even with the pool), we
        // add and then remove usage from a single usage scope.

        for bind_group in self.pass.binder.list_active() {
            self.intermediate_trackers
                .set_and_remove_from_usage_scope_sparse(&mut self.pass.scope, &bind_group.used);
        }

        flush_bindings_helper(
            &mut self.pass,
            crate::resource::MaxIntersectionIndex::Intersection(
                self.pipeline
                    .as_ref()
                    .unwrap()
                    .shader_binding_data
                    .num_intersection_groups as _,
            ),
        )?;

        CommandEncoder::drain_barriers(
            self.pass.base.raw_encoder,
            &mut self.intermediate_trackers,
            self.pass.base.snatch_guard,
        );
        Ok(())
    }
}

// Ray tracing pass commands

impl Global {
    /// Creates a ray tracing pass.
    ///
    /// If creation fails, an invalid pass is returned. Attempting to record
    /// commands into an invalid pass is permitted, but a validation error will
    /// ultimately be generated when the parent encoder is finished, and it is
    /// not possible to run any commands from the invalid pass.
    ///
    /// If successful, puts the encoder into the [`Locked`] state.
    ///
    /// [`Locked`]: crate::command::CommandEncoderStatus::Locked
    pub fn command_encoder_begin_ray_tracing_pass(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &RayTracingPassDescriptor<'_>,
    ) -> (RayTracingPass, Option<CommandEncoderError>) {
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
                        RayTracingPass::new_invalid(cmd_enc, &label, err.map_pass_err(scope)),
                        None,
                    );
                }

                (
                    RayTracingPass::new(cmd_enc, RayTracingPassDescriptor { label }),
                    None,
                )
            }
            Err(err @ SErr::Locked) => {
                // Attempting to open a new pass while the encoder is locked
                // invalidates the encoder, but does not generate a validation
                // error.
                cmd_buf_data.invalidate(err.clone());
                drop(cmd_buf_data);
                (
                    RayTracingPass::new_invalid(cmd_enc, &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(err @ (SErr::Ended | SErr::Submitted)) => {
                // Attempting to open a new pass after the encode has ended
                // generates an immediate validation error.
                drop(cmd_buf_data);
                (
                    RayTracingPass::new_invalid(cmd_enc, &label, err.clone().map_pass_err(scope)),
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
                    RayTracingPass::new_invalid(cmd_enc, &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(SErr::Unlocked) => {
                unreachable!("lock_encoder cannot fail due to the encoder being unlocked")
            }
        }
    }

    pub fn ray_tracing_pass_set_pipeline(
        &self,
        pass: &mut RayTracingPass,
        pipeline_id: id::RayTracingPipelineId,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetPipelineRender;

        let redundant = pass.current_pipeline.set_and_check_redundant(pipeline_id);

        // This statement will return an error if the pass is ended.
        // Its important the error check comes before the early-out for `redundant`.
        let base = pass_base!(pass, scope);

        if redundant {
            return Ok(());
        }

        let hub = &self.hub;
        let pipeline = pass_try!(
            base,
            scope,
            hub.ray_tracing_pipelines.get(pipeline_id).get()
        );

        base.commands
            .push(ArcRayTracingCommand::SetPipeline(pipeline));

        Ok(())
    }

    pub fn ray_tracing_pass_set_bind_group(
        &self,
        pass: &mut RayTracingPass,
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
        if let Some(bind_group_id) = bind_group_id {
            let hub = &self.hub;
            bind_group = Some(pass_try!(
                base,
                scope,
                hub.bind_groups.get(bind_group_id).get(),
            ));
        }

        base.commands.push(ArcRayTracingCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offsets.len(),
            bind_group,
        });

        Ok(())
    }

    pub fn ray_tracing_pass_set_immediates(
        &self,
        pass: &mut RayTracingPass,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetImmediate;
        let base = pass_base!(pass, scope);

        if offset & (wgt::IMMEDIATE_DATA_ALIGNMENT - 1) != 0 {
            pass_try!(
                base,
                scope,
                Err(RayTracingPassErrorInner::ImmediateOffsetAlignment)
            );
        }
        if data.len() as u32 & (wgt::IMMEDIATE_DATA_ALIGNMENT - 1) != 0 {
            pass_try!(
                base,
                scope,
                Err(RayTracingPassErrorInner::ImmediateDataizeAlignment)
            );
        }

        let value_offset = pass_try!(
            base,
            scope,
            base.immediates_data
                .len()
                .try_into()
                .map_err(|_| RayTracingPassErrorInner::ImmediateOutOfMemory),
        );

        base.immediates_data.extend(
            data.chunks_exact(wgt::IMMEDIATE_DATA_ALIGNMENT as usize)
                .map(|arr| u32::from_ne_bytes([arr[0], arr[1], arr[2], arr[3]])),
        );

        base.commands.push(ArcRayTracingCommand::SetImmediate {
            offset,
            size_bytes: data.len() as u32,
            values_offset: value_offset,
        });

        Ok(())
    }

    pub fn ray_tracing_pass_push_debug_group(
        &self,
        pass: &mut RayTracingPass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::PushDebugGroup);

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcRayTracingCommand::PushDebugGroup {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn ray_tracing_pass_pop_debug_group(
        &self,
        pass: &mut RayTracingPass,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::PopDebugGroup);

        base.commands.push(ArcRayTracingCommand::PopDebugGroup);

        Ok(())
    }

    pub fn ray_tracing_pass_insert_debug_marker(
        &self,
        pass: &mut RayTracingPass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(pass, PassErrorScope::InsertDebugMarker);

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcRayTracingCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn ray_tracing_pass_trace_rays(
        &self,
        pass: &mut RayTracingPass,
        count_x: u32,
        count_y: u32,
        count_z: u32,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::TraceRays;

        pass_base!(pass, scope)
            .commands
            .push(ArcRayTracingCommand::TraceRays([count_x, count_y, count_z]));

        Ok(())
    }

    pub fn ray_tracing_pass_end(&self, pass: &mut RayTracingPass) -> Result<(), EncoderStateError> {
        profiling::scope!(
            "CommandEncoder::encode_ray_tracing_pass {}",
            pass.base.label.as_deref().unwrap_or("")
        );

        let cmd_enc = pass.parent.take().ok_or(EncoderStateError::Ended)?;
        let mut cmd_buf_data = cmd_enc.data.lock();

        cmd_buf_data.unlock_encoder()?;

        let base = pass.base.take();

        if let Err(RayTracingPassError {
            inner:
                RayTracingPassErrorInner::EncoderState(
                    err @ (EncoderStateError::Locked | EncoderStateError::Ended),
                ),
            scope: _,
        }) = base
        {
            // Most encoding errors are detected and raised within `finish()`.
            //
            // However, we raise a validation error here if the pass was opened
            // within another pass, or on a finished encoder. The latter is
            // particularly important, because in that case reporting errors via
            // `CommandEncoder::finish` is not possible.
            return Err(err.clone());
        }

        cmd_buf_data.push_with(|| -> Result<_, RayTracingPassError> {
            Ok(ArcCommand::RunRayTracingPass { pass: base? })
        })
    }
}

pub(super) fn encode_ray_tracing_pass(
    parent_state: &mut EncodingState<InnerCommandEncoder>,
    mut base: BasePass<ArcRayTracingCommand, Infallible>,
) -> Result<(), RayTracingPassError> {
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

    let hal_desc = hal::RayTracingPassDescriptor {
        label: hal_label(base.label.as_deref(), device.instance_flags),
    };

    unsafe {
        state
            .pass
            .base
            .raw_encoder
            .begin_ray_tracing_pass(&hal_desc);
    }

    for command in base.commands.drain(..) {
        match command {
            ArcRayTracingCommand::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => {
                let scope = PassErrorScope::SetBindGroup;
                pass::set_bind_group::<RayTracingPassErrorInner>(
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
            ArcRayTracingCommand::SetPipeline(pipeline) => {
                let scope = PassErrorScope::SetPipelineCompute;
                set_pipeline(&mut state, device, pipeline).map_pass_err(scope)?;
            }
            ArcRayTracingCommand::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            } => {
                let scope = PassErrorScope::SetImmediate;

                pass::set_immediates::<RayTracingPassErrorInner, _>(
                    &mut state.pass,
                    &base.immediates_data,
                    offset,
                    size_bytes,
                    Some(values_offset),
                    |_| {},
                )
                .map_pass_err(scope)?;
            }
            ArcRayTracingCommand::PushDebugGroup { color: _, len } => {
                pass::push_debug_group(&mut state.pass, &base.string_data, len);
            }
            ArcRayTracingCommand::PopDebugGroup => {
                let scope = PassErrorScope::PopDebugGroup;
                pass::pop_debug_group::<RayTracingPassErrorInner>(&mut state.pass)
                    .map_pass_err(scope)?;
            }
            ArcRayTracingCommand::InsertDebugMarker { color: _, len } => {
                pass::insert_debug_marker(&mut state.pass, &base.string_data, len);
            }
            ArcRayTracingCommand::TraceRays(groups) => {
                let scope = PassErrorScope::TraceRays;
                trace_rays(&mut state, groups, device).map_pass_err(scope)?;
            }
        }
    }

    Ok(())
}

fn set_pipeline(
    state: &mut State,
    device: &Arc<Device>,
    pipeline: Arc<RayTracingPipeline>,
) -> Result<(), RayTracingPassErrorInner> {
    pipeline.same_device(device)?;

    state.pipeline = Some(pipeline.clone());

    let pipeline = state
        .pass
        .base
        .tracker
        .ray_tracing_pipelines
        .insert_single(pipeline)
        .clone();

    unsafe {
        state
            .pass
            .base
            .raw_encoder
            .set_ray_tracing_pipeline(pipeline.raw());
    }

    // Rebind resources
    pass::change_pipeline_layout::<RayTracingPassErrorInner, _>(
        &mut state.pass,
        &pipeline.layout,
        &pipeline.late_sized_buffer_groups,
        || {},
    )
}

fn trace_rays(
    state: &mut State,
    dims: [u32; 3],
    device: &Device,
) -> Result<(), RayTracingPassErrorInner> {
    api_log!("RayTracingPass::trace_rays {dims:?}");

    state.is_ready()?;

    state.flush_bindings()?;

    let limits = &state.pass.base.device.limits;

    // todo
    let dim_size_limit = [
        limits.max_compute_workgroup_size_x * limits.max_compute_workgroups_per_dimension,
        limits.max_compute_workgroup_size_y * limits.max_compute_workgroups_per_dimension,
        limits.max_compute_workgroup_size_z * limits.max_compute_workgroups_per_dimension,
    ];

    if dims[0] > dim_size_limit[0] {
        return Err(RayTracingPassErrorInner::TraceRay(
            TraceRayError::InvalidDimensionSize {
                current: dims,
                limit: dim_size_limit,
            },
        ));
    }

    if dims[1] > dim_size_limit[1] {
        return Err(RayTracingPassErrorInner::TraceRay(
            TraceRayError::InvalidDimensionSize {
                current: dims,
                limit: dim_size_limit,
            },
        ));
    }

    if dims[2] > dim_size_limit[2] {
        return Err(RayTracingPassErrorInner::TraceRay(
            TraceRayError::InvalidDimensionSize {
                current: dims,
                limit: dim_size_limit,
            },
        ));
    }

    let tot_rays = dims[0] * dims[1] * dims[2];

    if tot_rays > limits.max_ray_dispatch_count {
        return Err(RayTracingPassErrorInner::TraceRay(
            TraceRayError::TooManyTotal {
                current: tot_rays,
                limit: limits.max_ray_dispatch_count,
            },
        ));
    }

    let current_pipeline = state.pipeline.as_ref().unwrap();

    unsafe {
        state.pass.base.raw_encoder.trace_rays(
            dims,
            hal::PipelineGroupData {
                buffer: current_pipeline.shader_binding_data.raw.as_ref(),
                offset: 0,
                stride: device.alignments.ray_tracing_pipeline_group_data_alignment as _,
                size: device.alignments.ray_tracing_pipeline_group_data_size as u64,
            },
            hal::PipelineGroupData {
                buffer: current_pipeline.shader_binding_data.raw.as_ref(),
                offset: device.alignments.ray_tracing_pipeline_group_data_size as u64,
                stride: device.alignments.ray_tracing_pipeline_group_data_alignment as _,
                size: device.alignments.ray_tracing_pipeline_group_data_size as u64,
            },
            hal::PipelineGroupData {
                buffer: current_pipeline.shader_binding_data.raw.as_ref(),
                offset: 2 * device.alignments.ray_tracing_pipeline_group_data_size as u64,
                stride: device.alignments.ray_tracing_pipeline_group_data_alignment as _,
                size: device.alignments.ray_tracing_pipeline_group_data_size as u64
                    * current_pipeline.shader_binding_data.num_intersection_groups,
            },
        );
    }
    Ok(())
}
