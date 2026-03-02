use core::{convert::Infallible, fmt};

use alloc::{borrow::Cow, boxed::Box, sync::Arc, vec::Vec};

use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    BufferAddress, DynamicOffset,
};

use crate::{
    binding_model::{BindError, BindGroup, ImmediateUploadError, LateMinBufferBindingSizeMismatch},
    command::{
        bind::{Binder, BinderError},
        memory_init::SurfacesInDiscardState,
        pass::{self, ImmediateState},
        pass_base, pass_try,
        ray_tracing_pass_commands::ArcRayTracingCommand,
        ArcCommand, BasePass, BindGroupStateChange, CommandEncoder, CommandEncoderError,
        DebugGroupError, EncoderStateError, EncodingState, InnerCommandEncoder, MapPassErr,
        PassErrorScope, PassStateError, StateChange,
    },
    device::{Device, DeviceError, MissingDownlevelFlags, MissingFeatures},
    hal_label, id, impl_resource_type,
    pipeline::RayTracingPipeline,
    resource::{
        DestroyedResourceError, InvalidResourceError, Labeled, MissingBufferUsageError,
        ParentDevice,
    },
    track::ResourceUsageCompatibilityError,
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

    device: Arc<Device>,

    current_bind_groups: BindGroupStateChange,
    current_pipeline: StateChange<Arc<RayTracingPipeline>>,
}

impl_resource_type!(RayTracingPass);

impl crate::storage::StorageItem for RayTracingPass {
    type Marker = id::markers::RayTracingPassEncoder;
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
            device: parent.device.clone(),
            parent: Some(parent),

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    fn new_invalid(parent: Arc<CommandEncoder>, label: &Label, err: RayTracingPassError) -> Self {
        Self {
            base: BasePass::new_invalid(label, err),
            device: parent.device.clone(),
            parent: Some(parent),
            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.base.label.as_deref()
    }

    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<Arc<BindGroup>>,
        offsets: &[DynamicOffset],
    ) {
        if let Err(err) = self.set_bind_group_inner(index, bind_group, offsets) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::set_bind_group");
        }
    }

    pub fn set_bind_group_inner(
        &mut self,
        index: u32,
        bind_group: Option<Arc<BindGroup>>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetBindGroup;

        // This statement will return an error if the pass is ended. It's
        // important the error check comes before the early-out for
        // `set_and_check_redundant`.
        let base = pass_base!(self, scope);

        if self.current_bind_groups.set_and_check_redundant(
            &bind_group,
            index,
            &mut base.dynamic_offsets,
            offsets,
        ) {
            return Ok(());
        }

        let bind_group = if let Some(bind_group) = bind_group {
            pass_try!(base, scope, bind_group.check_is_valid());
            Some(bind_group)
        } else {
            None
        };

        base.commands.push(ArcRayTracingCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offsets.len(),
            bind_group,
        });

        Ok(())
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
    #[error("Trace ray size ({current:?}) must be less or equal to {limit}")]
    InvalidGroupSize { current: [u32; 3], limit: u32 },
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
}

/// Error encountered when performing a ray tracing pass, stored for later reporting
/// when encoding ends.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct RayTracingPassError {
    pub scope: PassErrorScope,
    #[source]
    pub(super) inner: Box<RayTracingPassErrorInner>,
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
            inner: Box::new(self.into()),
        }
    }
}

impl WebGpuError for RayTracingPassError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self.inner.as_ref() {
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
}

// Ray tracing pass commands

impl RayTracingPass {
    pub fn set_pipeline(&mut self, pipeline: Arc<RayTracingPipeline>) {
        if let Err(err) = self.set_pipeline_inner(pipeline) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::set_pipeline");
        }
    }
    pub fn set_pipeline_inner(
        &mut self,
        pipeline: Arc<RayTracingPipeline>,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetPipelineRender;

        let redundant = self.current_pipeline.set_and_check_redundant(&pipeline);

        // This statement will return an error if the pass is ended.
        // Its important the error check comes before the early-out for `redundant`.
        let base = pass_base!(self, scope);

        if redundant {
            return Ok(());
        }

        pass_try!(base, scope, pipeline.check_valid());

        base.commands
            .push(ArcRayTracingCommand::SetPipeline(pipeline));

        Ok(())
    }
    pub fn end_inner(&mut self) -> Result<(), EncoderStateError> {
        profiling::scope!(
            "CommandEncoder::encode_ray_tracing_pass {}",
            self.base.label.as_deref().unwrap_or("")
        );

        let cmd_enc = self.parent.take().ok_or(EncoderStateError::Ended)?;
        let mut cmd_buf_data = cmd_enc.data.lock();

        cmd_buf_data.unlock_encoder()?;

        let base = self.base.take();

        if let Err(RayTracingPassError { inner, scope: _ }) = &base {
            if let RayTracingPassErrorInner::EncoderState(
                err @ (EncoderStateError::Locked | EncoderStateError::Ended),
            ) = inner.as_ref()
            {
                // Most encoding errors are detected and raised within `finish()`.
                //
                // However, we raise a validation error here if the pass was opened
                // within another pass, or on a finished encoder. The latter is
                // particularly important, because in that case reporting errors via
                // `CommandEncoder::finish` is not possible.
                return Err(err.clone());
            }
        }

        cmd_buf_data.push_with(|| -> Result<_, RayTracingPassError> {
            Ok(ArcCommand::RunRayTracingPass { pass: base? })
        })
    }

    pub fn end(&mut self) {
        if let Err(err) = self.end_inner() {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::end");
        }
    }
}

impl CommandEncoder {
    pub fn begin_ray_tracing_pass_inner(
        self: &Arc<Self>,
        desc: &RayTracingPassDescriptor<'_>,
    ) -> (RayTracingPass, Option<CommandEncoderError>) {
        use EncoderStateError as SErr;

        let label = desc.label.as_deref().map(Cow::Borrowed);

        let scope = PassErrorScope::Pass;
        let mut cmd_buf_data = self.data.lock();

        match cmd_buf_data.lock_encoder() {
            Ok(()) => {
                drop(cmd_buf_data);
                if let Err(err) = self.device.check_is_valid() {
                    return (
                        RayTracingPass::new_invalid(self.clone(), &label, err.map_pass_err(scope)),
                        None,
                    );
                }

                (
                    RayTracingPass::new(self.clone(), RayTracingPassDescriptor { label }),
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
                    RayTracingPass::new_invalid(self.clone(), &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(err @ (SErr::Ended | SErr::Submitted)) => {
                // Attempting to open a new pass after the encode has ended
                // generates an immediate validation error.
                drop(cmd_buf_data);
                (
                    RayTracingPass::new_invalid(
                        self.clone(),
                        &label,
                        err.clone().map_pass_err(scope),
                    ),
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
                    RayTracingPass::new_invalid(self.clone(), &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(SErr::Unlocked) => {
                unreachable!("lock_encoder cannot fail due to the encoder being unlocked")
            }
        }
    }

    pub fn begin_ray_tracing_pass(
        self: &Arc<Self>,
        desc: &RayTracingPassDescriptor<'_>,
    ) -> RayTracingPass {
        let (pass, err) = self.begin_ray_tracing_pass_inner(desc);
        if let Some(err) = err {
            self.device
                .handle_error(err, pass.label(), "CommandEncoder::begin_ray_tracing_pass");
        }
        pass
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
                query_set_writes: parent_state.query_set_writes,
                deferred_query_set_resolves: parent_state.deferred_query_set_resolves,
            },
            binder: Binder::new(),
            temp_offsets: Vec::new(),
            dynamic_offset_count: 0,
            pending_discard_init_fixups: SurfacesInDiscardState::new(),
            scope: device.new_usage_scope(),
            string_offset: 0,

            immediate_state: ImmediateState::default(),
        },
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
            _ => todo!(),
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
            .set_ray_tracing_pipeline(pipeline.raw()?);
    }

    // Rebind resources
    let pipeline_layout = pipeline.layout()?;
    pass::change_pipeline_layout::<RayTracingPassErrorInner>(
        &mut state.pass,
        pipeline_layout,
        &pipeline.late_sized_buffer_groups,
    )
}
