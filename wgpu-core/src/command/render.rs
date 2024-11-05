use crate::binding_model::BindGroup;
use crate::command::{
    validate_and_begin_occlusion_query, validate_and_begin_pipeline_statistics_query,
};
use crate::init_tracker::BufferInitTrackerAction;
use crate::pipeline::RenderPipeline;
use crate::resource::InvalidResourceError;
use crate::snatch::SnatchGuard;
use crate::{
    api_log,
    binding_model::BindError,
    command::{
        bind::Binder,
        end_occlusion_query, end_pipeline_statistics_query,
        memory_init::{fixup_discarded_surfaces, SurfacesInDiscardState},
        ArcPassTimestampWrites, BasePass, BindGroupStateChange, CommandBuffer, CommandEncoderError,
        CommandEncoderStatus, DrawError, ExecutionError, MapPassErr, PassErrorScope,
        PassTimestampWrites, QueryUseError, RenderCommandError, StateChange,
    },
    device::{
        AttachmentData, Device, DeviceError, MissingDownlevelFlags, MissingFeatures,
        RenderPassCompatibilityError, RenderPassContext,
    },
    global::Global,
    hal_label, id,
    init_tracker::{MemoryInitKind, TextureInitRange, TextureInitTrackerAction},
    pipeline::{self, PipelineFlags},
    resource::{
        DestroyedResourceError, Labeled, MissingBufferUsageError, MissingTextureUsageError,
        ParentDevice, QuerySet, Texture, TextureView, TextureViewNotRenderableReason,
    },
    track::{ResourceUsageCompatibilityError, TextureSelector, Tracker, UsageScope},
    Label,
};

use arrayvec::ArrayVec;
use thiserror::Error;
use wgt::{
    BufferAddress, BufferSize, BufferUsages, Color, DynamicOffset, IndexFormat, ShaderStages,
    TextureUsages, TextureViewDimension, VertexStepMode,
};

#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")]
use serde::Serialize;

use std::sync::Arc;
use std::{borrow::Cow, fmt, iter, mem::size_of, num::NonZeroU32, ops::Range, str};

use super::render_command::ArcRenderCommand;
use super::{
    memory_init::TextureSurfaceDiscard, CommandBufferTextureMemoryActions, CommandEncoder,
    QueryResetMap,
};
use super::{DrawKind, Rect};

/// Operation to perform to the output attachment at the start of a renderpass.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum LoadOp {
    /// Clear the output attachment with the clear color. Clearing is faster than loading.
    Clear = 0,
    /// Do not clear output attachment.
    Load = 1,
}

/// Operation to perform to the output attachment at the end of a renderpass.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum StoreOp {
    /// Discards the content of the render target.
    ///
    /// If you don't care about the contents of the target, this can be faster.
    Discard = 0,
    /// Store the result of the renderpass.
    Store = 1,
}

/// Describes an individual channel within a render pass, such as color, depth, or stencil.
#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PassChannel<V> {
    /// Operation to perform to the output attachment at the start of a
    /// renderpass.
    ///
    /// This must be clear if it is the first renderpass rendering to a swap
    /// chain image.
    pub load_op: LoadOp,
    /// Operation to perform to the output attachment at the end of a renderpass.
    pub store_op: StoreOp,
    /// If load_op is [`LoadOp::Clear`], the attachment will be cleared to this
    /// color.
    pub clear_value: V,
    /// If true, the relevant channel is not changed by a renderpass, and the
    /// corresponding attachment can be used inside the pass by other read-only
    /// usages.
    pub read_only: bool,
}

impl<V> PassChannel<V> {
    fn hal_ops(&self) -> hal::AttachmentOps {
        let mut ops = hal::AttachmentOps::empty();
        match self.load_op {
            LoadOp::Load => ops |= hal::AttachmentOps::LOAD,
            LoadOp::Clear => (),
        };
        match self.store_op {
            StoreOp::Store => ops |= hal::AttachmentOps::STORE,
            StoreOp::Discard => (),
        };
        ops
    }
}

/// Describes a color attachment to a render pass.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RenderPassColorAttachment {
    /// The view to use as an attachment.
    pub view: id::TextureViewId,
    /// The view that will receive the resolved output if multisampling is used.
    pub resolve_target: Option<id::TextureViewId>,
    /// What operations will be performed on this color attachment.
    pub channel: PassChannel<Color>,
}

/// Describes a color attachment to a render pass.
#[derive(Debug)]
struct ArcRenderPassColorAttachment {
    /// The view to use as an attachment.
    pub view: Arc<TextureView>,
    /// The view that will receive the resolved output if multisampling is used.
    pub resolve_target: Option<Arc<TextureView>>,
    /// What operations will be performed on this color attachment.
    pub channel: PassChannel<Color>,
}

/// Describes a depth/stencil attachment to a render pass.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RenderPassDepthStencilAttachment {
    /// The view to use as an attachment.
    pub view: id::TextureViewId,
    /// What operations will be performed on the depth part of the attachment.
    pub depth: PassChannel<f32>,
    /// What operations will be performed on the stencil part of the attachment.
    pub stencil: PassChannel<u32>,
}
/// Describes a depth/stencil attachment to a render pass.
#[derive(Debug)]
pub struct ArcRenderPassDepthStencilAttachment {
    /// The view to use as an attachment.
    pub view: Arc<TextureView>,
    /// What operations will be performed on the depth part of the attachment.
    pub depth: PassChannel<f32>,
    /// What operations will be performed on the stencil part of the attachment.
    pub stencil: PassChannel<u32>,
}

impl ArcRenderPassDepthStencilAttachment {
    /// Validate the given aspects' read-only flags against their load
    /// and store ops.
    ///
    /// When an aspect is read-only, its load and store ops must be
    /// `LoadOp::Load` and `StoreOp::Store`.
    ///
    /// On success, return a pair `(depth, stencil)` indicating
    /// whether the depth and stencil passes are read-only.
    fn depth_stencil_read_only(
        &self,
        aspects: hal::FormatAspects,
    ) -> Result<(bool, bool), RenderPassErrorInner> {
        let mut depth_read_only = true;
        let mut stencil_read_only = true;

        if aspects.contains(hal::FormatAspects::DEPTH) {
            if self.depth.read_only
                && (self.depth.load_op, self.depth.store_op) != (LoadOp::Load, StoreOp::Store)
            {
                return Err(RenderPassErrorInner::InvalidDepthOps);
            }
            depth_read_only = self.depth.read_only;
        }

        if aspects.contains(hal::FormatAspects::STENCIL) {
            if self.stencil.read_only
                && (self.stencil.load_op, self.stencil.store_op) != (LoadOp::Load, StoreOp::Store)
            {
                return Err(RenderPassErrorInner::InvalidStencilOps);
            }
            stencil_read_only = self.stencil.read_only;
        }

        Ok((depth_read_only, stencil_read_only))
    }
}

/// Describes the attachments of a render pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderPassDescriptor<'a> {
    pub label: Label<'a>,
    /// The color attachments of the render pass.
    pub color_attachments: Cow<'a, [Option<RenderPassColorAttachment>]>,
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<&'a RenderPassDepthStencilAttachment>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<&'a PassTimestampWrites>,
    /// Defines where the occlusion query results will be stored for this pass.
    pub occlusion_query_set: Option<id::QuerySetId>,
}

/// Describes the attachments of a render pass.
struct ArcRenderPassDescriptor<'a> {
    pub label: &'a Label<'a>,
    /// The color attachments of the render pass.
    pub color_attachments:
        ArrayVec<Option<ArcRenderPassColorAttachment>, { hal::MAX_COLOR_ATTACHMENTS }>,
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<ArcRenderPassDepthStencilAttachment>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<ArcPassTimestampWrites>,
    /// Defines where the occlusion query results will be stored for this pass.
    pub occlusion_query_set: Option<Arc<QuerySet>>,
}

pub struct RenderPass {
    /// All pass data & records is stored here.
    ///
    /// If this is `None`, the pass is in the 'ended' state and can no longer be used.
    /// Any attempt to record more commands will result in a validation error.
    base: Option<BasePass<ArcRenderCommand>>,

    /// Parent command buffer that this pass records commands into.
    ///
    /// If it is none, this pass is invalid and any operation on it will return an error.
    parent: Option<Arc<CommandBuffer>>,

    color_attachments:
        ArrayVec<Option<ArcRenderPassColorAttachment>, { hal::MAX_COLOR_ATTACHMENTS }>,
    depth_stencil_attachment: Option<ArcRenderPassDepthStencilAttachment>,
    timestamp_writes: Option<ArcPassTimestampWrites>,
    occlusion_query_set: Option<Arc<QuerySet>>,

    // Resource binding dedupe state.
    current_bind_groups: BindGroupStateChange,
    current_pipeline: StateChange<id::RenderPipelineId>,
}

impl RenderPass {
    /// If the parent command buffer is invalid, the returned pass will be invalid.
    fn new(parent: Option<Arc<CommandBuffer>>, desc: ArcRenderPassDescriptor) -> Self {
        let ArcRenderPassDescriptor {
            label,
            timestamp_writes,
            color_attachments,
            depth_stencil_attachment,
            occlusion_query_set,
        } = desc;

        Self {
            base: Some(BasePass::new(label)),
            parent,
            color_attachments,
            depth_stencil_attachment,
            timestamp_writes,
            occlusion_query_set,

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
    ) -> Result<&'a mut BasePass<ArcRenderCommand>, RenderPassError> {
        self.base
            .as_mut()
            .ok_or(RenderPassErrorInner::PassEnded)
            .map_pass_err(scope)
    }
}

impl fmt::Debug for RenderPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RenderPass")
            .field("label", &self.label())
            .field("color_attachments", &self.color_attachments)
            .field("depth_stencil_target", &self.depth_stencil_attachment)
            .field(
                "command count",
                &self.base.as_ref().map_or(0, |base| base.commands.len()),
            )
            .field(
                "dynamic offset count",
                &self
                    .base
                    .as_ref()
                    .map_or(0, |base| base.dynamic_offsets.len()),
            )
            .field(
                "push constant u32 count",
                &self
                    .base
                    .as_ref()
                    .map_or(0, |base| base.push_constant_data.len()),
            )
            .finish()
    }
}

#[derive(Debug, PartialEq)]
enum OptionalState {
    Unused,
    Required,
    Set,
}

impl OptionalState {
    fn require(&mut self, require: bool) {
        if require && *self == Self::Unused {
            *self = Self::Required;
        }
    }
}

#[derive(Debug, Default)]
struct IndexState {
    buffer_format: Option<IndexFormat>,
    limit: u64,
}

impl IndexState {
    fn update_buffer(&mut self, range: Range<BufferAddress>, format: IndexFormat) {
        self.buffer_format = Some(format);
        let shift = match format {
            IndexFormat::Uint16 => 1,
            IndexFormat::Uint32 => 2,
        };
        self.limit = (range.end - range.start) >> shift;
    }

    fn reset(&mut self) {
        self.buffer_format = None;
        self.limit = 0;
    }
}

#[derive(Clone, Copy, Debug)]
struct VertexBufferState {
    total_size: BufferAddress,
    step: pipeline::VertexStep,
    bound: bool,
}

impl VertexBufferState {
    const EMPTY: Self = Self {
        total_size: 0,
        step: pipeline::VertexStep {
            stride: 0,
            last_stride: 0,
            mode: VertexStepMode::Vertex,
        },
        bound: false,
    };
}

#[derive(Debug, Default)]
struct VertexState {
    inputs: ArrayVec<VertexBufferState, { hal::MAX_VERTEX_BUFFERS }>,
    /// Length of the shortest vertex rate vertex buffer
    vertex_limit: u64,
    /// Buffer slot which the shortest vertex rate vertex buffer is bound to
    vertex_limit_slot: u32,
    /// Length of the shortest instance rate vertex buffer
    instance_limit: u64,
    /// Buffer slot which the shortest instance rate vertex buffer is bound to
    instance_limit_slot: u32,
}

impl VertexState {
    fn update_limits(&mut self) {
        // Implements the validation from https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-draw
        // Except that the formula is shuffled to extract the number of vertices in order
        // to carry the bulk of the computation when changing states instead of when producing
        // draws. Draw calls tend to happen at a higher frequency. Here we determine vertex
        // limits that can be cheaply checked for each draw call.
        self.vertex_limit = u32::MAX as u64;
        self.instance_limit = u32::MAX as u64;
        for (idx, vbs) in self.inputs.iter().enumerate() {
            if !vbs.bound {
                continue;
            }

            let limit = if vbs.total_size < vbs.step.last_stride {
                // The buffer cannot fit the last vertex.
                0
            } else {
                if vbs.step.stride == 0 {
                    // We already checked that the last stride fits, the same
                    // vertex will be repeated so this slot can accommodate any number of
                    // vertices.
                    continue;
                }

                // The general case.
                (vbs.total_size - vbs.step.last_stride) / vbs.step.stride + 1
            };

            match vbs.step.mode {
                VertexStepMode::Vertex => {
                    if limit < self.vertex_limit {
                        self.vertex_limit = limit;
                        self.vertex_limit_slot = idx as _;
                    }
                }
                VertexStepMode::Instance => {
                    if limit < self.instance_limit {
                        self.instance_limit = limit;
                        self.instance_limit_slot = idx as _;
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        self.inputs.clear();
        self.vertex_limit = 0;
        self.instance_limit = 0;
    }
}

struct State<'scope, 'snatch_guard, 'cmd_buf, 'raw_encoder> {
    pipeline_flags: PipelineFlags,
    binder: Binder,
    blend_constant: OptionalState,
    stencil_reference: u32,
    pipeline: Option<Arc<RenderPipeline>>,
    index: IndexState,
    vertex: VertexState,
    debug_scope_depth: u32,

    info: RenderPassInfo<'scope>,

    snatch_guard: &'snatch_guard SnatchGuard<'snatch_guard>,

    device: &'cmd_buf Arc<Device>,

    raw_encoder: &'raw_encoder mut dyn hal::DynCommandEncoder,

    tracker: &'cmd_buf mut Tracker,
    buffer_memory_init_actions: &'cmd_buf mut Vec<BufferInitTrackerAction>,
    texture_memory_actions: &'cmd_buf mut CommandBufferTextureMemoryActions,

    temp_offsets: Vec<u32>,
    dynamic_offset_count: usize,
    string_offset: usize,

    active_occlusion_query: Option<(Arc<QuerySet>, u32)>,
    active_pipeline_statistics_query: Option<(Arc<QuerySet>, u32)>,
}

impl<'scope, 'snatch_guard, 'cmd_buf, 'raw_encoder>
    State<'scope, 'snatch_guard, 'cmd_buf, 'raw_encoder>
{
    fn is_ready(&self, indexed: bool) -> Result<(), DrawError> {
        if let Some(pipeline) = self.pipeline.as_ref() {
            self.binder.check_compatibility(pipeline.as_ref())?;
            self.binder.check_late_buffer_bindings()?;

            if self.blend_constant == OptionalState::Required {
                return Err(DrawError::MissingBlendConstant);
            }

            // Determine how many vertex buffers have already been bound
            let vertex_buffer_count =
                self.vertex.inputs.iter().take_while(|v| v.bound).count() as u32;
            // Compare with the needed quantity
            if vertex_buffer_count < pipeline.vertex_steps.len() as u32 {
                return Err(DrawError::MissingVertexBuffer {
                    pipeline: pipeline.error_ident(),
                    index: vertex_buffer_count,
                });
            }

            if indexed {
                // Pipeline expects an index buffer
                if let Some(pipeline_index_format) = pipeline.strip_index_format {
                    // We have a buffer bound
                    let buffer_index_format = self
                        .index
                        .buffer_format
                        .ok_or(DrawError::MissingIndexBuffer)?;

                    // The buffers are different formats
                    if pipeline_index_format != buffer_index_format {
                        return Err(DrawError::UnmatchedIndexFormats {
                            pipeline: pipeline.error_ident(),
                            pipeline_format: pipeline_index_format,
                            buffer_format: buffer_index_format,
                        });
                    }
                }
            }
            Ok(())
        } else {
            Err(DrawError::MissingPipeline)
        }
    }

    /// Reset the `RenderBundle`-related states.
    fn reset_bundle(&mut self) {
        self.binder.reset();
        self.pipeline = None;
        self.index.reset();
        self.vertex.reset();
    }
}

/// Describes an attachment location in words.
///
/// Can be used as "the {loc} has..." or "{loc} has..."
#[derive(Debug, Copy, Clone)]
pub enum AttachmentErrorLocation {
    Color { index: usize, resolve: bool },
    Depth,
}

impl fmt::Display for AttachmentErrorLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            AttachmentErrorLocation::Color {
                index,
                resolve: false,
            } => write!(f, "color attachment at index {index}'s texture view"),
            AttachmentErrorLocation::Color {
                index,
                resolve: true,
            } => write!(
                f,
                "color attachment at index {index}'s resolve texture view"
            ),
            AttachmentErrorLocation::Depth => write!(f, "depth attachment's texture view"),
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ColorAttachmentError {
    #[error("Attachment format {0:?} is not a color format")]
    InvalidFormat(wgt::TextureFormat),
    #[error("The number of color attachments {given} exceeds the limit {limit}")]
    TooMany { given: usize, limit: usize },
    #[error("The total number of bytes per sample in color attachments {total} exceeds the limit {limit}")]
    TooManyBytesPerSample { total: u32, limit: u32 },
}

/// Error encountered when performing a render pass.
#[derive(Clone, Debug, Error)]
pub enum RenderPassErrorInner {
    #[error(transparent)]
    Device(DeviceError),
    #[error(transparent)]
    ColorAttachment(#[from] ColorAttachmentError),
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),
    #[error("Parent encoder is invalid")]
    InvalidParentEncoder,
    #[error("The format of the depth-stencil attachment ({0:?}) is not a depth-stencil format")]
    InvalidDepthStencilAttachmentFormat(wgt::TextureFormat),
    #[error("The format of the {location} ({format:?}) is not resolvable")]
    UnsupportedResolveTargetFormat {
        location: AttachmentErrorLocation,
        format: wgt::TextureFormat,
    },
    #[error("No color attachments or depth attachments were provided, at least one attachment of any kind must be provided")]
    MissingAttachments,
    #[error("The {location} is not renderable:")]
    TextureViewIsNotRenderable {
        location: AttachmentErrorLocation,
        #[source]
        reason: TextureViewNotRenderableReason,
    },
    #[error("Attachments have differing sizes: the {expected_location} has extent {expected_extent:?} but is followed by the {actual_location} which has {actual_extent:?}")]
    AttachmentsDimensionMismatch {
        expected_location: AttachmentErrorLocation,
        expected_extent: wgt::Extent3d,
        actual_location: AttachmentErrorLocation,
        actual_extent: wgt::Extent3d,
    },
    #[error("Attachments have differing sample counts: the {expected_location} has count {expected_samples:?} but is followed by the {actual_location} which has count {actual_samples:?}")]
    AttachmentSampleCountMismatch {
        expected_location: AttachmentErrorLocation,
        expected_samples: u32,
        actual_location: AttachmentErrorLocation,
        actual_samples: u32,
    },
    #[error("The resolve source, {location}, must be multi-sampled (has {src} samples) while the resolve destination must not be multisampled (has {dst} samples)")]
    InvalidResolveSampleCounts {
        location: AttachmentErrorLocation,
        src: u32,
        dst: u32,
    },
    #[error(
        "Resource source, {location}, format ({src:?}) must match the resolve destination format ({dst:?})"
    )]
    MismatchedResolveTextureFormat {
        location: AttachmentErrorLocation,
        src: wgt::TextureFormat,
        dst: wgt::TextureFormat,
    },
    #[error("Surface texture is dropped before the render pass is finished")]
    SurfaceTextureDropped,
    #[error("Not enough memory left for render pass")]
    OutOfMemory,
    #[error("Unable to clear non-present/read-only depth")]
    InvalidDepthOps,
    #[error("Unable to clear non-present/read-only stencil")]
    InvalidStencilOps,
    #[error("Setting `values_offset` to be `None` is only for internal use in render bundles")]
    InvalidValuesOffset,
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("Indirect buffer offset {0:?} is not a multiple of 4")]
    UnalignedIndirectBufferOffset(BufferAddress),
    #[error("Indirect draw uses bytes {offset}..{end_offset} {} which overruns indirect buffer of size {buffer_size}",
        count.map_or_else(String::new, |v| format!("(using count {v})")))]
    IndirectBufferOverrun {
        count: Option<NonZeroU32>,
        offset: u64,
        end_offset: u64,
        buffer_size: u64,
    },
    #[error("Indirect draw uses bytes {begin_count_offset}..{end_count_offset} which overruns indirect buffer of size {count_buffer_size}")]
    IndirectCountBufferOverrun {
        begin_count_offset: u64,
        end_count_offset: u64,
        count_buffer_size: u64,
    },
    #[error("Cannot pop debug group, because number of pushed debug groups is zero")]
    InvalidPopDebugGroup,
    #[error(transparent)]
    ResourceUsageCompatibility(#[from] ResourceUsageCompatibilityError),
    #[error("Render bundle has incompatible targets, {0}")]
    IncompatibleBundleTargets(#[from] RenderPassCompatibilityError),
    #[error(
        "Render bundle has incompatible read-only flags: \
             bundle has flags depth = {bundle_depth} and stencil = {bundle_stencil}, \
             while the pass has flags depth = {pass_depth} and stencil = {pass_stencil}. \
             Read-only renderpasses are only compatible with read-only bundles for that aspect."
    )]
    IncompatibleBundleReadOnlyDepthStencil {
        pass_depth: bool,
        pass_stencil: bool,
        bundle_depth: bool,
        bundle_stencil: bool,
    },
    #[error(transparent)]
    RenderCommand(#[from] RenderCommandError),
    #[error(transparent)]
    Draw(#[from] DrawError),
    #[error(transparent)]
    Bind(#[from] BindError),
    #[error("Push constant offset must be aligned to 4 bytes")]
    PushConstantOffsetAlignment,
    #[error("Push constant size must be aligned to 4 bytes")]
    PushConstantSizeAlignment,
    #[error("Ran out of push constant space. Don't set 4gb of push constants per ComputePass.")]
    PushConstantOutOfMemory,
    #[error(transparent)]
    QueryUse(#[from] QueryUseError),
    #[error("Multiview layer count must match")]
    MultiViewMismatch,
    #[error(
        "Multiview pass texture views with more than one array layer must have D2Array dimension"
    )]
    MultiViewDimensionMismatch,
    #[error("missing occlusion query set")]
    MissingOcclusionQuerySet,
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error("The compute pass has already been ended and no further commands can be recorded")]
    PassEnded,
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

impl From<MissingBufferUsageError> for RenderPassErrorInner {
    fn from(error: MissingBufferUsageError) -> Self {
        Self::RenderCommand(error.into())
    }
}

impl From<MissingTextureUsageError> for RenderPassErrorInner {
    fn from(error: MissingTextureUsageError) -> Self {
        Self::RenderCommand(error.into())
    }
}

impl From<DeviceError> for RenderPassErrorInner {
    fn from(error: DeviceError) -> Self {
        Self::Device(error)
    }
}

/// Error encountered when performing a render pass.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct RenderPassError {
    pub scope: PassErrorScope,
    #[source]
    pub(super) inner: RenderPassErrorInner,
}

impl<T, E> MapPassErr<T, RenderPassError> for Result<T, E>
where
    E: Into<RenderPassErrorInner>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> Result<T, RenderPassError> {
        self.map_err(|inner| RenderPassError {
            scope,
            inner: inner.into(),
        })
    }
}

struct RenderAttachment {
    texture: Arc<Texture>,
    selector: TextureSelector,
    usage: hal::TextureUses,
}

impl TextureView {
    fn to_render_attachment(&self, usage: hal::TextureUses) -> RenderAttachment {
        RenderAttachment {
            texture: self.parent.clone(),
            selector: self.selector.clone(),
            usage,
        }
    }
}

const MAX_TOTAL_ATTACHMENTS: usize = hal::MAX_COLOR_ATTACHMENTS + hal::MAX_COLOR_ATTACHMENTS + 1;
type AttachmentDataVec<T> = ArrayVec<T, MAX_TOTAL_ATTACHMENTS>;

struct RenderPassInfo<'d> {
    context: RenderPassContext,
    usage_scope: UsageScope<'d>,
    /// All render attachments, including depth/stencil
    render_attachments: AttachmentDataVec<RenderAttachment>,
    is_depth_read_only: bool,
    is_stencil_read_only: bool,
    extent: wgt::Extent3d,

    pending_discard_init_fixups: SurfacesInDiscardState,
    divergent_discarded_depth_stencil_aspect: Option<(wgt::TextureAspect, Arc<TextureView>)>,
    multiview: Option<NonZeroU32>,
}

impl<'d> RenderPassInfo<'d> {
    fn add_pass_texture_init_actions<V>(
        channel: &PassChannel<V>,
        texture_memory_actions: &mut CommandBufferTextureMemoryActions,
        view: &TextureView,
        pending_discard_init_fixups: &mut SurfacesInDiscardState,
    ) {
        if channel.load_op == LoadOp::Load {
            pending_discard_init_fixups.extend(texture_memory_actions.register_init_action(
                &TextureInitTrackerAction {
                    texture: view.parent.clone(),
                    range: TextureInitRange::from(view.selector.clone()),
                    // Note that this is needed even if the target is discarded,
                    kind: MemoryInitKind::NeedsInitializedMemory,
                },
            ));
        } else if channel.store_op == StoreOp::Store {
            // Clear + Store
            texture_memory_actions.register_implicit_init(
                &view.parent,
                TextureInitRange::from(view.selector.clone()),
            );
        }
        if channel.store_op == StoreOp::Discard {
            // the discard happens at the *end* of a pass, but recording the
            // discard right away be alright since the texture can't be used
            // during the pass anyways
            texture_memory_actions.discard(TextureSurfaceDiscard {
                texture: view.parent.clone(),
                mip_level: view.selector.mips.start,
                layer: view.selector.layers.start,
            });
        }
    }

    fn start(
        device: &'d Arc<Device>,
        hal_label: Option<&str>,
        color_attachments: ArrayVec<
            Option<ArcRenderPassColorAttachment>,
            { hal::MAX_COLOR_ATTACHMENTS },
        >,
        mut depth_stencil_attachment: Option<ArcRenderPassDepthStencilAttachment>,
        mut timestamp_writes: Option<ArcPassTimestampWrites>,
        mut occlusion_query_set: Option<Arc<QuerySet>>,
        encoder: &mut CommandEncoder,
        trackers: &mut Tracker,
        texture_memory_actions: &mut CommandBufferTextureMemoryActions,
        pending_query_resets: &mut QueryResetMap,
        snatch_guard: &SnatchGuard<'_>,
    ) -> Result<Self, RenderPassErrorInner> {
        profiling::scope!("RenderPassInfo::start");

        // We default to false intentionally, even if depth-stencil isn't used at all.
        // This allows us to use the primary raw pipeline in `RenderPipeline`,
        // instead of the special read-only one, which would be `None`.
        let mut is_depth_read_only = false;
        let mut is_stencil_read_only = false;

        let mut render_attachments = AttachmentDataVec::<RenderAttachment>::new();
        let mut discarded_surfaces = AttachmentDataVec::new();
        let mut pending_discard_init_fixups = SurfacesInDiscardState::new();
        let mut divergent_discarded_depth_stencil_aspect = None;

        let mut attachment_location = AttachmentErrorLocation::Color {
            index: usize::MAX,
            resolve: false,
        };
        let mut extent = None;
        let mut sample_count = 0;

        let mut detected_multiview: Option<Option<NonZeroU32>> = None;

        let mut check_multiview = |view: &TextureView| {
            // Get the multiview configuration for this texture view
            let layers = view.selector.layers.end - view.selector.layers.start;
            let this_multiview = if layers >= 2 {
                // Trivially proven by the if above
                Some(unsafe { NonZeroU32::new_unchecked(layers) })
            } else {
                None
            };

            // Make sure that if this view is a multiview, it is set to be an array
            if this_multiview.is_some() && view.desc.dimension != TextureViewDimension::D2Array {
                return Err(RenderPassErrorInner::MultiViewDimensionMismatch);
            }

            // Validate matching first, or store the first one
            if let Some(multiview) = detected_multiview {
                if multiview != this_multiview {
                    return Err(RenderPassErrorInner::MultiViewMismatch);
                }
            } else {
                // Multiview is only supported if the feature is enabled
                if this_multiview.is_some() {
                    device.require_features(wgt::Features::MULTIVIEW)?;
                }

                detected_multiview = Some(this_multiview);
            }

            Ok(())
        };
        let mut add_view = |view: &TextureView, location| {
            let render_extent = view.render_extent.map_err(|reason| {
                RenderPassErrorInner::TextureViewIsNotRenderable { location, reason }
            })?;
            if let Some(ex) = extent {
                if ex != render_extent {
                    return Err(RenderPassErrorInner::AttachmentsDimensionMismatch {
                        expected_location: attachment_location,
                        expected_extent: ex,
                        actual_location: location,
                        actual_extent: render_extent,
                    });
                }
            } else {
                extent = Some(render_extent);
            }
            if sample_count == 0 {
                sample_count = view.samples;
            } else if sample_count != view.samples {
                return Err(RenderPassErrorInner::AttachmentSampleCountMismatch {
                    expected_location: attachment_location,
                    expected_samples: sample_count,
                    actual_location: location,
                    actual_samples: view.samples,
                });
            }
            attachment_location = location;
            Ok(())
        };

        let mut depth_stencil = None;

        if let Some(at) = depth_stencil_attachment.as_ref() {
            let view = &at.view;
            view.same_device(device)?;
            check_multiview(view)?;
            add_view(view, AttachmentErrorLocation::Depth)?;

            let ds_aspects = view.desc.aspects();
            if ds_aspects.contains(hal::FormatAspects::COLOR) {
                return Err(RenderPassErrorInner::InvalidDepthStencilAttachmentFormat(
                    view.desc.format,
                ));
            }

            if !ds_aspects.contains(hal::FormatAspects::STENCIL)
                || (at.stencil.load_op == at.depth.load_op
                    && at.stencil.store_op == at.depth.store_op)
            {
                Self::add_pass_texture_init_actions(
                    &at.depth,
                    texture_memory_actions,
                    view,
                    &mut pending_discard_init_fixups,
                );
            } else if !ds_aspects.contains(hal::FormatAspects::DEPTH) {
                Self::add_pass_texture_init_actions(
                    &at.stencil,
                    texture_memory_actions,
                    view,
                    &mut pending_discard_init_fixups,
                );
            } else {
                // This is the only place (anywhere in wgpu) where Stencil &
                // Depth init state can diverge.
                //
                // To safe us the overhead of tracking init state of texture
                // aspects everywhere, we're going to cheat a little bit in
                // order to keep the init state of both Stencil and Depth
                // aspects in sync. The expectation is that we hit this path
                // extremely rarely!
                //
                // Diverging LoadOp, i.e. Load + Clear:
                //
                // Record MemoryInitKind::NeedsInitializedMemory for the entire
                // surface, a bit wasteful on unit but no negative effect!
                //
                // Rationale: If the loaded channel is uninitialized it needs
                // clearing, the cleared channel doesn't care. (If everything is
                // already initialized nothing special happens)
                //
                // (possible minor optimization: Clear caused by
                // NeedsInitializedMemory should know that it doesn't need to
                // clear the aspect that was set to C)
                let need_init_beforehand =
                    at.depth.load_op == LoadOp::Load || at.stencil.load_op == LoadOp::Load;
                if need_init_beforehand {
                    pending_discard_init_fixups.extend(
                        texture_memory_actions.register_init_action(&TextureInitTrackerAction {
                            texture: view.parent.clone(),
                            range: TextureInitRange::from(view.selector.clone()),
                            kind: MemoryInitKind::NeedsInitializedMemory,
                        }),
                    );
                }

                // Diverging Store, i.e. Discard + Store:
                //
                // Immediately zero out channel that is set to discard after
                // we're done with the render pass. This allows us to set the
                // entire surface to MemoryInitKind::ImplicitlyInitialized (if
                // it isn't already set to NeedsInitializedMemory).
                //
                // (possible optimization: Delay and potentially drop this zeroing)
                if at.depth.store_op != at.stencil.store_op {
                    if !need_init_beforehand {
                        texture_memory_actions.register_implicit_init(
                            &view.parent,
                            TextureInitRange::from(view.selector.clone()),
                        );
                    }
                    divergent_discarded_depth_stencil_aspect = Some((
                        if at.depth.store_op == StoreOp::Discard {
                            wgt::TextureAspect::DepthOnly
                        } else {
                            wgt::TextureAspect::StencilOnly
                        },
                        view.clone(),
                    ));
                } else if at.depth.store_op == StoreOp::Discard {
                    // Both are discarded using the regular path.
                    discarded_surfaces.push(TextureSurfaceDiscard {
                        texture: view.parent.clone(),
                        mip_level: view.selector.mips.start,
                        layer: view.selector.layers.start,
                    });
                }
            }

            (is_depth_read_only, is_stencil_read_only) = at.depth_stencil_read_only(ds_aspects)?;

            let usage = if is_depth_read_only
                && is_stencil_read_only
                && device
                    .downlevel
                    .flags
                    .contains(wgt::DownlevelFlags::READ_ONLY_DEPTH_STENCIL)
            {
                hal::TextureUses::DEPTH_STENCIL_READ | hal::TextureUses::RESOURCE
            } else {
                hal::TextureUses::DEPTH_STENCIL_WRITE
            };
            render_attachments.push(view.to_render_attachment(usage));

            depth_stencil = Some(hal::DepthStencilAttachment {
                target: hal::Attachment {
                    view: view.try_raw(snatch_guard)?,
                    usage,
                },
                depth_ops: at.depth.hal_ops(),
                stencil_ops: at.stencil.hal_ops(),
                clear_value: (at.depth.clear_value, at.stencil.clear_value),
            });
        }

        let mut color_attachments_hal =
            ArrayVec::<Option<hal::ColorAttachment<_>>, { hal::MAX_COLOR_ATTACHMENTS }>::new();
        for (index, attachment) in color_attachments.iter().enumerate() {
            let at = if let Some(attachment) = attachment.as_ref() {
                attachment
            } else {
                color_attachments_hal.push(None);
                continue;
            };
            let color_view: &TextureView = &at.view;
            color_view.same_device(device)?;
            check_multiview(color_view)?;
            add_view(
                color_view,
                AttachmentErrorLocation::Color {
                    index,
                    resolve: false,
                },
            )?;

            if !color_view
                .desc
                .aspects()
                .contains(hal::FormatAspects::COLOR)
            {
                return Err(RenderPassErrorInner::ColorAttachment(
                    ColorAttachmentError::InvalidFormat(color_view.desc.format),
                ));
            }

            Self::add_pass_texture_init_actions(
                &at.channel,
                texture_memory_actions,
                color_view,
                &mut pending_discard_init_fixups,
            );
            render_attachments
                .push(color_view.to_render_attachment(hal::TextureUses::COLOR_TARGET));

            let mut hal_resolve_target = None;
            if let Some(resolve_view) = &at.resolve_target {
                resolve_view.same_device(device)?;
                check_multiview(resolve_view)?;

                let resolve_location = AttachmentErrorLocation::Color {
                    index,
                    resolve: true,
                };

                let render_extent = resolve_view.render_extent.map_err(|reason| {
                    RenderPassErrorInner::TextureViewIsNotRenderable {
                        location: resolve_location,
                        reason,
                    }
                })?;
                if color_view.render_extent.unwrap() != render_extent {
                    return Err(RenderPassErrorInner::AttachmentsDimensionMismatch {
                        expected_location: attachment_location,
                        expected_extent: extent.unwrap_or_default(),
                        actual_location: resolve_location,
                        actual_extent: render_extent,
                    });
                }
                if color_view.samples == 1 || resolve_view.samples != 1 {
                    return Err(RenderPassErrorInner::InvalidResolveSampleCounts {
                        location: resolve_location,
                        src: color_view.samples,
                        dst: resolve_view.samples,
                    });
                }
                if color_view.desc.format != resolve_view.desc.format {
                    return Err(RenderPassErrorInner::MismatchedResolveTextureFormat {
                        location: resolve_location,
                        src: color_view.desc.format,
                        dst: resolve_view.desc.format,
                    });
                }
                if !resolve_view
                    .format_features
                    .flags
                    .contains(wgt::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE)
                {
                    return Err(RenderPassErrorInner::UnsupportedResolveTargetFormat {
                        location: resolve_location,
                        format: resolve_view.desc.format,
                    });
                }

                texture_memory_actions.register_implicit_init(
                    &resolve_view.parent,
                    TextureInitRange::from(resolve_view.selector.clone()),
                );
                render_attachments
                    .push(resolve_view.to_render_attachment(hal::TextureUses::COLOR_TARGET));

                hal_resolve_target = Some(hal::Attachment {
                    view: resolve_view.try_raw(snatch_guard)?,
                    usage: hal::TextureUses::COLOR_TARGET,
                });
            }

            color_attachments_hal.push(Some(hal::ColorAttachment {
                target: hal::Attachment {
                    view: color_view.try_raw(snatch_guard)?,
                    usage: hal::TextureUses::COLOR_TARGET,
                },
                resolve_target: hal_resolve_target,
                ops: at.channel.hal_ops(),
                clear_value: at.channel.clear_value,
            }));
        }

        let extent = extent.ok_or(RenderPassErrorInner::MissingAttachments)?;
        let multiview = detected_multiview.expect("Multiview was not detected, no attachments");

        let attachment_formats = AttachmentData {
            colors: color_attachments
                .iter()
                .map(|at| at.as_ref().map(|at| at.view.desc.format))
                .collect(),
            resolves: color_attachments
                .iter()
                .filter_map(|at| {
                    at.as_ref().and_then(|at| {
                        at.resolve_target
                            .as_ref()
                            .map(|resolve| resolve.desc.format)
                    })
                })
                .collect(),
            depth_stencil: depth_stencil_attachment
                .as_ref()
                .map(|at| at.view.desc.format),
        };

        let context = RenderPassContext {
            attachments: attachment_formats,
            sample_count,
            multiview,
        };

        let timestamp_writes_hal = if let Some(tw) = timestamp_writes.as_ref() {
            let query_set = &tw.query_set;
            query_set.same_device(device)?;

            if let Some(index) = tw.beginning_of_pass_write_index {
                pending_query_resets.use_query_set(query_set, index);
            }
            if let Some(index) = tw.end_of_pass_write_index {
                pending_query_resets.use_query_set(query_set, index);
            }

            Some(hal::PassTimestampWrites {
                query_set: query_set.raw(),
                beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                end_of_pass_write_index: tw.end_of_pass_write_index,
            })
        } else {
            None
        };

        let occlusion_query_set_hal = if let Some(query_set) = occlusion_query_set.as_ref() {
            query_set.same_device(device)?;
            Some(query_set.raw())
        } else {
            None
        };

        let hal_desc = hal::RenderPassDescriptor {
            label: hal_label,
            extent,
            sample_count,
            color_attachments: &color_attachments_hal,
            depth_stencil_attachment: depth_stencil,
            multiview,
            timestamp_writes: timestamp_writes_hal,
            occlusion_query_set: occlusion_query_set_hal,
        };
        unsafe {
            encoder.raw.begin_render_pass(&hal_desc);
        };
        drop(color_attachments_hal); // Drop, so we can consume `color_attachments` for the tracker.

        // Can't borrow the tracker more than once, so have to add to the tracker after the `begin_render_pass` hal call.
        if let Some(tw) = timestamp_writes.take() {
            trackers.query_sets.insert_single(tw.query_set);
        };
        if let Some(occlusion_query_set) = occlusion_query_set.take() {
            trackers.query_sets.insert_single(occlusion_query_set);
        };
        if let Some(at) = depth_stencil_attachment.take() {
            trackers.views.insert_single(at.view.clone());
        }
        for at in color_attachments.into_iter().flatten() {
            trackers.views.insert_single(at.view.clone());
            if let Some(resolve_target) = at.resolve_target {
                trackers.views.insert_single(resolve_target);
            }
        }

        Ok(Self {
            context,
            usage_scope: device.new_usage_scope(),
            render_attachments,
            is_depth_read_only,
            is_stencil_read_only,
            extent,
            pending_discard_init_fixups,
            divergent_discarded_depth_stencil_aspect,
            multiview,
        })
    }

    fn finish(
        mut self,
        raw: &mut dyn hal::DynCommandEncoder,
        snatch_guard: &SnatchGuard,
    ) -> Result<(UsageScope<'d>, SurfacesInDiscardState), RenderPassErrorInner> {
        profiling::scope!("RenderPassInfo::finish");
        unsafe {
            raw.end_render_pass();
        }

        for ra in self.render_attachments {
            let texture = &ra.texture;
            texture.check_usage(TextureUsages::RENDER_ATTACHMENT)?;

            // the tracker set of the pass is always in "extend" mode
            unsafe {
                self.usage_scope.textures.merge_single(
                    texture,
                    Some(ra.selector.clone()),
                    ra.usage,
                )?
            };
        }

        // If either only stencil or depth was discarded, we put in a special
        // clear pass to keep the init status of the aspects in sync. We do this
        // so we don't need to track init state for depth/stencil aspects
        // individually.
        //
        // Note that we don't go the usual route of "brute force" initializing
        // the texture when need arises here, since this path is actually
        // something a user may genuinely want (where as the other cases are
        // more seen along the lines as gracefully handling a user error).
        if let Some((aspect, view)) = self.divergent_discarded_depth_stencil_aspect {
            let (depth_ops, stencil_ops) = if aspect == wgt::TextureAspect::DepthOnly {
                (
                    hal::AttachmentOps::STORE,                            // clear depth
                    hal::AttachmentOps::LOAD | hal::AttachmentOps::STORE, // unchanged stencil
                )
            } else {
                (
                    hal::AttachmentOps::LOAD | hal::AttachmentOps::STORE, // unchanged stencil
                    hal::AttachmentOps::STORE,                            // clear depth
                )
            };
            let desc = hal::RenderPassDescriptor::<'_, _, dyn hal::DynTextureView> {
                label: Some("(wgpu internal) Zero init discarded depth/stencil aspect"),
                extent: view.render_extent.unwrap(),
                sample_count: view.samples,
                color_attachments: &[],
                depth_stencil_attachment: Some(hal::DepthStencilAttachment {
                    target: hal::Attachment {
                        view: view.try_raw(snatch_guard)?,
                        usage: hal::TextureUses::DEPTH_STENCIL_WRITE,
                    },
                    depth_ops,
                    stencil_ops,
                    clear_value: (0.0, 0),
                }),
                multiview: self.multiview,
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            unsafe {
                raw.begin_render_pass(&desc);
                raw.end_render_pass();
            }
        }

        Ok((self.usage_scope, self.pending_discard_init_fixups))
    }
}

impl Global {
    /// Creates a render pass.
    ///
    /// If creation fails, an invalid pass is returned.
    /// Any operation on an invalid pass will return an error.
    ///
    /// If successful, puts the encoder into the [`CommandEncoderStatus::Locked`] state.
    pub fn command_encoder_create_render_pass(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &RenderPassDescriptor<'_>,
    ) -> (RenderPass, Option<CommandEncoderError>) {
        fn fill_arc_desc(
            hub: &crate::hub::Hub,
            desc: &RenderPassDescriptor<'_>,
            arc_desc: &mut ArcRenderPassDescriptor,
            device: &Device,
        ) -> Result<(), CommandEncoderError> {
            let query_sets = hub.query_sets.read();
            let texture_views = hub.texture_views.read();

            let max_color_attachments = device.limits.max_color_attachments as usize;
            if desc.color_attachments.len() > max_color_attachments {
                return Err(CommandEncoderError::InvalidColorAttachment(
                    ColorAttachmentError::TooMany {
                        given: desc.color_attachments.len(),
                        limit: max_color_attachments,
                    },
                ));
            }

            for color_attachment in desc.color_attachments.iter() {
                if let Some(RenderPassColorAttachment {
                    view: view_id,
                    resolve_target,
                    channel,
                }) = color_attachment
                {
                    let view = texture_views.get(*view_id).get()?;
                    view.same_device(device)?;

                    let resolve_target = if let Some(resolve_target_id) = resolve_target {
                        let rt_arc = texture_views.get(*resolve_target_id).get()?;
                        rt_arc.same_device(device)?;

                        Some(rt_arc)
                    } else {
                        None
                    };

                    arc_desc
                        .color_attachments
                        .push(Some(ArcRenderPassColorAttachment {
                            view,
                            resolve_target,
                            channel: channel.clone(),
                        }));
                } else {
                    arc_desc.color_attachments.push(None);
                }
            }

            arc_desc.depth_stencil_attachment =
                if let Some(depth_stencil_attachment) = desc.depth_stencil_attachment {
                    let view = texture_views.get(depth_stencil_attachment.view).get()?;
                    view.same_device(device)?;

                    Some(ArcRenderPassDepthStencilAttachment {
                        view,
                        depth: depth_stencil_attachment.depth.clone(),
                        stencil: depth_stencil_attachment.stencil.clone(),
                    })
                } else {
                    None
                };

            arc_desc.timestamp_writes = if let Some(tw) = desc.timestamp_writes {
                let query_set = query_sets.get(tw.query_set).get()?;
                query_set.same_device(device)?;

                device.require_features(wgt::Features::TIMESTAMP_QUERY)?;

                Some(ArcPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                })
            } else {
                None
            };

            arc_desc.occlusion_query_set =
                if let Some(occlusion_query_set) = desc.occlusion_query_set {
                    let query_set = query_sets.get(occlusion_query_set).get()?;
                    query_set.same_device(device)?;

                    Some(query_set)
                } else {
                    None
                };

            Ok(())
        }

        let hub = &self.hub;
        let mut arc_desc = ArcRenderPassDescriptor {
            label: &desc.label,
            timestamp_writes: None,
            color_attachments: ArrayVec::new(),
            depth_stencil_attachment: None,
            occlusion_query_set: None,
        };

        let make_err = |e, arc_desc| (RenderPass::new(None, arc_desc), Some(e));

        let cmd_buf = hub.command_buffers.get(encoder_id.into_command_buffer_id());

        match cmd_buf
            .try_get()
            .map_err(|e| e.into())
            .and_then(|mut cmd_buf_data| cmd_buf_data.lock_encoder())
        {
            Ok(_) => {}
            Err(e) => return make_err(e, arc_desc),
        };

        let err = fill_arc_desc(hub, desc, &mut arc_desc, &cmd_buf.device).err();

        (RenderPass::new(Some(cmd_buf), arc_desc), err)
    }

    /// Note that this differs from [`Self::render_pass_end`], it will
    /// create a new pass, replay the commands and end the pass.
    #[doc(hidden)]
    #[cfg(any(feature = "serde", feature = "replay"))]
    pub fn render_pass_end_with_unresolved_commands(
        &self,
        encoder_id: id::CommandEncoderId,
        base: BasePass<super::RenderCommand>,
        color_attachments: &[Option<RenderPassColorAttachment>],
        depth_stencil_attachment: Option<&RenderPassDepthStencilAttachment>,
        timestamp_writes: Option<&PassTimestampWrites>,
        occlusion_query_set: Option<id::QuerySetId>,
    ) -> Result<(), RenderPassError> {
        let pass_scope = PassErrorScope::Pass;

        #[cfg(feature = "trace")]
        {
            let cmd_buf = self
                .hub
                .command_buffers
                .get(encoder_id.into_command_buffer_id());
            let mut cmd_buf_data = cmd_buf.try_get().map_pass_err(pass_scope)?;

            if let Some(ref mut list) = cmd_buf_data.commands {
                list.push(crate::device::trace::Command::RunRenderPass {
                    base: BasePass {
                        label: base.label.clone(),
                        commands: base.commands.clone(),
                        dynamic_offsets: base.dynamic_offsets.clone(),
                        string_data: base.string_data.clone(),
                        push_constant_data: base.push_constant_data.clone(),
                    },
                    target_colors: color_attachments.to_vec(),
                    target_depth_stencil: depth_stencil_attachment.cloned(),
                    timestamp_writes: timestamp_writes.cloned(),
                    occlusion_query_set_id: occlusion_query_set,
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

        let (mut render_pass, encoder_error) = self.command_encoder_create_render_pass(
            encoder_id,
            &RenderPassDescriptor {
                label: label.as_deref().map(Cow::Borrowed),
                color_attachments: Cow::Borrowed(color_attachments),
                depth_stencil_attachment,
                timestamp_writes,
                occlusion_query_set,
            },
        );
        if let Some(err) = encoder_error {
            return Err(RenderPassError {
                scope: pass_scope,
                inner: err.into(),
            });
        };

        render_pass.base = Some(BasePass {
            label,
            commands: super::RenderCommand::resolve_render_command_ids(&self.hub, &commands)?,
            dynamic_offsets,
            string_data,
            push_constant_data,
        });

        self.render_pass_end(&mut render_pass)
    }

    pub fn render_pass_end(&self, pass: &mut RenderPass) -> Result<(), RenderPassError> {
        let pass_scope = PassErrorScope::Pass;

        let cmd_buf = pass
            .parent
            .as_ref()
            .ok_or(RenderPassErrorInner::InvalidParentEncoder)
            .map_pass_err(pass_scope)?;

        let base = pass
            .base
            .take()
            .ok_or(RenderPassErrorInner::PassEnded)
            .map_pass_err(pass_scope)?;

        profiling::scope!(
            "CommandEncoder::run_render_pass {}",
            base.label.as_deref().unwrap_or("")
        );

        let mut cmd_buf_data = cmd_buf.try_get().map_pass_err(pass_scope)?;
        cmd_buf_data.unlock_encoder().map_pass_err(pass_scope)?;
        let cmd_buf_data = &mut *cmd_buf_data;

        let device = &cmd_buf.device;
        let snatch_guard = &device.snatchable_lock.read();

        let hal_label = hal_label(base.label.as_deref(), device.instance_flags);

        let (scope, pending_discard_init_fixups) = {
            device.check_is_valid().map_pass_err(pass_scope)?;

            let encoder = &mut cmd_buf_data.encoder;
            let status = &mut cmd_buf_data.status;
            let tracker = &mut cmd_buf_data.trackers;
            let buffer_memory_init_actions = &mut cmd_buf_data.buffer_memory_init_actions;
            let texture_memory_actions = &mut cmd_buf_data.texture_memory_actions;
            let pending_query_resets = &mut cmd_buf_data.pending_query_resets;

            // We automatically keep extending command buffers over time, and because
            // we want to insert a command buffer _before_ what we're about to record,
            // we need to make sure to close the previous one.
            encoder.close(&cmd_buf.device).map_pass_err(pass_scope)?;
            // We will reset this to `Recording` if we succeed, acts as a fail-safe.
            *status = CommandEncoderStatus::Error;
            encoder
                .open_pass(hal_label, &cmd_buf.device)
                .map_pass_err(pass_scope)?;

            let info = RenderPassInfo::start(
                device,
                hal_label,
                pass.color_attachments.take(),
                pass.depth_stencil_attachment.take(),
                pass.timestamp_writes.take(),
                // Still needed down the line.
                // TODO(wumpf): by restructuring the code, we could get rid of some of this Arc clone.
                pass.occlusion_query_set.clone(),
                encoder,
                tracker,
                texture_memory_actions,
                pending_query_resets,
                snatch_guard,
            )
            .map_pass_err(pass_scope)?;

            let indices = &device.tracker_indices;
            tracker.buffers.set_size(indices.buffers.size());
            tracker.textures.set_size(indices.textures.size());

            let mut state = State {
                pipeline_flags: PipelineFlags::empty(),
                binder: Binder::new(),
                blend_constant: OptionalState::Unused,
                stencil_reference: 0,
                pipeline: None,
                index: IndexState::default(),
                vertex: VertexState::default(),
                debug_scope_depth: 0,

                info,

                snatch_guard,

                device,
                raw_encoder: encoder.raw.as_mut(),
                tracker,
                buffer_memory_init_actions,
                texture_memory_actions,

                temp_offsets: Vec::new(),
                dynamic_offset_count: 0,
                string_offset: 0,

                active_occlusion_query: None,
                active_pipeline_statistics_query: None,
            };

            for command in base.commands {
                match command {
                    ArcRenderCommand::SetBindGroup {
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
                    ArcRenderCommand::SetPipeline(pipeline) => {
                        let scope = PassErrorScope::SetPipelineRender;
                        set_pipeline(&mut state, cmd_buf, pipeline).map_pass_err(scope)?;
                    }
                    ArcRenderCommand::SetIndexBuffer {
                        buffer,
                        index_format,
                        offset,
                        size,
                    } => {
                        let scope = PassErrorScope::SetIndexBuffer;
                        set_index_buffer(&mut state, cmd_buf, buffer, index_format, offset, size)
                            .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::SetVertexBuffer {
                        slot,
                        buffer,
                        offset,
                        size,
                    } => {
                        let scope = PassErrorScope::SetVertexBuffer;
                        set_vertex_buffer(&mut state, cmd_buf, slot, buffer, offset, size)
                            .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::SetBlendConstant(ref color) => {
                        set_blend_constant(&mut state, color);
                    }
                    ArcRenderCommand::SetStencilReference(value) => {
                        set_stencil_reference(&mut state, value);
                    }
                    ArcRenderCommand::SetViewport {
                        rect,
                        depth_min,
                        depth_max,
                    } => {
                        let scope = PassErrorScope::SetViewport;
                        set_viewport(&mut state, rect, depth_min, depth_max).map_pass_err(scope)?;
                    }
                    ArcRenderCommand::SetPushConstant {
                        stages,
                        offset,
                        size_bytes,
                        values_offset,
                    } => {
                        let scope = PassErrorScope::SetPushConstant;
                        set_push_constant(
                            &mut state,
                            &base.push_constant_data,
                            stages,
                            offset,
                            size_bytes,
                            values_offset,
                        )
                        .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::SetScissor(rect) => {
                        let scope = PassErrorScope::SetScissorRect;
                        set_scissor(&mut state, rect).map_pass_err(scope)?;
                    }
                    ArcRenderCommand::Draw {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    } => {
                        let scope = PassErrorScope::Draw {
                            kind: DrawKind::Draw,
                            indexed: false,
                        };
                        draw(
                            &mut state,
                            vertex_count,
                            instance_count,
                            first_vertex,
                            first_instance,
                        )
                        .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::DrawIndexed {
                        index_count,
                        instance_count,
                        first_index,
                        base_vertex,
                        first_instance,
                    } => {
                        let scope = PassErrorScope::Draw {
                            kind: DrawKind::Draw,
                            indexed: true,
                        };
                        draw_indexed(
                            &mut state,
                            index_count,
                            instance_count,
                            first_index,
                            base_vertex,
                            first_instance,
                        )
                        .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::MultiDrawIndirect {
                        buffer,
                        offset,
                        count,
                        indexed,
                    } => {
                        let scope = PassErrorScope::Draw {
                            kind: if count.is_some() {
                                DrawKind::MultiDrawIndirect
                            } else {
                                DrawKind::DrawIndirect
                            },
                            indexed,
                        };
                        multi_draw_indirect(&mut state, cmd_buf, buffer, offset, count, indexed)
                            .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::MultiDrawIndirectCount {
                        buffer,
                        offset,
                        count_buffer,
                        count_buffer_offset,
                        max_count,
                        indexed,
                    } => {
                        let scope = PassErrorScope::Draw {
                            kind: DrawKind::MultiDrawIndirectCount,
                            indexed,
                        };
                        multi_draw_indirect_count(
                            &mut state,
                            cmd_buf,
                            buffer,
                            offset,
                            count_buffer,
                            count_buffer_offset,
                            max_count,
                            indexed,
                        )
                        .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::PushDebugGroup { color: _, len } => {
                        push_debug_group(&mut state, &base.string_data, len);
                    }
                    ArcRenderCommand::PopDebugGroup => {
                        let scope = PassErrorScope::PopDebugGroup;
                        pop_debug_group(&mut state).map_pass_err(scope)?;
                    }
                    ArcRenderCommand::InsertDebugMarker { color: _, len } => {
                        insert_debug_marker(&mut state, &base.string_data, len);
                    }
                    ArcRenderCommand::WriteTimestamp {
                        query_set,
                        query_index,
                    } => {
                        let scope = PassErrorScope::WriteTimestamp;
                        write_timestamp(
                            &mut state,
                            cmd_buf,
                            &mut cmd_buf_data.pending_query_resets,
                            query_set,
                            query_index,
                        )
                        .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::BeginOcclusionQuery { query_index } => {
                        api_log!("RenderPass::begin_occlusion_query {query_index}");
                        let scope = PassErrorScope::BeginOcclusionQuery;

                        let query_set = pass
                            .occlusion_query_set
                            .clone()
                            .ok_or(RenderPassErrorInner::MissingOcclusionQuerySet)
                            .map_pass_err(scope)?;

                        validate_and_begin_occlusion_query(
                            query_set,
                            state.raw_encoder,
                            &mut state.tracker.query_sets,
                            query_index,
                            Some(&mut cmd_buf_data.pending_query_resets),
                            &mut state.active_occlusion_query,
                        )
                        .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::EndOcclusionQuery => {
                        api_log!("RenderPass::end_occlusion_query");
                        let scope = PassErrorScope::EndOcclusionQuery;

                        end_occlusion_query(state.raw_encoder, &mut state.active_occlusion_query)
                            .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::BeginPipelineStatisticsQuery {
                        query_set,
                        query_index,
                    } => {
                        api_log!(
                            "RenderPass::begin_pipeline_statistics_query {query_index} {}",
                            query_set.error_ident()
                        );
                        let scope = PassErrorScope::BeginPipelineStatisticsQuery;

                        validate_and_begin_pipeline_statistics_query(
                            query_set,
                            state.raw_encoder,
                            &mut state.tracker.query_sets,
                            cmd_buf.as_ref(),
                            query_index,
                            Some(&mut cmd_buf_data.pending_query_resets),
                            &mut state.active_pipeline_statistics_query,
                        )
                        .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::EndPipelineStatisticsQuery => {
                        api_log!("RenderPass::end_pipeline_statistics_query");
                        let scope = PassErrorScope::EndPipelineStatisticsQuery;

                        end_pipeline_statistics_query(
                            state.raw_encoder,
                            &mut state.active_pipeline_statistics_query,
                        )
                        .map_pass_err(scope)?;
                    }
                    ArcRenderCommand::ExecuteBundle(bundle) => {
                        let scope = PassErrorScope::ExecuteBundle;
                        execute_bundle(&mut state, cmd_buf, bundle).map_pass_err(scope)?;
                    }
                }
            }

            let (trackers, pending_discard_init_fixups) = state
                .info
                .finish(state.raw_encoder, state.snatch_guard)
                .map_pass_err(pass_scope)?;

            encoder.close(&cmd_buf.device).map_pass_err(pass_scope)?;
            (trackers, pending_discard_init_fixups)
        };

        let encoder = &mut cmd_buf_data.encoder;
        let status = &mut cmd_buf_data.status;
        let tracker = &mut cmd_buf_data.trackers;

        {
            let transit = encoder.open(&cmd_buf.device).map_pass_err(pass_scope)?;

            fixup_discarded_surfaces(
                pending_discard_init_fixups.into_iter(),
                transit,
                &mut tracker.textures,
                &cmd_buf.device,
                snatch_guard,
            );

            cmd_buf_data.pending_query_resets.reset_queries(transit);

            CommandBuffer::insert_barriers_from_scope(transit, tracker, &scope, snatch_guard);
        }

        *status = CommandEncoderStatus::Recording;
        encoder
            .close_and_swap(&cmd_buf.device)
            .map_pass_err(pass_scope)?;

        Ok(())
    }
}

fn set_bind_group(
    state: &mut State,
    cmd_buf: &Arc<CommandBuffer>,
    dynamic_offsets: &[DynamicOffset],
    index: u32,
    num_dynamic_offsets: usize,
    bind_group: Option<Arc<BindGroup>>,
) -> Result<(), RenderPassErrorInner> {
    if bind_group.is_none() {
        api_log!("RenderPass::set_bind_group {index} None");
    } else {
        api_log!(
            "RenderPass::set_bind_group {index} {}",
            bind_group.as_ref().unwrap().error_ident()
        );
    }

    let max_bind_groups = state.device.limits.max_bind_groups;
    if index >= max_bind_groups {
        return Err(RenderCommandError::BindGroupIndexOutOfRange {
            index,
            max: max_bind_groups,
        }
        .into());
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

    bind_group.same_device_as(cmd_buf.as_ref())?;

    bind_group.validate_dynamic_bindings(index, &state.temp_offsets)?;

    // merge the resource tracker in
    unsafe {
        state.info.usage_scope.merge_bind_group(&bind_group.used)?;
    }
    //Note: stateless trackers are not merged: the lifetime reference
    // is held to the bind group itself.

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
            .info
            .pending_discard_init_fixups
            .extend(state.texture_memory_actions.register_init_action(action));
    }

    let pipeline_layout = state.binder.pipeline_layout.clone();
    let entries = state
        .binder
        .assign_group(index as usize, bind_group, &state.temp_offsets);
    if !entries.is_empty() && pipeline_layout.is_some() {
        let pipeline_layout = pipeline_layout.as_ref().unwrap().raw();
        for (i, e) in entries.iter().enumerate() {
            if let Some(group) = e.group.as_ref() {
                let raw_bg = group.try_raw(state.snatch_guard)?;
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
    cmd_buf: &Arc<CommandBuffer>,
    pipeline: Arc<RenderPipeline>,
) -> Result<(), RenderPassErrorInner> {
    api_log!("RenderPass::set_pipeline {}", pipeline.error_ident());

    state.pipeline = Some(pipeline.clone());

    let pipeline = state.tracker.render_pipelines.insert_single(pipeline);

    pipeline.same_device_as(cmd_buf.as_ref())?;

    state
        .info
        .context
        .check_compatible(&pipeline.pass_context, pipeline.as_ref())
        .map_err(RenderCommandError::IncompatiblePipelineTargets)?;

    state.pipeline_flags = pipeline.flags;

    if pipeline.flags.contains(PipelineFlags::WRITES_DEPTH) && state.info.is_depth_read_only {
        return Err(RenderCommandError::IncompatibleDepthAccess(pipeline.error_ident()).into());
    }
    if pipeline.flags.contains(PipelineFlags::WRITES_STENCIL) && state.info.is_stencil_read_only {
        return Err(RenderCommandError::IncompatibleStencilAccess(pipeline.error_ident()).into());
    }

    state
        .blend_constant
        .require(pipeline.flags.contains(PipelineFlags::BLEND_CONSTANT));

    unsafe {
        state.raw_encoder.set_render_pipeline(pipeline.raw());
    }

    if pipeline.flags.contains(PipelineFlags::STENCIL_REFERENCE) {
        unsafe {
            state
                .raw_encoder
                .set_stencil_reference(state.stencil_reference);
        }
    }

    // Rebind resource
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
                    let raw_bg = group.try_raw(state.snatch_guard)?;
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

        // Clear push constant ranges
        let non_overlapping =
            super::bind::compute_nonoverlapping_ranges(&pipeline.layout.push_constant_ranges);
        for range in non_overlapping {
            let offset = range.range.start;
            let size_bytes = range.range.end - offset;
            super::push_constant_clear(offset, size_bytes, |clear_offset, clear_data| unsafe {
                state.raw_encoder.set_push_constants(
                    pipeline.layout.raw(),
                    range.stages,
                    clear_offset,
                    clear_data,
                );
            });
        }
    }

    // Initialize each `vertex.inputs[i].step` from
    // `pipeline.vertex_steps[i]`.  Enlarge `vertex.inputs`
    // as necessary to accommodate all slots in the
    // pipeline. If `vertex.inputs` is longer, fill the
    // extra entries with default `VertexStep`s.
    while state.vertex.inputs.len() < pipeline.vertex_steps.len() {
        state.vertex.inputs.push(VertexBufferState::EMPTY);
    }

    // This is worse as a `zip`, but it's close.
    let mut steps = pipeline.vertex_steps.iter();
    for input in state.vertex.inputs.iter_mut() {
        input.step = steps.next().cloned().unwrap_or_default();
    }

    // Update vertex buffer limits.
    state.vertex.update_limits();
    Ok(())
}

fn set_index_buffer(
    state: &mut State,
    cmd_buf: &Arc<CommandBuffer>,
    buffer: Arc<crate::resource::Buffer>,
    index_format: IndexFormat,
    offset: u64,
    size: Option<BufferSize>,
) -> Result<(), RenderPassErrorInner> {
    api_log!("RenderPass::set_index_buffer {}", buffer.error_ident());

    state
        .info
        .usage_scope
        .buffers
        .merge_single(&buffer, hal::BufferUses::INDEX)?;

    buffer.same_device_as(cmd_buf.as_ref())?;

    buffer.check_usage(BufferUsages::INDEX)?;
    let buf_raw = buffer.try_raw(state.snatch_guard)?;

    let end = match size {
        Some(s) => offset + s.get(),
        None => buffer.size,
    };
    state.index.update_buffer(offset..end, index_format);

    state
        .buffer_memory_init_actions
        .extend(buffer.initialization_status.read().create_action(
            &buffer,
            offset..end,
            MemoryInitKind::NeedsInitializedMemory,
        ));

    let bb = hal::BufferBinding {
        buffer: buf_raw,
        offset,
        size,
    };
    unsafe {
        hal::DynCommandEncoder::set_index_buffer(state.raw_encoder, bb, index_format);
    }
    Ok(())
}

fn set_vertex_buffer(
    state: &mut State,
    cmd_buf: &Arc<CommandBuffer>,
    slot: u32,
    buffer: Arc<crate::resource::Buffer>,
    offset: u64,
    size: Option<BufferSize>,
) -> Result<(), RenderPassErrorInner> {
    api_log!(
        "RenderPass::set_vertex_buffer {slot} {}",
        buffer.error_ident()
    );

    state
        .info
        .usage_scope
        .buffers
        .merge_single(&buffer, hal::BufferUses::VERTEX)?;

    buffer.same_device_as(cmd_buf.as_ref())?;

    let max_vertex_buffers = state.device.limits.max_vertex_buffers;
    if slot >= max_vertex_buffers {
        return Err(RenderCommandError::VertexBufferIndexOutOfRange {
            index: slot,
            max: max_vertex_buffers,
        }
        .into());
    }

    buffer.check_usage(BufferUsages::VERTEX)?;
    let buf_raw = buffer.try_raw(state.snatch_guard)?;

    let empty_slots = (1 + slot as usize).saturating_sub(state.vertex.inputs.len());
    state
        .vertex
        .inputs
        .extend(iter::repeat(VertexBufferState::EMPTY).take(empty_slots));
    let vertex_state = &mut state.vertex.inputs[slot as usize];
    //TODO: where are we checking that the offset is in bound?
    vertex_state.total_size = match size {
        Some(s) => s.get(),
        None => buffer.size - offset,
    };
    vertex_state.bound = true;

    state
        .buffer_memory_init_actions
        .extend(buffer.initialization_status.read().create_action(
            &buffer,
            offset..(offset + vertex_state.total_size),
            MemoryInitKind::NeedsInitializedMemory,
        ));

    let bb = hal::BufferBinding {
        buffer: buf_raw,
        offset,
        size,
    };
    unsafe {
        hal::DynCommandEncoder::set_vertex_buffer(state.raw_encoder, slot, bb);
    }
    state.vertex.update_limits();
    Ok(())
}

fn set_blend_constant(state: &mut State, color: &Color) {
    api_log!("RenderPass::set_blend_constant");

    state.blend_constant = OptionalState::Set;
    let array = [
        color.r as f32,
        color.g as f32,
        color.b as f32,
        color.a as f32,
    ];
    unsafe {
        state.raw_encoder.set_blend_constants(&array);
    }
}

fn set_stencil_reference(state: &mut State, value: u32) {
    api_log!("RenderPass::set_stencil_reference {value}");

    state.stencil_reference = value;
    if state
        .pipeline_flags
        .contains(PipelineFlags::STENCIL_REFERENCE)
    {
        unsafe {
            state.raw_encoder.set_stencil_reference(value);
        }
    }
}

fn set_viewport(
    state: &mut State,
    rect: Rect<f32>,
    depth_min: f32,
    depth_max: f32,
) -> Result<(), RenderPassErrorInner> {
    api_log!("RenderPass::set_viewport {rect:?}");
    if rect.x < 0.0
        || rect.y < 0.0
        || rect.w <= 0.0
        || rect.h <= 0.0
        || rect.x + rect.w > state.info.extent.width as f32
        || rect.y + rect.h > state.info.extent.height as f32
    {
        return Err(RenderCommandError::InvalidViewportRect(rect, state.info.extent).into());
    }
    if !(0.0..=1.0).contains(&depth_min) || !(0.0..=1.0).contains(&depth_max) {
        return Err(RenderCommandError::InvalidViewportDepth(depth_min, depth_max).into());
    }
    let r = hal::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    };
    unsafe {
        state.raw_encoder.set_viewport(&r, depth_min..depth_max);
    }
    Ok(())
}

fn set_push_constant(
    state: &mut State,
    push_constant_data: &[u32],
    stages: ShaderStages,
    offset: u32,
    size_bytes: u32,
    values_offset: Option<u32>,
) -> Result<(), RenderPassErrorInner> {
    api_log!("RenderPass::set_push_constants");

    let values_offset = values_offset.ok_or(RenderPassErrorInner::InvalidValuesOffset)?;

    let end_offset_bytes = offset + size_bytes;
    let values_end_offset = (values_offset + size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
    let data_slice = &push_constant_data[(values_offset as usize)..values_end_offset];

    let pipeline_layout = state
        .binder
        .pipeline_layout
        .as_ref()
        .ok_or(DrawError::MissingPipeline)?;

    pipeline_layout
        .validate_push_constant_ranges(stages, offset, end_offset_bytes)
        .map_err(RenderCommandError::from)?;

    unsafe {
        state
            .raw_encoder
            .set_push_constants(pipeline_layout.raw(), stages, offset, data_slice)
    }
    Ok(())
}

fn set_scissor(state: &mut State, rect: Rect<u32>) -> Result<(), RenderPassErrorInner> {
    api_log!("RenderPass::set_scissor_rect {rect:?}");

    if rect.x + rect.w > state.info.extent.width || rect.y + rect.h > state.info.extent.height {
        return Err(RenderCommandError::InvalidScissorRect(rect, state.info.extent).into());
    }
    let r = hal::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    };
    unsafe {
        state.raw_encoder.set_scissor_rect(&r);
    }
    Ok(())
}

fn draw(
    state: &mut State,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) -> Result<(), DrawError> {
    api_log!("RenderPass::draw {vertex_count} {instance_count} {first_vertex} {first_instance}");

    state.is_ready(false)?;

    let last_vertex = first_vertex as u64 + vertex_count as u64;
    let vertex_limit = state.vertex.vertex_limit;
    if last_vertex > vertex_limit {
        return Err(DrawError::VertexBeyondLimit {
            last_vertex,
            vertex_limit,
            slot: state.vertex.vertex_limit_slot,
        });
    }
    let last_instance = first_instance as u64 + instance_count as u64;
    let instance_limit = state.vertex.instance_limit;
    if last_instance > instance_limit {
        return Err(DrawError::InstanceBeyondLimit {
            last_instance,
            instance_limit,
            slot: state.vertex.instance_limit_slot,
        });
    }

    unsafe {
        if instance_count > 0 && vertex_count > 0 {
            state
                .raw_encoder
                .draw(first_vertex, vertex_count, first_instance, instance_count);
        }
    }
    Ok(())
}

fn draw_indexed(
    state: &mut State,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) -> Result<(), DrawError> {
    api_log!("RenderPass::draw_indexed {index_count} {instance_count} {first_index} {base_vertex} {first_instance}");

    state.is_ready(true)?;

    let last_index = first_index as u64 + index_count as u64;
    let index_limit = state.index.limit;
    if last_index > index_limit {
        return Err(DrawError::IndexBeyondLimit {
            last_index,
            index_limit,
        });
    }
    let last_instance = first_instance as u64 + instance_count as u64;
    let instance_limit = state.vertex.instance_limit;
    if last_instance > instance_limit {
        return Err(DrawError::InstanceBeyondLimit {
            last_instance,
            instance_limit,
            slot: state.vertex.instance_limit_slot,
        });
    }

    unsafe {
        if instance_count > 0 && index_count > 0 {
            state.raw_encoder.draw_indexed(
                first_index,
                index_count,
                base_vertex,
                first_instance,
                instance_count,
            );
        }
    }
    Ok(())
}

fn multi_draw_indirect(
    state: &mut State,
    cmd_buf: &Arc<CommandBuffer>,
    indirect_buffer: Arc<crate::resource::Buffer>,
    offset: u64,
    count: Option<NonZeroU32>,
    indexed: bool,
) -> Result<(), RenderPassErrorInner> {
    api_log!(
        "RenderPass::draw_indirect (indexed:{indexed}) {} {offset} {count:?}",
        indirect_buffer.error_ident()
    );

    state.is_ready(indexed)?;

    let stride = match indexed {
        false => size_of::<wgt::DrawIndirectArgs>(),
        true => size_of::<wgt::DrawIndexedIndirectArgs>(),
    };

    if count.is_some() {
        state
            .device
            .require_features(wgt::Features::MULTI_DRAW_INDIRECT)?;
    }
    state
        .device
        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)?;

    indirect_buffer.same_device_as(cmd_buf.as_ref())?;

    state
        .info
        .usage_scope
        .buffers
        .merge_single(&indirect_buffer, hal::BufferUses::INDIRECT)?;

    indirect_buffer.check_usage(BufferUsages::INDIRECT)?;
    let indirect_raw = indirect_buffer.try_raw(state.snatch_guard)?;

    let actual_count = count.map_or(1, |c| c.get());

    if offset % 4 != 0 {
        return Err(RenderPassErrorInner::UnalignedIndirectBufferOffset(offset));
    }

    let end_offset = offset + stride as u64 * actual_count as u64;
    if end_offset > indirect_buffer.size {
        return Err(RenderPassErrorInner::IndirectBufferOverrun {
            count,
            offset,
            end_offset,
            buffer_size: indirect_buffer.size,
        });
    }

    state.buffer_memory_init_actions.extend(
        indirect_buffer.initialization_status.read().create_action(
            &indirect_buffer,
            offset..end_offset,
            MemoryInitKind::NeedsInitializedMemory,
        ),
    );

    match indexed {
        false => unsafe {
            state
                .raw_encoder
                .draw_indirect(indirect_raw, offset, actual_count);
        },
        true => unsafe {
            state
                .raw_encoder
                .draw_indexed_indirect(indirect_raw, offset, actual_count);
        },
    }
    Ok(())
}

fn multi_draw_indirect_count(
    state: &mut State,
    cmd_buf: &Arc<CommandBuffer>,
    indirect_buffer: Arc<crate::resource::Buffer>,
    offset: u64,
    count_buffer: Arc<crate::resource::Buffer>,
    count_buffer_offset: u64,
    max_count: u32,
    indexed: bool,
) -> Result<(), RenderPassErrorInner> {
    api_log!(
        "RenderPass::multi_draw_indirect_count (indexed:{indexed}) {} {offset} {} {count_buffer_offset:?} {max_count:?}",
        indirect_buffer.error_ident(),
        count_buffer.error_ident()
    );

    state.is_ready(indexed)?;

    let stride = match indexed {
        false => size_of::<wgt::DrawIndirectArgs>(),
        true => size_of::<wgt::DrawIndexedIndirectArgs>(),
    } as u64;

    state
        .device
        .require_features(wgt::Features::MULTI_DRAW_INDIRECT_COUNT)?;
    state
        .device
        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)?;

    indirect_buffer.same_device_as(cmd_buf.as_ref())?;
    count_buffer.same_device_as(cmd_buf.as_ref())?;

    state
        .info
        .usage_scope
        .buffers
        .merge_single(&indirect_buffer, hal::BufferUses::INDIRECT)?;

    indirect_buffer.check_usage(BufferUsages::INDIRECT)?;
    let indirect_raw = indirect_buffer.try_raw(state.snatch_guard)?;

    state
        .info
        .usage_scope
        .buffers
        .merge_single(&count_buffer, hal::BufferUses::INDIRECT)?;

    count_buffer.check_usage(BufferUsages::INDIRECT)?;
    let count_raw = count_buffer.try_raw(state.snatch_guard)?;

    if offset % 4 != 0 {
        return Err(RenderPassErrorInner::UnalignedIndirectBufferOffset(offset));
    }

    let end_offset = offset + stride * max_count as u64;
    if end_offset > indirect_buffer.size {
        return Err(RenderPassErrorInner::IndirectBufferOverrun {
            count: None,
            offset,
            end_offset,
            buffer_size: indirect_buffer.size,
        });
    }
    state.buffer_memory_init_actions.extend(
        indirect_buffer.initialization_status.read().create_action(
            &indirect_buffer,
            offset..end_offset,
            MemoryInitKind::NeedsInitializedMemory,
        ),
    );

    let begin_count_offset = count_buffer_offset;
    let end_count_offset = count_buffer_offset + 4;
    if end_count_offset > count_buffer.size {
        return Err(RenderPassErrorInner::IndirectCountBufferOverrun {
            begin_count_offset,
            end_count_offset,
            count_buffer_size: count_buffer.size,
        });
    }
    state.buffer_memory_init_actions.extend(
        count_buffer.initialization_status.read().create_action(
            &count_buffer,
            count_buffer_offset..end_count_offset,
            MemoryInitKind::NeedsInitializedMemory,
        ),
    );

    match indexed {
        false => unsafe {
            state.raw_encoder.draw_indirect_count(
                indirect_raw,
                offset,
                count_raw,
                count_buffer_offset,
                max_count,
            );
        },
        true => unsafe {
            state.raw_encoder.draw_indexed_indirect_count(
                indirect_raw,
                offset,
                count_raw,
                count_buffer_offset,
                max_count,
            );
        },
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

        api_log!("RenderPass::push_debug_group {label:?}");
        unsafe {
            state.raw_encoder.begin_debug_marker(label);
        }
    }
    state.string_offset += len;
}

fn pop_debug_group(state: &mut State) -> Result<(), RenderPassErrorInner> {
    api_log!("RenderPass::pop_debug_group");

    if state.debug_scope_depth == 0 {
        return Err(RenderPassErrorInner::InvalidPopDebugGroup);
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
        api_log!("RenderPass::insert_debug_marker {label:?}");
        unsafe {
            state.raw_encoder.insert_debug_marker(label);
        }
    }
    state.string_offset += len;
}

fn write_timestamp(
    state: &mut State,
    cmd_buf: &CommandBuffer,
    pending_query_resets: &mut QueryResetMap,
    query_set: Arc<QuerySet>,
    query_index: u32,
) -> Result<(), RenderPassErrorInner> {
    api_log!(
        "RenderPass::write_timestamps {query_index} {}",
        query_set.error_ident()
    );

    query_set.same_device_as(cmd_buf)?;

    state
        .device
        .require_features(wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES)?;

    let query_set = state.tracker.query_sets.insert_single(query_set);

    query_set.validate_and_write_timestamp(
        state.raw_encoder,
        query_index,
        Some(pending_query_resets),
    )?;
    Ok(())
}

fn execute_bundle(
    state: &mut State,
    cmd_buf: &Arc<CommandBuffer>,
    bundle: Arc<super::RenderBundle>,
) -> Result<(), RenderPassErrorInner> {
    api_log!("RenderPass::execute_bundle {}", bundle.error_ident());

    let bundle = state.tracker.bundles.insert_single(bundle);

    bundle.same_device_as(cmd_buf.as_ref())?;

    state
        .info
        .context
        .check_compatible(&bundle.context, bundle.as_ref())
        .map_err(RenderPassErrorInner::IncompatibleBundleTargets)?;

    if (state.info.is_depth_read_only && !bundle.is_depth_read_only)
        || (state.info.is_stencil_read_only && !bundle.is_stencil_read_only)
    {
        return Err(
            RenderPassErrorInner::IncompatibleBundleReadOnlyDepthStencil {
                pass_depth: state.info.is_depth_read_only,
                pass_stencil: state.info.is_stencil_read_only,
                bundle_depth: bundle.is_depth_read_only,
                bundle_stencil: bundle.is_stencil_read_only,
            },
        );
    }

    state
        .buffer_memory_init_actions
        .extend(
            bundle
                .buffer_memory_init_actions
                .iter()
                .filter_map(|action| {
                    action
                        .buffer
                        .initialization_status
                        .read()
                        .check_action(action)
                }),
        );
    for action in bundle.texture_memory_init_actions.iter() {
        state
            .info
            .pending_discard_init_fixups
            .extend(state.texture_memory_actions.register_init_action(action));
    }

    unsafe { bundle.execute(state.raw_encoder, state.snatch_guard) }.map_err(|e| match e {
        ExecutionError::DestroyedResource(e) => RenderCommandError::DestroyedResource(e),
        ExecutionError::Unimplemented(what) => RenderCommandError::Unimplemented(what),
    })?;

    unsafe {
        state.info.usage_scope.merge_render_bundle(&bundle.used)?;
    };
    state.reset_bundle();
    Ok(())
}

impl Global {
    fn resolve_render_pass_buffer_id(
        &self,
        scope: PassErrorScope,
        buffer_id: id::Id<id::markers::Buffer>,
    ) -> Result<Arc<crate::resource::Buffer>, RenderPassError> {
        let hub = &self.hub;
        let buffer = hub.buffers.get(buffer_id).get().map_pass_err(scope)?;

        Ok(buffer)
    }

    fn resolve_render_pass_query_set(
        &self,
        scope: PassErrorScope,
        query_set_id: id::Id<id::markers::QuerySet>,
    ) -> Result<Arc<QuerySet>, RenderPassError> {
        let hub = &self.hub;
        let query_set = hub.query_sets.get(query_set_id).get().map_pass_err(scope)?;

        Ok(query_set)
    }

    pub fn render_pass_set_bind_group(
        &self,
        pass: &mut RenderPass,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetBindGroup;
        let base = pass
            .base
            .as_mut()
            .ok_or(RenderPassErrorInner::PassEnded)
            .map_pass_err(scope)?;

        if pass.current_bind_groups.set_and_check_redundant(
            bind_group_id,
            index,
            &mut base.dynamic_offsets,
            offsets,
        ) {
            // Do redundant early-out **after** checking whether the pass is ended or not.
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

        base.commands.push(ArcRenderCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offsets.len(),
            bind_group,
        });

        Ok(())
    }

    pub fn render_pass_set_pipeline(
        &self,
        pass: &mut RenderPass,
        pipeline_id: id::RenderPipelineId,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetPipelineRender;

        let redundant = pass.current_pipeline.set_and_check_redundant(pipeline_id);
        let base = pass.base_mut(scope)?;

        if redundant {
            // Do redundant early-out **after** checking whether the pass is ended or not.
            return Ok(());
        }

        let hub = &self.hub;
        let pipeline = hub
            .render_pipelines
            .get(pipeline_id)
            .get()
            .map_pass_err(scope)?;

        base.commands.push(ArcRenderCommand::SetPipeline(pipeline));

        Ok(())
    }

    pub fn render_pass_set_index_buffer(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetIndexBuffer;
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::SetIndexBuffer {
            buffer: self.resolve_render_pass_buffer_id(scope, buffer_id)?,
            index_format,
            offset,
            size,
        });

        Ok(())
    }

    pub fn render_pass_set_vertex_buffer(
        &self,
        pass: &mut RenderPass,
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetVertexBuffer;
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::SetVertexBuffer {
            slot,
            buffer: self.resolve_render_pass_buffer_id(scope, buffer_id)?,
            offset,
            size,
        });

        Ok(())
    }

    pub fn render_pass_set_blend_constant(
        &self,
        pass: &mut RenderPass,
        color: Color,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetBlendConstant;
        let base = pass.base_mut(scope)?;

        base.commands
            .push(ArcRenderCommand::SetBlendConstant(color));

        Ok(())
    }

    pub fn render_pass_set_stencil_reference(
        &self,
        pass: &mut RenderPass,
        value: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetStencilReference;
        let base = pass.base_mut(scope)?;

        base.commands
            .push(ArcRenderCommand::SetStencilReference(value));

        Ok(())
    }

    pub fn render_pass_set_viewport(
        &self,
        pass: &mut RenderPass,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        depth_min: f32,
        depth_max: f32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetViewport;
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::SetViewport {
            rect: Rect { x, y, w, h },
            depth_min,
            depth_max,
        });

        Ok(())
    }

    pub fn render_pass_set_scissor_rect(
        &self,
        pass: &mut RenderPass,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetScissorRect;
        let base = pass.base_mut(scope)?;

        base.commands
            .push(ArcRenderCommand::SetScissor(Rect { x, y, w, h }));

        Ok(())
    }

    pub fn render_pass_set_push_constants(
        &self,
        pass: &mut RenderPass,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::SetPushConstant;
        let base = pass.base_mut(scope)?;

        if offset & (wgt::PUSH_CONSTANT_ALIGNMENT - 1) != 0 {
            return Err(RenderPassErrorInner::PushConstantOffsetAlignment).map_pass_err(scope);
        }
        if data.len() as u32 & (wgt::PUSH_CONSTANT_ALIGNMENT - 1) != 0 {
            return Err(RenderPassErrorInner::PushConstantSizeAlignment).map_pass_err(scope);
        }

        let value_offset = base
            .push_constant_data
            .len()
            .try_into()
            .map_err(|_| RenderPassErrorInner::PushConstantOutOfMemory)
            .map_pass_err(scope)?;

        base.push_constant_data.extend(
            data.chunks_exact(wgt::PUSH_CONSTANT_ALIGNMENT as usize)
                .map(|arr| u32::from_ne_bytes([arr[0], arr[1], arr[2], arr[3]])),
        );

        base.commands.push(ArcRenderCommand::SetPushConstant {
            stages,
            offset,
            size_bytes: data.len() as u32,
            values_offset: Some(value_offset),
        });

        Ok(())
    }

    pub fn render_pass_draw(
        &self,
        pass: &mut RenderPass,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::Draw {
            kind: DrawKind::Draw,
            indexed: false,
        };
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });

        Ok(())
    }

    pub fn render_pass_draw_indexed(
        &self,
        pass: &mut RenderPass,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::Draw {
            kind: DrawKind::Draw,
            indexed: true,
        };
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });

        Ok(())
    }

    pub fn render_pass_draw_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::Draw {
            kind: DrawKind::DrawIndirect,
            indexed: false,
        };
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::MultiDrawIndirect {
            buffer: self.resolve_render_pass_buffer_id(scope, buffer_id)?,
            offset,
            count: None,
            indexed: false,
        });

        Ok(())
    }

    pub fn render_pass_draw_indexed_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::Draw {
            kind: DrawKind::DrawIndirect,
            indexed: true,
        };
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::MultiDrawIndirect {
            buffer: self.resolve_render_pass_buffer_id(scope, buffer_id)?,
            offset,
            count: None,
            indexed: true,
        });

        Ok(())
    }

    pub fn render_pass_multi_draw_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::Draw {
            kind: DrawKind::MultiDrawIndirect,
            indexed: false,
        };
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::MultiDrawIndirect {
            buffer: self.resolve_render_pass_buffer_id(scope, buffer_id)?,
            offset,
            count: NonZeroU32::new(count),
            indexed: false,
        });

        Ok(())
    }

    pub fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::Draw {
            kind: DrawKind::MultiDrawIndirect,
            indexed: true,
        };
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::MultiDrawIndirect {
            buffer: self.resolve_render_pass_buffer_id(scope, buffer_id)?,
            offset,
            count: NonZeroU32::new(count),
            indexed: true,
        });

        Ok(())
    }

    pub fn render_pass_multi_draw_indirect_count(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::Draw {
            kind: DrawKind::MultiDrawIndirectCount,
            indexed: false,
        };
        let base = pass.base_mut(scope)?;

        base.commands
            .push(ArcRenderCommand::MultiDrawIndirectCount {
                buffer: self.resolve_render_pass_buffer_id(scope, buffer_id)?,
                offset,
                count_buffer: self.resolve_render_pass_buffer_id(scope, count_buffer_id)?,
                count_buffer_offset,
                max_count,
                indexed: false,
            });

        Ok(())
    }

    pub fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::Draw {
            kind: DrawKind::MultiDrawIndirectCount,
            indexed: true,
        };
        let base = pass.base_mut(scope)?;

        base.commands
            .push(ArcRenderCommand::MultiDrawIndirectCount {
                buffer: self.resolve_render_pass_buffer_id(scope, buffer_id)?,
                offset,
                count_buffer: self.resolve_render_pass_buffer_id(scope, count_buffer_id)?,
                count_buffer_offset,
                max_count,
                indexed: true,
            });

        Ok(())
    }

    pub fn render_pass_push_debug_group(
        &self,
        pass: &mut RenderPass,
        label: &str,
        color: u32,
    ) -> Result<(), RenderPassError> {
        let base = pass.base_mut(PassErrorScope::PushDebugGroup)?;

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcRenderCommand::PushDebugGroup {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn render_pass_pop_debug_group(
        &self,
        pass: &mut RenderPass,
    ) -> Result<(), RenderPassError> {
        let base = pass.base_mut(PassErrorScope::PopDebugGroup)?;

        base.commands.push(ArcRenderCommand::PopDebugGroup);

        Ok(())
    }

    pub fn render_pass_insert_debug_marker(
        &self,
        pass: &mut RenderPass,
        label: &str,
        color: u32,
    ) -> Result<(), RenderPassError> {
        let base = pass.base_mut(PassErrorScope::InsertDebugMarker)?;

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcRenderCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn render_pass_write_timestamp(
        &self,
        pass: &mut RenderPass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::WriteTimestamp;
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::WriteTimestamp {
            query_set: self.resolve_render_pass_query_set(scope, query_set_id)?,
            query_index,
        });

        Ok(())
    }

    pub fn render_pass_begin_occlusion_query(
        &self,
        pass: &mut RenderPass,
        query_index: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::BeginOcclusionQuery;
        let base = pass.base_mut(scope)?;

        base.commands
            .push(ArcRenderCommand::BeginOcclusionQuery { query_index });

        Ok(())
    }

    pub fn render_pass_end_occlusion_query(
        &self,
        pass: &mut RenderPass,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::EndOcclusionQuery;
        let base = pass.base_mut(scope)?;

        base.commands.push(ArcRenderCommand::EndOcclusionQuery);

        Ok(())
    }

    pub fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut RenderPass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::BeginPipelineStatisticsQuery;
        let base = pass.base_mut(scope)?;

        base.commands
            .push(ArcRenderCommand::BeginPipelineStatisticsQuery {
                query_set: self.resolve_render_pass_query_set(scope, query_set_id)?,
                query_index,
            });

        Ok(())
    }

    pub fn render_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut RenderPass,
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::EndPipelineStatisticsQuery;
        let base = pass.base_mut(scope)?;

        base.commands
            .push(ArcRenderCommand::EndPipelineStatisticsQuery);

        Ok(())
    }

    pub fn render_pass_execute_bundles(
        &self,
        pass: &mut RenderPass,
        render_bundle_ids: &[id::RenderBundleId],
    ) -> Result<(), RenderPassError> {
        let scope = PassErrorScope::ExecuteBundle;
        let base = pass.base_mut(scope)?;

        let hub = &self.hub;
        let bundles = hub.render_bundles.read();

        for &bundle_id in render_bundle_ids {
            let bundle = bundles.get(bundle_id).get().map_pass_err(scope)?;

            base.commands.push(ArcRenderCommand::ExecuteBundle(bundle));
        }
        pass.current_pipeline.reset();
        pass.current_bind_groups.reset();

        Ok(())
    }
}
