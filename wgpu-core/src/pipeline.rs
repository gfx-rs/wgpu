pub use crate::pipeline_cache::PipelineCacheValidationError;
use crate::{
    binding_model::{CreateBindGroupLayoutError, CreatePipelineLayoutError, PipelineLayout},
    command::ColorAttachmentError,
    device::{Device, DeviceError, MissingDownlevelFlags, MissingFeatures, RenderPassContext},
    id::{PipelineCacheId, PipelineLayoutId, ShaderModuleId},
    resource::{InvalidResourceError, Labeled, TrackingData},
    resource_log, validation, Label,
};
use arrayvec::ArrayVec;
use naga::error::ShaderError;
use std::{borrow::Cow, marker::PhantomData, mem::ManuallyDrop, num::NonZeroU32, sync::Arc};
use thiserror::Error;

/// Information about buffer bindings, which
/// is validated against the shader (and pipeline)
/// at draw time as opposed to initialization time.
#[derive(Debug)]
pub(crate) struct LateSizedBufferGroup {
    // The order has to match `BindGroup::late_buffer_binding_sizes`.
    pub(crate) shader_sizes: Vec<wgt::BufferAddress>,
}

#[allow(clippy::large_enum_variant)]
pub enum ShaderModuleSource<'a> {
    #[cfg(feature = "wgsl")]
    Wgsl(Cow<'a, str>),
    #[cfg(feature = "glsl")]
    Glsl(Cow<'a, str>, naga::front::glsl::Options),
    #[cfg(feature = "spirv")]
    SpirV(Cow<'a, [u32]>, naga::front::spv::Options),
    Naga(Cow<'static, naga::Module>),
    /// Dummy variant because `Naga` doesn't have a lifetime and without enough active features it
    /// could be the last one active.
    #[doc(hidden)]
    Dummy(PhantomData<&'a ()>),
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShaderModuleDescriptor<'a> {
    pub label: Label<'a>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub shader_bound_checks: wgt::ShaderBoundChecks,
}

#[derive(Debug)]
pub struct ShaderModule {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynShaderModule>>,
    pub(crate) device: Arc<Device>,
    pub(crate) interface: Option<validation::Interface>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_shader_module(raw);
        }
    }
}

crate::impl_resource_type!(ShaderModule);
crate::impl_labeled!(ShaderModule);
crate::impl_parent_device!(ShaderModule);
crate::impl_storage_item!(ShaderModule);

impl ShaderModule {
    pub(crate) fn raw(&self) -> &dyn hal::DynShaderModule {
        self.raw.as_ref()
    }

    pub(crate) fn finalize_entry_point_name(
        &self,
        stage_bit: wgt::ShaderStages,
        entry_point: Option<&str>,
    ) -> Result<String, validation::StageError> {
        match &self.interface {
            Some(interface) => interface.finalize_entry_point_name(stage_bit, entry_point),
            None => entry_point
                .map(|ep| ep.to_string())
                .ok_or(validation::StageError::NoEntryPointFound),
        }
    }
}

//Note: `Clone` would require `WithSpan: Clone`.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateShaderModuleError {
    #[cfg(any(feature = "wgsl", feature = "indirect-validation"))]
    #[error(transparent)]
    Parsing(#[from] ShaderError<naga::front::wgsl::ParseError>),
    #[cfg(feature = "glsl")]
    #[error(transparent)]
    ParsingGlsl(#[from] ShaderError<naga::front::glsl::ParseErrors>),
    #[cfg(feature = "spirv")]
    #[error(transparent)]
    ParsingSpirV(#[from] ShaderError<naga::front::spv::Error>),
    #[error("Failed to generate the backend-specific code")]
    Generation,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    Validation(#[from] ShaderError<naga::WithSpan<naga::valid::ValidationError>>),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(
        "Shader global {bind:?} uses a group index {group} that exceeds the max_bind_groups limit of {limit}."
    )]
    InvalidGroupIndex {
        bind: naga::ResourceBinding,
        group: u32,
        limit: u32,
    },
}

/// Describes a programmable pipeline stage.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProgrammableStageDescriptor<'a> {
    /// The compiled shader module for this stage.
    pub module: ShaderModuleId,
    /// The name of the entry point in the compiled shader. The name is selected using the
    /// following logic:
    ///
    /// * If `Some(name)` is specified, there must be a function with this name in the shader.
    /// * If a single entry point associated with this stage must be in the shader, then proceed as
    ///   if `Some(…)` was specified with that entry point's name.
    pub entry_point: Option<Cow<'a, str>>,
    /// Specifies the values of pipeline-overridable constants in the shader module.
    ///
    /// If an `@id` attribute was specified on the declaration,
    /// the key must be the pipeline constant ID as a decimal ASCII number; if not,
    /// the key must be the constant's identifier name.
    ///
    /// The value may represent any of WGSL's concrete scalar types.
    pub constants: Cow<'a, naga::back::PipelineConstants>,
    /// Whether workgroup scoped memory will be initialized with zero values for this stage.
    ///
    /// This is required by the WebGPU spec, but may have overhead which can be avoided
    /// for cross-platform applications
    pub zero_initialize_workgroup_memory: bool,
}

/// Describes a programmable pipeline stage.
#[derive(Clone, Debug)]
pub struct ResolvedProgrammableStageDescriptor<'a> {
    /// The compiled shader module for this stage.
    pub module: Arc<ShaderModule>,
    /// The name of the entry point in the compiled shader. The name is selected using the
    /// following logic:
    ///
    /// * If `Some(name)` is specified, there must be a function with this name in the shader.
    /// * If a single entry point associated with this stage must be in the shader, then proceed as
    ///   if `Some(…)` was specified with that entry point's name.
    pub entry_point: Option<Cow<'a, str>>,
    /// Specifies the values of pipeline-overridable constants in the shader module.
    ///
    /// If an `@id` attribute was specified on the declaration,
    /// the key must be the pipeline constant ID as a decimal ASCII number; if not,
    /// the key must be the constant's identifier name.
    ///
    /// The value may represent any of WGSL's concrete scalar types.
    pub constants: Cow<'a, naga::back::PipelineConstants>,
    /// Whether workgroup scoped memory will be initialized with zero values for this stage.
    ///
    /// This is required by the WebGPU spec, but may have overhead which can be avoided
    /// for cross-platform applications
    pub zero_initialize_workgroup_memory: bool,
}

/// Number of implicit bind groups derived at pipeline creation.
pub type ImplicitBindGroupCount = u8;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ImplicitLayoutError {
    #[error("The implicit_pipeline_ids arg is required")]
    MissingImplicitPipelineIds,
    #[error("Missing IDs for deriving {0} bind groups")]
    MissingIds(ImplicitBindGroupCount),
    #[error("Unable to reflect the shader {0:?} interface")]
    ReflectionError(wgt::ShaderStages),
    #[error(transparent)]
    BindGroup(#[from] CreateBindGroupLayoutError),
    #[error(transparent)]
    Pipeline(#[from] CreatePipelineLayoutError),
}

/// Describes a compute pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputePipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<PipelineLayoutId>,
    /// The compiled compute stage and its entry point.
    pub stage: ProgrammableStageDescriptor<'a>,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<PipelineCacheId>,
}

/// Describes a compute pipeline.
#[derive(Clone, Debug)]
pub struct ResolvedComputePipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<Arc<PipelineLayout>>,
    /// The compiled compute stage and its entry point.
    pub stage: ResolvedProgrammableStageDescriptor<'a>,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<Arc<PipelineCache>>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateComputePipelineError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Unable to derive an implicit layout")]
    Implicit(#[from] ImplicitLayoutError),
    #[error("Error matching shader requirements against the pipeline")]
    Stage(#[from] validation::StageError),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Pipeline constant error: {0}")]
    PipelineConstants(String),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

#[derive(Debug)]
pub struct ComputePipeline {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynComputePipeline>>,
    pub(crate) layout: Arc<PipelineLayout>,
    pub(crate) device: Arc<Device>,
    pub(crate) _shader_module: Arc<ShaderModule>,
    pub(crate) late_sized_buffer_groups: ArrayVec<LateSizedBufferGroup, { hal::MAX_BIND_GROUPS }>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_compute_pipeline(raw);
        }
    }
}

crate::impl_resource_type!(ComputePipeline);
crate::impl_labeled!(ComputePipeline);
crate::impl_parent_device!(ComputePipeline);
crate::impl_storage_item!(ComputePipeline);
crate::impl_trackable!(ComputePipeline);

impl ComputePipeline {
    pub(crate) fn raw(&self) -> &dyn hal::DynComputePipeline {
        self.raw.as_ref()
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreatePipelineCacheError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Pipeline cache validation failed")]
    Validation(#[from] PipelineCacheValidationError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

#[derive(Debug)]
pub struct PipelineCache {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynPipelineCache>>,
    pub(crate) device: Arc<Device>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_pipeline_cache(raw);
        }
    }
}

crate::impl_resource_type!(PipelineCache);
crate::impl_labeled!(PipelineCache);
crate::impl_parent_device!(PipelineCache);
crate::impl_storage_item!(PipelineCache);

impl PipelineCache {
    pub(crate) fn raw(&self) -> &dyn hal::DynPipelineCache {
        self.raw.as_ref()
    }
}

/// Describes how the vertex buffer is interpreted.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: wgt::BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: wgt::VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: Cow<'a, [wgt::VertexAttribute]>,
}

/// Describes the vertex process in a render pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexState<'a> {
    /// The compiled vertex stage and its entry point.
    pub stage: ProgrammableStageDescriptor<'a>,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: Cow<'a, [VertexBufferLayout<'a>]>,
}

/// Describes the vertex process in a render pipeline.
#[derive(Clone, Debug)]
pub struct ResolvedVertexState<'a> {
    /// The compiled vertex stage and its entry point.
    pub stage: ResolvedProgrammableStageDescriptor<'a>,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: Cow<'a, [VertexBufferLayout<'a>]>,
}

/// Describes fragment processing in a render pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentState<'a> {
    /// The compiled fragment stage and its entry point.
    pub stage: ProgrammableStageDescriptor<'a>,
    /// The effect of draw calls on the color aspect of the output target.
    pub targets: Cow<'a, [Option<wgt::ColorTargetState>]>,
}

/// Describes fragment processing in a render pipeline.
#[derive(Clone, Debug)]
pub struct ResolvedFragmentState<'a> {
    /// The compiled fragment stage and its entry point.
    pub stage: ResolvedProgrammableStageDescriptor<'a>,
    /// The effect of draw calls on the color aspect of the output target.
    pub targets: Cow<'a, [Option<wgt::ColorTargetState>]>,
}

/// Describes a render (graphics) pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderPipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<PipelineLayoutId>,
    /// The vertex processing state for this pipeline.
    pub vertex: VertexState<'a>,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive: wgt::PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_stencil: Option<wgt::DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    #[cfg_attr(feature = "serde", serde(default))]
    pub multisample: wgt::MultisampleState,
    /// The fragment processing state for this pipeline.
    pub fragment: Option<FragmentState<'a>>,
    /// If the pipeline will be used with a multiview render pass, this indicates how many array
    /// layers the attachments will have.
    pub multiview: Option<NonZeroU32>,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<PipelineCacheId>,
}

/// Describes a render (graphics) pipeline.
#[derive(Clone, Debug)]
pub struct ResolvedRenderPipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<Arc<PipelineLayout>>,
    /// The vertex processing state for this pipeline.
    pub vertex: ResolvedVertexState<'a>,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    pub primitive: wgt::PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil: Option<wgt::DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    pub multisample: wgt::MultisampleState,
    /// The fragment processing state for this pipeline.
    pub fragment: Option<ResolvedFragmentState<'a>>,
    /// If the pipeline will be used with a multiview render pass, this indicates how many array
    /// layers the attachments will have.
    pub multiview: Option<NonZeroU32>,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<Arc<PipelineCache>>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineCacheDescriptor<'a> {
    pub label: Label<'a>,
    pub data: Option<Cow<'a, [u8]>>,
    pub fallback: bool,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ColorStateError {
    #[error("Format {0:?} is not renderable")]
    FormatNotRenderable(wgt::TextureFormat),
    #[error("Format {0:?} is not blendable")]
    FormatNotBlendable(wgt::TextureFormat),
    #[error("Format {0:?} does not have a color aspect")]
    FormatNotColor(wgt::TextureFormat),
    #[error("Sample count {0} is not supported by format {1:?} on this device. The WebGPU spec guarantees {2:?} samples are supported by this format. With the TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES feature your device supports {3:?}.")]
    InvalidSampleCount(u32, wgt::TextureFormat, Vec<u32>, Vec<u32>),
    #[error("Output format {pipeline} is incompatible with the shader {shader}")]
    IncompatibleFormat {
        pipeline: validation::NumericType,
        shader: validation::NumericType,
    },
    #[error("Invalid write mask {0:?}")]
    InvalidWriteMask(wgt::ColorWrites),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DepthStencilStateError {
    #[error("Format {0:?} is not renderable")]
    FormatNotRenderable(wgt::TextureFormat),
    #[error("Format {0:?} does not have a depth aspect, but depth test/write is enabled")]
    FormatNotDepth(wgt::TextureFormat),
    #[error("Format {0:?} does not have a stencil aspect, but stencil test/write is enabled")]
    FormatNotStencil(wgt::TextureFormat),
    #[error("Sample count {0} is not supported by format {1:?} on this device. The WebGPU spec guarantees {2:?} samples are supported by this format. With the TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES feature your device supports {3:?}.")]
    InvalidSampleCount(u32, wgt::TextureFormat, Vec<u32>, Vec<u32>),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateRenderPipelineError {
    #[error(transparent)]
    ColorAttachment(#[from] ColorAttachmentError),
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Unable to derive an implicit layout")]
    Implicit(#[from] ImplicitLayoutError),
    #[error("Color state [{0}] is invalid")]
    ColorState(u8, #[source] ColorStateError),
    #[error("Depth/stencil state is invalid")]
    DepthStencilState(#[from] DepthStencilStateError),
    #[error("Invalid sample count {0}")]
    InvalidSampleCount(u32),
    #[error("The number of vertex buffers {given} exceeds the limit {limit}")]
    TooManyVertexBuffers { given: u32, limit: u32 },
    #[error("The total number of vertex attributes {given} exceeds the limit {limit}")]
    TooManyVertexAttributes { given: u32, limit: u32 },
    #[error("Vertex buffer {index} stride {given} exceeds the limit {limit}")]
    VertexStrideTooLarge { index: u32, given: u32, limit: u32 },
    #[error("Vertex buffer {index} stride {stride} does not respect `VERTEX_STRIDE_ALIGNMENT`")]
    UnalignedVertexStride {
        index: u32,
        stride: wgt::BufferAddress,
    },
    #[error("Vertex attribute at location {location} has invalid offset {offset}")]
    InvalidVertexAttributeOffset {
        location: wgt::ShaderLocation,
        offset: wgt::BufferAddress,
    },
    #[error("Two or more vertex attributes were assigned to the same location in the shader: {0}")]
    ShaderLocationClash(u32),
    #[error("Strip index format was not set to None but to {strip_index_format:?} while using the non-strip topology {topology:?}")]
    StripIndexFormatForNonStripTopology {
        strip_index_format: Option<wgt::IndexFormat>,
        topology: wgt::PrimitiveTopology,
    },
    #[error("Conservative Rasterization is only supported for wgt::PolygonMode::Fill")]
    ConservativeRasterizationNonFillPolygonMode,
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("Error matching {stage:?} shader requirements against the pipeline")]
    Stage {
        stage: wgt::ShaderStages,
        #[source]
        error: validation::StageError,
    },
    #[error("Internal error in {stage:?} shader: {error}")]
    Internal {
        stage: wgt::ShaderStages,
        error: String,
    },
    #[error("Pipeline constant error in {stage:?} shader: {error}")]
    PipelineConstants {
        stage: wgt::ShaderStages,
        error: String,
    },
    #[error("In the provided shader, the type given for group {group} binding {binding} has a size of {size}. As the device does not support `DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED`, the type must have a size that is a multiple of 16 bytes.")]
    UnalignedShader { group: u32, binding: u32, size: u64 },
    #[error("Using the blend factor {factor:?} for render target {target} is not possible. Only the first render target may be used when dual-source blending.")]
    BlendFactorOnUnsupportedTarget {
        factor: wgt::BlendFactor,
        target: u32,
    },
    #[error("Pipeline expects the shader entry point to make use of dual-source blending.")]
    PipelineExpectsShaderToUseDualSourceBlending,
    #[error("Shader entry point expects the pipeline to make use of dual-source blending.")]
    ShaderExpectsPipelineToUseDualSourceBlending,
    #[error("{}", concat!(
        "At least one color attachment or depth-stencil attachment was expected, ",
        "but no render target for the pipeline was specified."
    ))]
    NoTargetSpecified,
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct PipelineFlags: u32 {
        const BLEND_CONSTANT = 1 << 0;
        const STENCIL_REFERENCE = 1 << 1;
        const WRITES_DEPTH = 1 << 2;
        const WRITES_STENCIL = 1 << 3;
    }
}

/// How a render pipeline will retrieve attributes from a particular vertex buffer.
#[derive(Clone, Copy, Debug)]
pub struct VertexStep {
    /// The byte stride in the buffer between one attribute value and the next.
    pub stride: wgt::BufferAddress,

    /// The byte size required to fit the last vertex in the stream.
    pub last_stride: wgt::BufferAddress,

    /// Whether the buffer is indexed by vertex number or instance number.
    pub mode: wgt::VertexStepMode,
}

impl Default for VertexStep {
    fn default() -> Self {
        Self {
            stride: 0,
            last_stride: 0,
            mode: wgt::VertexStepMode::Vertex,
        }
    }
}

#[derive(Debug)]
pub struct RenderPipeline {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynRenderPipeline>>,
    pub(crate) device: Arc<Device>,
    pub(crate) layout: Arc<PipelineLayout>,
    pub(crate) _shader_modules: ArrayVec<Arc<ShaderModule>, { hal::MAX_CONCURRENT_SHADER_STAGES }>,
    pub(crate) pass_context: RenderPassContext,
    pub(crate) flags: PipelineFlags,
    pub(crate) strip_index_format: Option<wgt::IndexFormat>,
    pub(crate) vertex_steps: Vec<VertexStep>,
    pub(crate) late_sized_buffer_groups: ArrayVec<LateSizedBufferGroup, { hal::MAX_BIND_GROUPS }>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_render_pipeline(raw);
        }
    }
}

crate::impl_resource_type!(RenderPipeline);
crate::impl_labeled!(RenderPipeline);
crate::impl_parent_device!(RenderPipeline);
crate::impl_storage_item!(RenderPipeline);
crate::impl_trackable!(RenderPipeline);

impl RenderPipeline {
    pub(crate) fn raw(&self) -> &dyn hal::DynRenderPipeline {
        self.raw.as_ref()
    }
}
