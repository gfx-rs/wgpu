/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model::{CreateBindGroupLayoutError, CreatePipelineLayoutError},
    device::{DeviceError, RenderPassContext},
    hub::Resource,
    id::{DeviceId, PipelineLayoutId, ShaderModuleId},
    validation::StageError,
    Label, LifeGuard, Stored,
};
use std::borrow::Cow;
use thiserror::Error;
use wgt::{BufferAddress, IndexFormat, InputStepMode};

#[derive(Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum ShaderModuleSource<'a> {
    SpirV(Cow<'a, [u32]>),
    Wgsl(Cow<'a, str>),
    // Unable to serialize with `naga::Module` in here:
    // requires naga serialization feature.
    //Naga(naga::Module),
}

#[derive(Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ShaderModuleDescriptor<'a> {
    pub label: Label<'a>,
    pub source: ShaderModuleSource<'a>,
}

#[derive(Debug)]
pub struct ShaderModule<B: hal::Backend> {
    pub(crate) raw: B::ShaderModule,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) module: Option<naga::Module>,
    #[cfg(debug_assertions)]
    pub(crate) label: String,
}

impl<B: hal::Backend> Resource for ShaderModule<B> {
    const TYPE: &'static str = "ShaderModule";

    fn life_guard(&self) -> &LifeGuard {
        unreachable!()
    }

    fn label(&self) -> &str {
        #[cfg(debug_assertions)]
        return &self.label;
        #[cfg(not(debug_assertions))]
        return "";
    }
}

#[derive(Clone, Debug, Error)]
pub enum CreateShaderModuleError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    Validation(#[from] naga::proc::ValidationError),
}

/// Describes a programmable pipeline stage.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ProgrammableStageDescriptor<'a> {
    /// The compiled shader module for this stage.
    pub module: ShaderModuleId,
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: Cow<'a, str>,
}

/// Number of implicit bind groups derived at pipeline creation.
pub type ImplicitBindGroupCount = u8;

#[derive(Clone, Debug, Error)]
pub enum ImplicitLayoutError {
    #[error("missing IDs for deriving {0} bind groups")]
    MissingIds(ImplicitBindGroupCount),
    #[error("unable to reflect the shader {0:?} interface")]
    ReflectionError(wgt::ShaderStage),
    #[error(transparent)]
    BindGroup(#[from] CreateBindGroupLayoutError),
    #[error(transparent)]
    Pipeline(#[from] CreatePipelineLayoutError),
}

/// Describes a compute pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ComputePipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<PipelineLayoutId>,
    /// The compiled compute stage and its entry point.
    pub compute_stage: ProgrammableStageDescriptor<'a>,
}

#[derive(Clone, Debug, Error)]
pub enum CreateComputePipelineError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("pipeline layout is invalid")]
    InvalidLayout,
    #[error("unable to derive an implicit layout")]
    Implicit(#[from] ImplicitLayoutError),
    #[error(transparent)]
    Stage(StageError),
}

#[derive(Debug)]
pub struct ComputePipeline<B: hal::Backend> {
    pub(crate) raw: B::ComputePipeline,
    pub(crate) layout_id: Stored<PipelineLayoutId>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Resource for ComputePipeline<B> {
    const TYPE: &'static str = "ComputePipeline";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

/// Describes how the vertex buffer is interpreted.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct VertexBufferDescriptor<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub stride: BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: InputStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: Cow<'a, [wgt::VertexAttributeDescriptor]>,
}

/// Describes vertex input state for a render pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct VertexStateDescriptor<'a> {
    /// The format of any index buffers used with this pipeline.
    pub index_format: IndexFormat,
    /// The format of any vertex buffers used with this pipeline.
    pub vertex_buffers: Cow<'a, [VertexBufferDescriptor<'a>]>,
}

/// Describes a render (graphics) pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct RenderPipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<PipelineLayoutId>,
    /// The compiled vertex stage and its entry point.
    pub vertex_stage: ProgrammableStageDescriptor<'a>,
    /// The compiled fragment stage and its entry point, if any.
    pub fragment_stage: Option<ProgrammableStageDescriptor<'a>>,
    /// The rasterization process for this pipeline.
    pub rasterization_state: Option<wgt::RasterizationStateDescriptor>,
    /// The primitive topology used to interpret vertices.
    pub primitive_topology: wgt::PrimitiveTopology,
    /// The effect of draw calls on the color aspect of the output target.
    pub color_states: Cow<'a, [wgt::ColorStateDescriptor]>,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil_state: Option<wgt::DepthStencilStateDescriptor>,
    /// The vertex input state for this pipeline.
    pub vertex_state: VertexStateDescriptor<'a>,
    /// The number of samples calculated per pixel (for MSAA). For non-multisampled textures,
    /// this should be `1`
    pub sample_count: u32,
    /// Bitmask that restricts the samples of a pixel modified by this pipeline. All samples
    /// can be enabled using the value `!0`
    pub sample_mask: u32,
    /// When enabled, produces another sample mask per pixel based on the alpha output value, that
    /// is ANDed with the sample_mask and the primitive coverage to restrict the set of samples
    /// affected by a primitive.
    ///
    /// The implicit mask produced for alpha of zero is guaranteed to be zero, and for alpha of one
    /// is guaranteed to be all 1-s.
    pub alpha_to_coverage_enabled: bool,
}

#[derive(Clone, Debug, Error)]
pub enum CreateRenderPipelineError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("pipelie layout is invalid")]
    InvalidLayout,
    #[error("unable to derive an implicit layout")]
    Implicit(#[from] ImplicitLayoutError),
    #[error("missing output at index {index}")]
    MissingOutput { index: u8 },
    #[error("incompatible output format at index {index}")]
    IncompatibleOutputFormat { index: u8 },
    #[error("invalid sample count {0}")]
    InvalidSampleCount(u32),
    #[error("vertex buffer {index} stride {stride} does not respect `VERTEX_STRIDE_ALIGNMENT`")]
    UnalignedVertexStride { index: u32, stride: BufferAddress },
    #[error("vertex attribute at location {location} has invalid offset {offset}")]
    InvalidVertexAttributeOffset {
        location: wgt::ShaderLocation,
        offset: BufferAddress,
    },
    #[error("missing required device features {0:?}")]
    MissingFeature(wgt::Features),
    #[error("error in stage {flag:?}")]
    Stage {
        flag: wgt::ShaderStage,
        #[source]
        error: StageError,
    },
}

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct PipelineFlags: u32 {
        const BLEND_COLOR = 1;
        const STENCIL_REFERENCE = 2;
        const WRITES_DEPTH_STENCIL = 4;
    }
}

#[derive(Debug)]
pub struct RenderPipeline<B: hal::Backend> {
    pub(crate) raw: B::GraphicsPipeline,
    pub(crate) layout_id: Stored<PipelineLayoutId>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) pass_context: RenderPassContext,
    pub(crate) flags: PipelineFlags,
    pub(crate) index_format: IndexFormat,
    pub(crate) vertex_strides: Vec<(BufferAddress, InputStepMode)>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Resource for RenderPipeline<B> {
    const TYPE: &'static str = "RenderPipeline";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}
