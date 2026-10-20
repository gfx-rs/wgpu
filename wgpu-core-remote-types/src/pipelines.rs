use crate::{id, Label};
use alloc::borrow::Cow;
use hashbrown::HashMap;

/// Describes a programmable pipeline stage.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
/// cbindgen:ignore
pub struct ProgrammableStageDescriptor<'a> {
    /// The compiled shader module for this stage.
    pub module: id::ShaderModuleId,

    /// The name of the entry point in `module` that this stage should use.
    ///
    /// - If this is `Some(name)`, `module` must contain an entry point with the
    ///   given name.
    ///
    /// - If this is `None`, `module` must have only one entry point for this
    ///   stage; we use that one.
    pub entry_point: Option<Cow<'a, str>>,

    /// Values for pipeline-overridable constants in `module` that this stage
    /// should use.
    ///
    /// If an `@id` attribute was specified on the declaration,
    /// the key must be the pipeline constant ID as a decimal ASCII number; if not,
    /// the key must be the constant's identifier name.
    ///
    /// The value may represent any of WGSL's concrete scalar types.
    pub constants: HashMap<String, f64>,
}

/// Describes a compute pipeline.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
/// cbindgen:ignore
pub struct ComputePipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<id::PipelineLayoutId>,
    /// The compiled compute stage and its entry point.
    pub stage: ProgrammableStageDescriptor<'a>,
}

/// Describes how the vertex buffer is interpreted.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: wgt::BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: wgt::VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: Cow<'a, [wgt::VertexAttribute]>,
}

/// Describes the vertex process in a render pipeline.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
/// cbindgen:ignore
pub struct VertexState<'a> {
    /// The compiled vertex stage and its entry point.
    pub stage: ProgrammableStageDescriptor<'a>,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: Cow<'a, [Option<VertexBufferLayout<'a>>]>,
}

/// Describes fragment processing in a render pipeline.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
/// cbindgen:ignore
pub struct FragmentState<'a> {
    /// The compiled fragment stage and its entry point.
    pub stage: ProgrammableStageDescriptor<'a>,
    /// The effect of draw calls on the color aspect of the output target.
    pub targets: Cow<'a, [Option<wgt::ColorTargetState>]>,
}

/// Describes a render (graphics) pipeline.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RenderPipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<id::PipelineLayoutId>,
    /// The vertex processing state for this pipeline.
    pub vertex: VertexState<'a>,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    #[serde(default)]
    pub primitive: wgt::PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    #[serde(default)]
    pub depth_stencil: Option<wgt::DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    #[serde(default)]
    pub multisample: wgt::MultisampleState,
    /// The fragment processing state for this pipeline.
    pub fragment: Option<FragmentState<'a>>,
}
