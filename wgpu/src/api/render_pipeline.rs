use std::{num::NonZeroU32, sync::Arc, thread};

use crate::context::ObjectId;
use crate::*;

/// Handle to a rendering (graphics) pipeline.
///
/// A `RenderPipeline` object represents a graphics pipeline and its stages, bindings, vertex
/// buffers and targets. It can be created with [`Device::create_render_pipeline`].
///
/// Corresponds to [WebGPU `GPURenderPipeline`](https://gpuweb.github.io/gpuweb/#render-pipeline).
#[derive(Debug)]
pub struct RenderPipeline {
    pub(crate) context: Arc<C>,
    pub(crate) id: ObjectId,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderPipeline: Send, Sync);

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .render_pipeline_drop(&self.id, self.data.as_ref());
        }
    }
}

impl RenderPipeline {
    /// Returns a globally-unique identifier for this `RenderPipeline`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id::new(self.id)
    }

    /// Get an object representing the bind group layout at a given index.
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        let context = Arc::clone(&self.context);
        let (id, data) =
            self.context
                .render_pipeline_get_bind_group_layout(&self.id, self.data.as_ref(), index);
        BindGroupLayout { context, id, data }
    }
}

/// Describes how the vertex buffer is interpreted.
///
/// For use in [`VertexState`].
///
/// Corresponds to [WebGPU `GPUVertexBufferLayout`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuvertexbufferlayout).
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: &'a [VertexAttribute],
}
static_assertions::assert_impl_all!(VertexBufferLayout<'_>: Send, Sync);

/// Describes the vertex processing in a render pipeline.
///
/// For use in [`RenderPipelineDescriptor`].
///
/// Corresponds to [WebGPU `GPUVertexState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuvertexstate).
#[derive(Clone, Debug)]
pub struct VertexState<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function with this name
    /// in the shader.
    pub entry_point: &'a str,
    /// Advanced options for when this pipeline is compiled
    ///
    /// This implements `Default`, and for most users can be set to `Default::default()`
    pub compilation_options: PipelineCompilationOptions<'a>,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: &'a [VertexBufferLayout<'a>],
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(VertexState<'_>: Send, Sync);

/// Describes the fragment processing in a render pipeline.
///
/// For use in [`RenderPipelineDescriptor`].
///
/// Corresponds to [WebGPU `GPUFragmentState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpufragmentstate).
#[derive(Clone, Debug)]
pub struct FragmentState<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function with this name
    /// in the shader.
    pub entry_point: &'a str,
    /// Advanced options for when this pipeline is compiled
    ///
    /// This implements `Default`, and for most users can be set to `Default::default()`
    pub compilation_options: PipelineCompilationOptions<'a>,
    /// The color state of the render targets.
    pub targets: &'a [Option<ColorTargetState>],
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(FragmentState<'_>: Send, Sync);

/// Describes a render (graphics) pipeline.
///
/// For use with [`Device::create_render_pipeline`].
///
/// Corresponds to [WebGPU `GPURenderPipelineDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpipelinedescriptor).
#[derive(Clone, Debug)]
pub struct RenderPipelineDescriptor<'a> {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<&'a PipelineLayout>,
    /// The compiled vertex stage, its entry point, and the input buffers layout.
    pub vertex: VertexState<'a>,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    pub primitive: PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil: Option<DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    pub multisample: MultisampleState,
    /// The compiled fragment stage, its entry point, and the color targets.
    pub fragment: Option<FragmentState<'a>>,
    /// If the pipeline will be used with a multiview render pass, this indicates how many array
    /// layers the attachments will have.
    pub multiview: Option<NonZeroU32>,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<&'a PipelineCache>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderPipelineDescriptor<'_>: Send, Sync);
