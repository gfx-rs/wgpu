use crate::*;

/// Handle to a ray tracing pipeline.
///
/// A `RayTracingPipeline` object represents a ray traing pipeline and its stages.
/// It can be created with [`Device::create_ray_tracing_pipeline`].
#[derive(Debug, Clone)]
pub struct RayTracingPipeline {
    pub(crate) inner: dispatch::DispatchRayTracingPipeline,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RayTracingPipeline: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(RayTracingPipeline => .inner);

impl RayTracingPipeline {
    /// Get an object representing the bind group layout at a given index.
    ///
    /// If this pipeline was created with a [default layout][RayTracingPipelineDescriptor::layout], then
    /// bind groups created with the returned `BindGroupLayout` can only be used with this pipeline.
    ///
    /// This method will raise a validation error if there is no bind group layout at `index`.
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        let layout = self.inner.get_bind_group_layout(index);
        BindGroupLayout { inner: layout }
    }

    #[cfg(custom)]
    /// Returns custom implementation of RayTracingPipeline (if custom backend and is internally T)
    pub fn as_custom<T: custom::RayTracingPipelineInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

/// Describes a stage in a ray tracing pipeline
///
/// For use in [`RayTracingPipelineDescriptor`]
#[derive(Clone, Debug)]
pub struct RayTracingStage<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader to use.
    ///
    /// If [`Some`], there must be a shader entry point with this name in `module` of the stage required.
    /// Otherwise, expect exactly one entry point in `module` of the stage required, which will be
    /// selected.
    // NOTE: keep phrasing in sync. with `ComputePipelineDescriptor::entry_point`
    // NOTE: keep phrasing in sync. with `VertexState::entry_point`
    // NOTE: keep phrasing in sync. with `FragmentState::entry_point`
    pub entry_point: Option<&'a str>,
    /// Advanced options for when this pipeline is compiled
    ///
    /// This implements `Default`, and for most users can be set to `Default::default()`
    pub compilation_options: PipelineCompilationOptions<'a>,
}

/// Describes a group of stages to be called for an intersection in a ray tracing pipeline
///
/// For use in [`RayTracingPipelineDescriptor`]
#[derive(Clone, Debug)]
pub enum RayTracingIntersectionDescriptor<'a> {
    /// This group of shaders may only be used when
    /// a BLAS with triangle geometry is intersected.
    Triangle {
        /// Stage to call if, after the entire intersection process is complete, a triangle within an instance bound to this
        /// descriptor is the closest hit.
        closest_hit: RayTracingStage<'a>,
        /// Optional stage to call when a triangle within an instance bound to this descriptor is hit at any point during the
        /// intersection process.
        any_hit: Option<RayTracingStage<'a>>,
    },
}

/// Describes a ray tracing pipeline.
///
/// For use with [`Device::create_ray_tracing_pipeline`].
#[derive(Clone, Debug)]
pub struct RayTracingPipelineDescriptor<'a> {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    ///
    /// If this is set, then [`Device::create_ray_tracing_pipeline`] will raise a validation error if
    /// the layout doesn't match what the shader module(s) expect.
    ///
    /// Using the same [`PipelineLayout`] for many [`RayTracingPipeline`] or [`RenderPipeline`] or [`ComputePipeline`]
    /// pipelines guarantees that you don't have to rebind any resources when switching between
    /// those pipelines.
    ///
    /// ## Default pipeline layout
    ///
    /// If `layout` is `None`, then the pipeline has a [default layout] created and used instead.
    /// The default layout is deduced from the shader modules.
    ///
    /// You can use [`RayTracingPipeline::get_bind_group_layout`] to create bind groups for use with the
    /// default layout. However, these bind groups cannot be used with any other pipelines. This is
    /// convenient for simple pipelines, but using an explicit layout is recommended in most cases.
    ///
    /// Keep phrasing in sync. with [`RenderPipelineDescriptor`] and [`ComputePipelineDescriptor`].
    ///
    /// [default layout]: https://www.w3.org/TR/webgpu/#default-pipeline-layout
    pub layout: Option<&'a PipelineLayout>,
    /// The ray generation stage. The shader stage invoked by command encoder trace rays.
    pub ray_generation: RayTracingStage<'a>,
    /// The miss stage. Called if a ray does not hit any object.
    pub miss: RayTracingStage<'a>,
    /// The list of intersection descriptors
    pub intersection_descs: &'a [RayTracingIntersectionDescriptor<'a>],
    /// The maximum depth of entry points able to be recursed into, discounting the ray generation stage.
    pub max_recersion_depth: u32,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<&'a PipelineCache>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RayTracingPipelineDescriptor<'_>: Send, Sync);
