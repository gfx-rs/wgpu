use crate::*;

#[derive(Clone, Debug)]
/// Advanced options for use when a pipeline is compiled
///
/// This implements `Default`, and for most users can be set to `Default::default()`
pub struct PipelineCompilationOptions<'a> {
    /// Specifies the values of pipeline-overridable constants in the shader module.
    ///
    /// If an `@id` attribute was specified on the declaration,
    /// the key must be the pipeline constant ID as a decimal ASCII number; if not,
    /// the key must be the constant's identifier name.
    ///
    /// If the given constant is specified more than once, the last value specified is used.
    ///
    /// The value may represent any of WGSL's concrete scalar types.
    pub constants: &'a [(&'a str, f64)],
    /// Whether workgroup scoped memory will be initialized with zero values for this stage.
    ///
    /// This is required by the WebGPU spec, but may have overhead which can be avoided
    /// for cross-platform applications
    pub zero_initialize_workgroup_memory: bool,
    /// Required subgroup size for this stage.
    ///
    /// Defaults to [`SubgroupSize::Varying`]. Setting any other value requires
    /// [`Features::SUBGROUP_SIZE_CONTROL`].
    ///
    /// Intended for passthrough shaders. The WebGPU [`subgroup-size-control`
    /// proposal][proposal] specifies a `@subgroup_size` WGSL attribute on the
    /// entry point as the way to request a specific size for WGSL/Naga shaders;
    /// passthrough shaders cannot carry such an attribute, so the request must
    /// be threaded through the pipeline-creation API instead. For non-passthrough
    /// shaders, prefer the WGSL attribute once it is supported, and leave this
    /// field at its default.
    ///
    /// [proposal]: https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-size-control.md
    pub subgroup_size: SubgroupSize,
}

impl Default for PipelineCompilationOptions<'_> {
    fn default() -> Self {
        Self {
            constants: Default::default(),
            zero_initialize_workgroup_memory: true,
            subgroup_size: SubgroupSize::Varying,
        }
    }
}

/// Describes a pipeline cache, which allows reusing compilation work
/// between program runs.
///
/// For use with [`Device::create_pipeline_cache`].
///
/// This type is unique to the Rust API of `wgpu`.
#[derive(Clone, Debug)]
pub struct PipelineCacheDescriptor<'a> {
    /// Debug label of the pipeline cache. This might show up in some logs from `wgpu`
    pub label: Label<'a>,
    /// The data used to initialise the cache initialise
    ///
    /// # Safety
    ///
    /// This data must have been provided from a previous call to
    /// [`PipelineCache::get_data`], if not `None`
    pub data: Option<&'a [u8]>,
    /// Whether to create a cache without data when the provided data
    /// is invalid.
    ///
    /// Recommended to set to true
    pub fallback: bool,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(PipelineCacheDescriptor<'_>: Send, Sync);
