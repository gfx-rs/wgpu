use std::collections::HashMap;

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
    /// The value may represent any of WGSL's concrete scalar types.
    pub constants: &'a HashMap<String, f64>,
    /// Whether workgroup scoped memory will be initialized with zero values for this stage.
    ///
    /// This is required by the WebGPU spec, but may have overhead which can be avoided
    /// for cross-platform applications
    pub zero_initialize_workgroup_memory: bool,
}

impl Default for PipelineCompilationOptions<'_> {
    fn default() -> Self {
        // HashMap doesn't have a const constructor, due to the use of RandomState
        // This does introduce some synchronisation costs, but these should be minor,
        // and might be cheaper than the alternative of getting new random state
        static DEFAULT_CONSTANTS: std::sync::OnceLock<HashMap<String, f64>> =
            std::sync::OnceLock::new();
        let constants = DEFAULT_CONSTANTS.get_or_init(Default::default);
        Self {
            constants,
            zero_initialize_workgroup_memory: true,
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
