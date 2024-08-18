use std::{sync::Arc, thread};

use crate::context::ObjectId;
use crate::*;

/// Handle to a compute pipeline.
///
/// A `ComputePipeline` object represents a compute pipeline and its single shader stage.
/// It can be created with [`Device::create_compute_pipeline`].
///
/// Corresponds to [WebGPU `GPUComputePipeline`](https://gpuweb.github.io/gpuweb/#compute-pipeline).
#[derive(Debug)]
pub struct ComputePipeline {
    pub(crate) context: Arc<C>,
    pub(crate) id: ObjectId,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ComputePipeline: Send, Sync);

impl ComputePipeline {
    /// Returns a globally-unique identifier for this `ComputePipeline`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id::new(self.id)
    }

    /// Get an object representing the bind group layout at a given index.
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        let context = Arc::clone(&self.context);
        let (id, data) = self.context.compute_pipeline_get_bind_group_layout(
            &self.id,
            self.data.as_ref(),
            index,
        );
        BindGroupLayout { context, id, data }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .compute_pipeline_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Describes a compute pipeline.
///
/// For use with [`Device::create_compute_pipeline`].
///
/// Corresponds to [WebGPU `GPUComputePipelineDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucomputepipelinedescriptor).
#[derive(Clone, Debug)]
pub struct ComputePipelineDescriptor<'a> {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<&'a PipelineLayout>,
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader to use.
    ///
    /// If [`Some`], there must be a compute shader entry point with this name in `module`.
    /// Otherwise, expect exactly one compute shader entry point in `module`, which will be
    /// selected.
    // NOTE: keep phrasing in sync. with `FragmentState::entry_point`
    // NOTE: keep phrasing in sync. with `VertexState::entry_point`
    pub entry_point: Option<&'a str>,
    /// Advanced options for when this pipeline is compiled
    ///
    /// This implements `Default`, and for most users can be set to `Default::default()`
    pub compilation_options: PipelineCompilationOptions<'a>,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<&'a PipelineCache>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ComputePipelineDescriptor<'_>: Send, Sync);
