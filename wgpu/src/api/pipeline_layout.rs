use std::{sync::Arc, thread};

use crate::*;

/// Handle to a pipeline layout.
///
/// A `PipelineLayout` object describes the available binding groups of a pipeline.
/// It can be created with [`Device::create_pipeline_layout`].
///
/// Corresponds to [WebGPU `GPUPipelineLayout`](https://gpuweb.github.io/gpuweb/#gpupipelinelayout).
#[derive(Debug)]
pub struct PipelineLayout {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(PipelineLayout: Send, Sync);

super::impl_partialeq_eq_hash!(PipelineLayout);

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.pipeline_layout_drop(self.data.as_ref());
        }
    }
}

/// Describes a [`PipelineLayout`].
///
/// For use with [`Device::create_pipeline_layout`].
///
/// Corresponds to [WebGPU `GPUPipelineLayoutDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpupipelinelayoutdescriptor).
#[derive(Clone, Debug, Default)]
pub struct PipelineLayoutDescriptor<'a> {
    /// Debug label of the pipeline layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Bind groups that this pipeline uses. The first entry will provide all the bindings for
    /// "set = 0", second entry will provide all the bindings for "set = 1" etc.
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
    /// Set of push constant ranges this pipeline uses. Each shader stage that uses push constants
    /// must define the range in push constant memory that corresponds to its single `var<push_constant>`
    /// buffer.
    ///
    /// If this array is non-empty, the [`Features::PUSH_CONSTANTS`] must be enabled.
    pub push_constant_ranges: &'a [PushConstantRange],
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(PipelineLayoutDescriptor<'_>: Send, Sync);
