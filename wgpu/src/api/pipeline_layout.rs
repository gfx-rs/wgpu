use crate::*;

/// Handle to a pipeline layout.
///
/// A `PipelineLayout` object describes the available binding groups of a pipeline.
/// It can be created with [`Device::create_pipeline_layout`].
///
/// Corresponds to [WebGPU `GPUPipelineLayout`](https://gpuweb.github.io/gpuweb/#gpupipelinelayout).
#[derive(Debug)]
pub struct PipelineLayout {
    pub(crate) inner: dispatch::DispatchPipelineLayout,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(PipelineLayout: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(PipelineLayout => .inner);

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
