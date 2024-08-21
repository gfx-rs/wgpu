use std::{sync::Arc, thread};

use crate::*;

/// Handle to a binding group layout.
///
/// A `BindGroupLayout` is a handle to the GPU-side layout of a binding group. It can be used to
/// create a [`BindGroupDescriptor`] object, which in turn can be used to create a [`BindGroup`]
/// object with [`Device::create_bind_group`]. A series of `BindGroupLayout`s can also be used to
/// create a [`PipelineLayoutDescriptor`], which can be used to create a [`PipelineLayout`].
///
/// It can be created with [`Device::create_bind_group_layout`].
///
/// Corresponds to [WebGPU `GPUBindGroupLayout`](
/// https://gpuweb.github.io/gpuweb/#gpubindgrouplayout).
#[derive(Debug)]
pub struct BindGroupLayout {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(BindGroupLayout: Send, Sync);

super::impl_partialeq_eq_hash!(BindGroupLayout);

impl Drop for BindGroupLayout {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.bind_group_layout_drop(self.data.as_ref());
        }
    }
}

/// Describes a [`BindGroupLayout`].
///
/// For use with [`Device::create_bind_group_layout`].
///
/// Corresponds to [WebGPU `GPUBindGroupLayoutDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgrouplayoutdescriptor).
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,

    /// Array of entries in this BindGroupLayout
    pub entries: &'a [BindGroupLayoutEntry],
}
static_assertions::assert_impl_all!(BindGroupLayoutDescriptor<'_>: Send, Sync);
