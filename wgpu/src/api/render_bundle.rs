use std::{sync::Arc, thread};

use crate::context::ObjectId;
use crate::*;

/// Pre-prepared reusable bundle of GPU operations.
///
/// It only supports a handful of render commands, but it makes them reusable. Executing a
/// [`RenderBundle`] is often more efficient than issuing the underlying commands manually.
///
/// It can be created by use of a [`RenderBundleEncoder`], and executed onto a [`CommandEncoder`]
/// using [`RenderPass::execute_bundles`].
///
/// Corresponds to [WebGPU `GPURenderBundle`](https://gpuweb.github.io/gpuweb/#render-bundle).
#[derive(Debug)]
pub struct RenderBundle {
    pub(crate) context: Arc<C>,
    pub(crate) id: ObjectId,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderBundle: Send, Sync);

impl RenderBundle {
    /// Returns a globally-unique identifier for this `RenderBundle`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id::new(self.id)
    }
}

impl Drop for RenderBundle {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .render_bundle_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Describes a [`RenderBundle`].
///
/// For use with [`RenderBundleEncoder::finish`].
///
/// Corresponds to [WebGPU `GPURenderBundleDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderbundledescriptor).
pub type RenderBundleDescriptor<'a> = wgt::RenderBundleDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(RenderBundleDescriptor<'_>: Send, Sync);
