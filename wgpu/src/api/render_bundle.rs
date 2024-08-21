use std::{sync::Arc, thread};

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
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderBundle: Send, Sync);

super::impl_partialeq_eq_hash!(RenderBundle);

impl Drop for RenderBundle {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.render_bundle_drop(self.data.as_ref());
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
