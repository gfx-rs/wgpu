use std::{sync::Arc, thread};

use crate::*;

/// Handle to a texture view.
///
/// A `TextureView` object describes a texture and associated metadata needed by a
/// [`RenderPipeline`] or [`BindGroup`].
///
/// Corresponds to [WebGPU `GPUTextureView`](https://gpuweb.github.io/gpuweb/#gputextureview).
#[derive(Debug)]
pub struct TextureView {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(TextureView: Send, Sync);

super::impl_partialeq_eq_hash!(TextureView);

impl TextureView {
    /// Returns the inner hal TextureView using a callback. The hal texture will be `None` if the
    /// backend type argument does not match with this wgpu Texture
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal TextureView must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::TextureView>) -> R, R>(
        &self,
        hal_texture_view_callback: F,
    ) -> R {
        if let Some(ctx) = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
        {
            unsafe {
                ctx.texture_view_as_hal::<A, F, R>(
                    crate::context::downcast_ref(self.data.as_ref()),
                    hal_texture_view_callback,
                )
            }
        } else {
            hal_texture_view_callback(None)
        }
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.texture_view_drop(self.data.as_ref());
        }
    }
}

/// Describes a [`TextureView`].
///
/// For use with [`Texture::create_view`].
///
/// Corresponds to [WebGPU `GPUTextureViewDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputextureviewdescriptor).
pub type TextureViewDescriptor<'a> = wgt::TextureViewDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(TextureViewDescriptor<'_>: Send, Sync);
