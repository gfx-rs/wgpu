use crate::*;

/// Handle to a texture view.
///
/// A `TextureView` object refers to a [`Texture`], or a subset of its layers and mip levels, and
/// specifies an interpretation of the texture’s texels, which is needed to use a texture as a
/// binding in a [`BindGroup`] or as an attachment in a [`RenderPass`].
/// It can be created using [`Texture::create_view()`], which accepts a [`TextureViewDescriptor`]
/// specifying the properties of the view.
///
/// Corresponds to [WebGPU `GPUTextureView`](https://gpuweb.github.io/gpuweb/#gputextureview).
#[derive(Debug, Clone)]
pub struct TextureView {
    pub(crate) inner: dispatch::DispatchTextureView,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(TextureView: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(TextureView => .inner);

impl TextureView {
    /// Returns the inner hal `TextureView` using a callback. The hal texture will be `None` if the
    /// backend type argument does not match with this wgpu Texture
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal `TextureView` must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::TextureView>) -> R, R>(
        &self,
        hal_texture_view_callback: F,
    ) -> R {
        if let Some(core_view) = self.inner.as_core_opt() {
            unsafe {
                core_view
                    .context
                    .texture_view_as_hal::<A, F, R>(core_view, hal_texture_view_callback)
            }
        } else {
            hal_texture_view_callback(None)
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
