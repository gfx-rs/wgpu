use std::{sync::Arc, thread};

use crate::context::ObjectId;
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
    pub(crate) id: ObjectId,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(TextureView: Send, Sync);

impl TextureView {
    /// Returns a globally-unique identifier for this `TextureView`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id::new(self.id)
    }

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
        use wgc::id::TextureViewId;

        let texture_view_id = TextureViewId::from(self.id);

        if let Some(ctx) = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
        {
            unsafe {
                ctx.texture_view_as_hal::<A, F, R>(texture_view_id, hal_texture_view_callback)
            }
        } else {
            hal_texture_view_callback(None)
        }
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.texture_view_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Describes a [`TextureView`].
///
/// For use with [`Texture::create_view`].
///
/// Corresponds to [WebGPU `GPUTextureViewDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputextureviewdescriptor).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TextureViewDescriptor<'a> {
    /// Debug label of the texture view. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Format of the texture view. Either must be the same as the texture format or in the list
    /// of `view_formats` in the texture's descriptor.
    pub format: Option<TextureFormat>,
    /// The dimension of the texture view. For 1D textures, this must be `D1`. For 2D textures it must be one of
    /// `D2`, `D2Array`, `Cube`, and `CubeArray`. For 3D textures it must be `D3`
    pub dimension: Option<TextureViewDimension>,
    /// Aspect of the texture. Color textures must be [`TextureAspect::All`].
    pub aspect: TextureAspect,
    /// Base mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    /// If `Some(count)`, `base_mip_level + count` must be less or equal to underlying texture mip count.
    /// If `None`, considered to include the rest of the mipmap levels, but at least 1 in total.
    pub mip_level_count: Option<u32>,
    /// Base array layer.
    pub base_array_layer: u32,
    /// Layer count.
    /// If `Some(count)`, `base_array_layer + count` must be less or equal to the underlying array count.
    /// If `None`, considered to include the rest of the array layers, but at least 1 in total.
    pub array_layer_count: Option<u32>,
}
static_assertions::assert_impl_all!(TextureViewDescriptor<'_>: Send, Sync);
