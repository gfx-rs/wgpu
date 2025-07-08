use crate::*;

/// Handle to a texture on the GPU.
///
/// It can be created with [`Device::create_texture`].
///
/// Corresponds to [WebGPU `GPUTexture`](https://gpuweb.github.io/gpuweb/#texture-interface).
#[derive(Debug, Clone)]
pub struct Texture {
    pub(crate) inner: dispatch::DispatchTexture,
    pub(crate) descriptor: TextureDescriptor<'static>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Texture: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(Texture => .inner);

impl Texture {
    /// Returns the inner hal Texture using a callback. The hal texture will be `None` if the
    /// backend type argument does not match with this wgpu Texture
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal Texture must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Texture>) -> R, R>(
        &self,
        hal_texture_callback: F,
    ) -> R {
        if let Some(tex) = self.inner.as_core_opt() {
            unsafe {
                tex.context
                    .texture_as_hal::<A, F, R>(tex, hal_texture_callback)
            }
        } else {
            hal_texture_callback(None)
        }
    }

    #[cfg(custom)]
    /// Returns custom implementation of Texture (if custom backend and is internally T)
    pub fn as_custom<T: custom::TextureInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }

    /// Creates a view of this texture, specifying an interpretation of its texels and
    /// possibly a subset of its layers and mip levels.
    ///
    /// Texture views are needed to use a texture as a binding in a [`BindGroup`]
    /// or as an attachment in a [`RenderPass`].
    pub fn create_view(&self, desc: &TextureViewDescriptor<'_>) -> TextureView {
        let view = self.inner.create_view(desc);

        let &TextureViewDescriptor {
            label: _,
            format,
            dimension,
            usage,
            aspect,
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
        } = desc;

        // WebGPU spec requires us to fill in optional fields for later access.
        // We could do this by accessing the underlying implementation, but duplicating this
        // logic here is a lot simpler than piping this through from a backend.
        // See <https://www.w3.org/TR/webgpu/#abstract-opdef-resolving-gputextureviewdescriptor-defaults>
        // See also wgpu-core's `create_texture_view`

        let resolved_format = format.unwrap_or_else(|| {
            self.descriptor
                .format
                .aspect_specific_format(aspect)
                .unwrap_or(self.descriptor.format)
        });

        let resolved_dimension = dimension.unwrap_or_else(|| match self.descriptor.dimension {
            TextureDimension::D1 => TextureViewDimension::D1,
            TextureDimension::D2 => {
                if array_layer_count == Some(1) {
                    TextureViewDimension::D2
                } else {
                    TextureViewDimension::D2Array
                }
            }
            TextureDimension::D3 => TextureViewDimension::D3,
        });

        let resolved_mip_level_count = mip_level_count.unwrap_or_else(|| {
            self.descriptor
                .mip_level_count
                .saturating_sub(base_mip_level)
        });

        let resolved_array_layer_count =
            array_layer_count.unwrap_or_else(|| match resolved_dimension {
                TextureViewDimension::D1 | TextureViewDimension::D2 | TextureViewDimension::D3 => 1,
                TextureViewDimension::Cube => 6,
                TextureViewDimension::D2Array | TextureViewDimension::CubeArray => self
                    .descriptor
                    .array_layer_count()
                    .saturating_sub(base_array_layer),
            });

        let resolved_usage = {
            let usage = usage.unwrap_or(wgt::TextureUsages::empty());
            if usage.is_empty() {
                self.descriptor.usage
            } else {
                usage
                // If usage is still empty we have an error, but that's handled by the backend.
            }
        };

        let filled_descriptor = TextureViewDescriptor {
            label: None,
            format: Some(resolved_format),
            dimension: Some(resolved_dimension),
            usage: Some(resolved_usage),
            aspect,
            base_mip_level,
            mip_level_count: Some(resolved_mip_level_count),
            base_array_layer,
            array_layer_count: Some(resolved_array_layer_count),
        };

        TextureView {
            inner: view,
            filled_descriptor,
        }
    }

    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        self.inner.destroy();
    }

    /// Make an `TexelCopyTextureInfo` representing the whole texture.
    pub fn as_image_copy(&self) -> TexelCopyTextureInfo<'_> {
        TexelCopyTextureInfo {
            texture: self,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        }
    }

    /// Returns the size of this `Texture`.
    ///
    /// This is always equal to the `size` that was specified when creating the texture.
    pub fn size(&self) -> Extent3d {
        self.descriptor.size
    }

    /// Returns the width of this `Texture`.
    ///
    /// This is always equal to the `size.width` that was specified when creating the texture.
    pub fn width(&self) -> u32 {
        self.descriptor.size.width
    }

    /// Returns the height of this `Texture`.
    ///
    /// This is always equal to the `size.height` that was specified when creating the texture.
    pub fn height(&self) -> u32 {
        self.descriptor.size.height
    }

    /// Returns the depth or layer count of this `Texture`.
    ///
    /// This is always equal to the `size.depth_or_array_layers` that was specified when creating the texture.
    pub fn depth_or_array_layers(&self) -> u32 {
        self.descriptor.size.depth_or_array_layers
    }

    /// Returns the mip_level_count of this `Texture`.
    ///
    /// This is always equal to the `mip_level_count` that was specified when creating the texture.
    pub fn mip_level_count(&self) -> u32 {
        self.descriptor.mip_level_count
    }

    /// Returns the sample_count of this `Texture`.
    ///
    /// This is always equal to the `sample_count` that was specified when creating the texture.
    pub fn sample_count(&self) -> u32 {
        self.descriptor.sample_count
    }

    /// Returns the dimension of this `Texture`.
    ///
    /// This is always equal to the `dimension` that was specified when creating the texture.
    pub fn dimension(&self) -> TextureDimension {
        self.descriptor.dimension
    }

    /// Returns the format of this `Texture`.
    ///
    /// This is always equal to the `format` that was specified when creating the texture.
    pub fn format(&self) -> TextureFormat {
        self.descriptor.format
    }

    /// Returns the allowed usages of this `Texture`.
    ///
    /// This is always equal to the `usage` that was specified when creating the texture.
    pub fn usage(&self) -> TextureUsages {
        self.descriptor.usage
    }
}

/// Describes a [`Texture`].
///
/// For use with [`Device::create_texture`].
///
/// Corresponds to [WebGPU `GPUTextureDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputexturedescriptor).
pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>, &'a [TextureFormat]>;
static_assertions::assert_impl_all!(TextureDescriptor<'_>: Send, Sync);
