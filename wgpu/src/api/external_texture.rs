use crate::*;

/// Handle to an external texture on the GPU.
///
/// It can be created with [`Device::create_external_texture`].
///
/// Corresponds to [WebGPU `GPUExternalTexture`](https://gpuweb.github.io/gpuweb/#gpuexternaltexture).
#[derive(Debug, Clone)]
pub struct ExternalTexture {
    pub(crate) inner: dispatch::DispatchExternalTexture,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ExternalTexture: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(ExternalTexture => .inner);

impl ExternalTexture {
    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        self.inner.destroy();
    }

    /// Returns custom implementation of ExternalTexture (if custom backend and is internally T)
    #[cfg(custom)]
    pub fn as_custom<T: custom::ExternalTextureInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

#[cfg(wgpu_core)]
impl ExternalTexture {
    /// Create a new external texture of wgpu from a wgpu-core external texture.
    ///
    /// # Arguments
    ///
    /// - `core_external_texture` - wgpu-core external texture.
    pub fn from_core(
        core_external_texture: alloc::sync::Arc<wgc::resource::ExternalTexture>,
    ) -> Self {
        Self {
            inner: crate::backend::wgpu_core::CoreExternalTexture::from_core(core_external_texture)
                .into(),
        }
    }

    /// Returns the underlying wgpu-core external texture if this `ExternalTexture` is on the wgpu-core backend, otherwise `None`.
    pub fn as_core(&self) -> Option<alloc::sync::Arc<wgc::resource::ExternalTexture>> {
        self.inner.as_core_opt().map(|et| et.as_core())
    }
}
/// Describes an [`ExternalTexture`].
///
/// For use with [`Device::create_external_texture`].
///
/// Corresponds to [WebGPU `GPUExternalTextureDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuexternaltexturedescriptor).
pub type ExternalTextureDescriptor<'a> = wgt::ExternalTextureDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(ExternalTextureDescriptor<'_>: Send, Sync);
