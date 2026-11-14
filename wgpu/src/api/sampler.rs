use crate::*;

/// Handle to a sampler.
///
/// A `Sampler` object defines how a pipeline will sample from a [`TextureView`]. Samplers define
/// image filters (including anisotropy) and address (wrapping) modes, among other things. See
/// the documentation for [`SamplerDescriptor`] for more information.
///
/// It can be created with [`Device::create_sampler`].
///
/// Corresponds to [WebGPU `GPUSampler`](https://gpuweb.github.io/gpuweb/#sampler-interface).
#[derive(Debug, Clone)]
pub struct Sampler {
    pub(crate) inner: dispatch::DispatchSampler,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Sampler: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(Sampler => .inner);

impl Sampler {
    #[cfg(custom)]
    /// Returns custom implementation of Sampler (if custom backend and is internally T)
    pub fn as_custom<T: custom::SamplerInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

#[cfg(wgpu_core)]
impl Sampler {
    /// Create a new sampler of wgpu from a wgpu-core sampler.
    ///
    /// # Arguments
    ///
    /// - `core_sampler` - wgpu-core sampler.
    pub fn from_core(core_sampler: alloc::sync::Arc<wgc::resource::Sampler>) -> Self {
        Self {
            inner: crate::backend::wgpu_core::CoreSampler::from_core(core_sampler).into(),
        }
    }

    /// Returns the underlying wgpu-core sampler if this `Sampler` is on the wgpu-core backend, otherwise `None`.
    pub fn as_core(&self) -> Option<alloc::sync::Arc<wgc::resource::Sampler>> {
        self.inner.as_core_opt().map(|s| s.as_core())
    }
}

/// Describes a [`Sampler`].
///
/// For use with [`Device::create_sampler`].
///
/// Corresponds to [WebGPU `GPUSamplerDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerdescriptor).
pub type SamplerDescriptor<'a> = wgt::SamplerDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(SamplerDescriptor<'_>: Send, Sync);
