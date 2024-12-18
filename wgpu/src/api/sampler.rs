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
#[derive(Debug)]
pub struct Sampler {
    pub(crate) inner: dispatch::DispatchSampler,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Sampler: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(Sampler => .inner);

/// Describes a [`Sampler`].
///
/// For use with [`Device::create_sampler`].
///
/// Corresponds to [WebGPU `GPUSamplerDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerdescriptor).
pub type SamplerDescriptor<'a> = wgt::SamplerDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(SamplerDescriptor<'_>: Send, Sync);
