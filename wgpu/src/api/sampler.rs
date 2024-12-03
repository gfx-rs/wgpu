use std::{sync::Arc, thread};

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
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Sampler: Send, Sync);

super::impl_partialeq_eq_hash!(Sampler);

impl Drop for Sampler {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.sampler_drop(self.data.as_ref());
        }
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
