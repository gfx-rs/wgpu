use std::{sync::Arc, thread};

use crate::context::ObjectId;
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
    pub(crate) id: ObjectId,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Sampler: Send, Sync);

impl Sampler {
    /// Returns a globally-unique identifier for this `Sampler`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id::new(self.id)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.sampler_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Describes a [`Sampler`].
///
/// For use with [`Device::create_sampler`].
///
/// Corresponds to [WebGPU `GPUSamplerDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerdescriptor).
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerDescriptor<'a> {
    /// Debug label of the sampler. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// How to deal with out of bounds accesses in the u (i.e. x) direction
    pub address_mode_u: AddressMode,
    /// How to deal with out of bounds accesses in the v (i.e. y) direction
    pub address_mode_v: AddressMode,
    /// How to deal with out of bounds accesses in the w (i.e. z) direction
    pub address_mode_w: AddressMode,
    /// How to filter the texture when it needs to be magnified (made larger)
    pub mag_filter: FilterMode,
    /// How to filter the texture when it needs to be minified (made smaller)
    pub min_filter: FilterMode,
    /// How to filter between mip map levels
    pub mipmap_filter: FilterMode,
    /// Minimum level of detail (i.e. mip level) to use
    pub lod_min_clamp: f32,
    /// Maximum level of detail (i.e. mip level) to use
    pub lod_max_clamp: f32,
    /// If this is enabled, this is a comparison sampler using the given comparison function.
    pub compare: Option<CompareFunction>,
    /// Must be at least 1. If this is not 1, all filter modes must be linear.
    pub anisotropy_clamp: u16,
    /// Border color to use when address_mode is [`AddressMode::ClampToBorder`]
    pub border_color: Option<SamplerBorderColor>,
}
static_assertions::assert_impl_all!(SamplerDescriptor<'_>: Send, Sync);

impl Default for SamplerDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: Default::default(),
            address_mode_v: Default::default(),
            address_mode_w: Default::default(),
            mag_filter: Default::default(),
            min_filter: Default::default(),
            mipmap_filter: Default::default(),
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }
}
