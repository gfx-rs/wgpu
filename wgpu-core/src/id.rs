/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{Epoch, Index};
use std::{cmp::Ordering, fmt, marker::PhantomData, num::NonZeroU64};
use wgt::Backend;

const BACKEND_BITS: usize = 3;
const EPOCH_MASK: u32 = (1 << (32 - BACKEND_BITS)) - 1;
type Dummy = crate::backend::Empty;

#[repr(transparent)]
#[cfg_attr(feature = "trace", derive(serde::Serialize), serde(into = "SerialId"))]
#[cfg_attr(
    feature = "replay",
    derive(serde::Deserialize),
    serde(from = "SerialId")
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "trace")),
    derive(serde::Serialize)
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "replay")),
    derive(serde::Deserialize)
)]
pub struct Id<T>(NonZeroU64, PhantomData<T>);

// This type represents Id in a more readable (and editable) way.
#[allow(dead_code)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
enum SerialId {
    // The only variant forces RON to not ignore "Id"
    Id(Index, Epoch, Backend),
}
#[cfg(feature = "trace")]
impl<T> From<Id<T>> for SerialId {
    fn from(id: Id<T>) -> Self {
        let (index, epoch, backend) = id.unzip();
        Self::Id(index, epoch, backend)
    }
}
#[cfg(feature = "replay")]
impl<T> From<SerialId> for Id<T> {
    fn from(id: SerialId) -> Self {
        match id {
            SerialId::Id(index, epoch, backend) => TypedId::zip(index, epoch, backend),
        }
    }
}

impl<T> Id<T> {
    #[cfg(test)]
    pub(crate) fn dummy() -> Valid<Self> {
        Valid(Id(NonZeroU64::new(1).unwrap(), PhantomData))
    }

    pub fn backend(self) -> Backend {
        match self.0.get() >> (64 - BACKEND_BITS) as u8 {
            0 => Backend::Empty,
            1 => Backend::Vulkan,
            2 => Backend::Metal,
            3 => Backend::Dx12,
            4 => Backend::Dx11,
            5 => Backend::Gl,
            _ => unreachable!(),
        }
    }
}

impl<T> Copy for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.unzip().fmt(formatter)
    }
}

impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// An internal ID that has been checked to point to
/// a valid object in the storages.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub(crate) struct Valid<I>(pub I);

pub trait TypedId {
    fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self;
    fn unzip(self) -> (Index, Epoch, Backend);
}

impl<T> TypedId for Id<T> {
    fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self {
        assert_eq!(0, epoch >> (32 - BACKEND_BITS));
        let v = index as u64 | ((epoch as u64) << 32) | ((backend as u64) << (64 - BACKEND_BITS));
        Id(NonZeroU64::new(v).unwrap(), PhantomData)
    }

    fn unzip(self) -> (Index, Epoch, Backend) {
        (
            self.0.get() as u32,
            (self.0.get() >> 32) as u32 & EPOCH_MASK,
            self.backend(),
        )
    }
}

pub type AdapterId = Id<crate::instance::Adapter<Dummy>>;
pub type SurfaceId = Id<crate::instance::Surface>;
// Device
pub type DeviceId = Id<crate::device::Device<Dummy>>;
pub type QueueId = DeviceId;
// Resource
pub type BufferId = Id<crate::resource::Buffer<Dummy>>;
pub type TextureViewId = Id<crate::resource::TextureView<Dummy>>;
pub type TextureId = Id<crate::resource::Texture<Dummy>>;
pub type SamplerId = Id<crate::resource::Sampler<Dummy>>;
// Binding model
pub type BindGroupLayoutId = Id<crate::binding_model::BindGroupLayout<Dummy>>;
pub type PipelineLayoutId = Id<crate::binding_model::PipelineLayout<Dummy>>;
pub type BindGroupId = Id<crate::binding_model::BindGroup<Dummy>>;
// Pipeline
pub type ShaderModuleId = Id<crate::pipeline::ShaderModule<Dummy>>;
pub type RenderPipelineId = Id<crate::pipeline::RenderPipeline<Dummy>>;
pub type ComputePipelineId = Id<crate::pipeline::ComputePipeline<Dummy>>;
// Command
pub type CommandEncoderId = CommandBufferId;
pub type CommandBufferId = Id<crate::command::CommandBuffer<Dummy>>;
pub type RenderPassEncoderId = *mut crate::command::RenderPass;
pub type ComputePassEncoderId = *mut crate::command::ComputePass;
pub type RenderBundleEncoderId = *mut crate::command::RenderBundleEncoder;
pub type RenderBundleId = Id<crate::command::RenderBundle>;
pub type QuerySetId = Id<crate::resource::QuerySet<Dummy>>;
// Swap chain
pub type SwapChainId = Id<crate::swap_chain::SwapChain<Dummy>>;

impl SurfaceId {
    pub(crate) fn to_swap_chain_id(self, backend: Backend) -> SwapChainId {
        let (index, epoch, _) = self.unzip();
        Id::zip(index, epoch, backend)
    }
}
impl SwapChainId {
    pub fn to_surface_id(self) -> SurfaceId {
        let (index, epoch, _) = self.unzip();
        Id::zip(index, epoch, Backend::Empty)
    }
}

#[test]
fn test_id_backend() {
    for &b in &[
        Backend::Empty,
        Backend::Vulkan,
        Backend::Metal,
        Backend::Dx12,
        Backend::Dx11,
        Backend::Gl,
    ] {
        let id: Id<()> = Id::zip(1, 0, b);
        assert_eq!(id.backend(), b);
    }
}
