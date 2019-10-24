/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{Backend, Epoch, Index};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{fmt, marker::PhantomData};

const BACKEND_BITS: usize = 3;
const EPOCH_MASK: u32 = (1 << (32 - BACKEND_BITS)) - 1;
type Dummy = crate::backend::Empty;

#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Id<T>(u64, PhantomData<T>);

impl<T> Id<T> {
    pub fn backend(&self) -> Backend {
        match self.0 >> (64 - BACKEND_BITS) as u8 {
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

pub trait TypedId {
    fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self;
    fn unzip(self) -> (Index, Epoch, Backend);
}

impl<T> TypedId for Id<T> {
    fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self {
        assert_eq!(0, epoch >> (32 - BACKEND_BITS));
        let v = index as u64 | ((epoch as u64) << 32) | ((backend as u64) << (64 - BACKEND_BITS));
        Id(v, PhantomData)
    }

    fn unzip(self) -> (Index, Epoch, Backend) {
        (
            self.0 as u32,
            (self.0 >> 32) as u32 & EPOCH_MASK,
            self.backend(),
        )
    }
}

#[cfg(not(feature = "local"))]
pub type Input<T> = T;
#[cfg(feature = "local")]
pub type Input<T> = PhantomData<T>;
#[cfg(feature = "local")]
pub type Output<T> = T;
#[cfg(not(feature = "local"))]
pub type Output<T> = PhantomData<T>;


pub type AdapterId = Id<crate::Adapter<Dummy>>;
pub type DeviceId = Id<crate::Device<Dummy>>;
pub type QueueId = DeviceId;
// Resource
pub type BufferId = Id<crate::Buffer<Dummy>>;
pub type TextureViewId = Id<crate::TextureView<Dummy>>;
pub type TextureId = Id<crate::Texture<Dummy>>;
pub type SamplerId = Id<crate::Sampler<Dummy>>;
// Binding model
pub type BindGroupLayoutId = Id<crate::BindGroupLayout<Dummy>>;
pub type PipelineLayoutId = Id<crate::PipelineLayout<Dummy>>;
pub type BindGroupId = Id<crate::BindGroup<Dummy>>;
// Pipeline
pub type InputStateId = Id<crate::InputState>;
pub type ShaderModuleId = Id<crate::ShaderModule<Dummy>>;
pub type RenderPipelineId = Id<crate::RenderPipeline<Dummy>>;
pub type ComputePipelineId = Id<crate::ComputePipeline<Dummy>>;
// Command
pub type CommandBufferId = Id<crate::CommandBuffer<Dummy>>;
pub type CommandEncoderId = CommandBufferId;
pub type RenderBundleId = Id<crate::RenderBundle<Dummy>>;
pub type RenderPassId = Id<crate::RenderPass<Dummy>>;
pub type ComputePassId = Id<crate::ComputePass<Dummy>>;
// Swap chain
pub type SurfaceId = Id<crate::Surface>;
pub type SwapChainId = Id<crate::SwapChain<Dummy>>;

impl SurfaceId {
    pub(crate) fn to_swap_chain_id(&self, backend: Backend) -> SwapChainId {
        let (index, epoch, _) = self.unzip();
        Id::zip(index, epoch, backend)
    }
}
impl SwapChainId {
    pub(crate) fn to_surface_id(&self) -> SurfaceId {
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
        let id: Id<()> = Id::zip(0, 0, b);
        assert_eq!(id.backend(), b);
    }
}
