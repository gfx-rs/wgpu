/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{Epoch, Index};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use wgt::Backend;
use std::{fmt, marker::PhantomData, mem};

const BACKEND_BITS: usize = 3;
const EPOCH_MASK: u32 = (1 << (32 - BACKEND_BITS)) - 1;
type Dummy = crate::backend::Empty;

#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(crate="serde_crate"))]
pub struct Id<T>(u64, PhantomData<T>);

impl<T> Id<T> {
    pub const ERROR: Self = Self(0, PhantomData);

    pub fn backend(self) -> Backend {
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

unsafe impl<T> peek_poke::Poke for Id<T> {
    fn max_size() -> usize {
         mem::size_of::<u64>()
    }
    unsafe fn poke_into(&self, data: *mut u8) -> *mut u8 {
        self.0.poke_into(data)
    }
}

impl<T> peek_poke::Peek for Id<T> {
    unsafe fn peek_from(&mut self, data: *const u8) -> *const u8 {
        self.0.peek_from(data)
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


pub type AdapterId = Id<crate::instance::Adapter<Dummy>>;
pub type SurfaceId = Id<crate::instance::Surface>;
// Device
pub type DeviceId = Id<crate::device::Device<Dummy>>;
pub type QueueId = DeviceId;
pub type ShaderModuleId = Id<crate::device::ShaderModule<Dummy>>;
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
pub type RenderPipelineId = Id<crate::pipeline::RenderPipeline<Dummy>>;
pub type ComputePipelineId = Id<crate::pipeline::ComputePipeline<Dummy>>;
// Command
pub type CommandBufferId = Id<crate::command::CommandBuffer<Dummy>>;
pub type CommandEncoderId = CommandBufferId;
pub type RenderPassId = *mut crate::command::RawPass;
pub type ComputePassId = *mut crate::command::RawPass;
pub type RenderBundleId = Id<crate::command::RenderBundle<Dummy>>;
// Swap chain
pub type SwapChainId = Id<crate::swap_chain::SwapChain<Dummy>>;

impl SurfaceId {
    pub(crate) fn to_swap_chain_id(self, backend: Backend) -> SwapChainId {
        let (index, epoch, _) = self.unzip();
        Id::zip(index, epoch, backend)
    }
}
impl SwapChainId {
    pub(crate) fn to_surface_id(self) -> SurfaceId {
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
