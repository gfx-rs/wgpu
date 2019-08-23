use crate::{Backend, Epoch, Index};
use std::{
    fmt,
    marker::PhantomData,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Id<T>(u64, PhantomData<T>);

impl<T> Copy for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(formatter)
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
    fn zip(index: Index, epoch: Epoch, _backend: Backend) -> Self {
        Id(index as u64 | ((epoch as u64) << 32), PhantomData)
    }

    fn unzip(self) -> (Index, Epoch, Backend) {
        (self.0 as u32, (self.0 >> 32) as u32, Backend::Vulkan)
    }
}

#[cfg(not(feature = "gfx-backend-gl"))]
pub type InstanceId = Id<InstanceHandle>;

#[cfg(feature = "gfx-backend-gl")]
pub type InstanceId = SurfaceId;

#[cfg(not(feature = "gfx-backend-gl"))]
pub type InstanceHandle = back::Instance;

pub type AdapterId = Id<AdapterHandle>;
pub type AdapterHandle = hal::Adapter<back::Backend>;

pub type DeviceId = Id<DeviceHandle>;
pub type DeviceHandle = crate::Device<back::Backend>;

pub type QueueId = DeviceId;

// Resource
pub type BufferId = Id<BufferHandle>;
pub type BufferHandle = crate::Buffer<back::Backend>;

pub type TextureViewId = Id<TextureViewHandle>;
pub type TextureViewHandle = crate::TextureView<back::Backend>;

pub type TextureId = Id<TextureHandle>;
pub type TextureHandle = crate::Texture<back::Backend>;

pub type SamplerId = Id<SamplerHandle>;
pub type SamplerHandle = crate::Sampler<back::Backend>;

// Binding model
pub type BindGroupLayoutId = Id<BindGroupLayoutHandle>;
pub type BindGroupLayoutHandle = crate::BindGroupLayout<back::Backend>;

pub type PipelineLayoutId = Id<PipelineLayoutHandle>;
pub type PipelineLayoutHandle = crate::PipelineLayout<back::Backend>;

pub type BindGroupId = Id<BindGroupHandle>;
pub type BindGroupHandle = crate::BindGroup<back::Backend>;

// Pipeline
pub type InputStateId = Id<InputStateHandle>;
pub enum InputStateHandle {}

pub type ShaderModuleId = Id<ShaderModuleHandle>;
pub type ShaderModuleHandle = crate::ShaderModule<back::Backend>;

pub type RenderPipelineId = Id<RenderPipelineHandle>;
pub type RenderPipelineHandle = crate::RenderPipeline<back::Backend>;

pub type ComputePipelineId = Id<ComputePipelineHandle>;
pub type ComputePipelineHandle = crate::ComputePipeline<back::Backend>;

// Command
pub type CommandBufferId = Id<CommandBufferHandle>;
pub type CommandBufferHandle = crate::CommandBuffer<back::Backend>;

pub type CommandEncoderId = CommandBufferId;

pub type RenderBundleId = Id<RenderBundleHandle>;
pub enum RenderBundleHandle {}

pub type RenderPassId = Id<RenderPassHandle>;
pub type RenderPassHandle = crate::RenderPass<back::Backend>;

pub type ComputePassId = Id<ComputePassHandle>;
pub type ComputePassHandle = crate::ComputePass<back::Backend>;

// Swap chain
pub type SurfaceId = Id<SurfaceHandle>;
pub type SurfaceHandle = crate::Surface<back::Backend>;

pub type SwapChainId = SurfaceId;
