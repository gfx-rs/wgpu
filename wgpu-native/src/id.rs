use crate::{Epoch, Index};
use std::marker::PhantomData;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Id(Index, Epoch);

#[repr(transparent)]
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GenericId<T>(Id, PhantomData<T>);

impl<T> Copy for GenericId<T> {}

impl<T> Clone for GenericId<T> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<T> std::hash::Hash for GenericId<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> PartialEq for GenericId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

pub trait TypedId {
    fn new(index: Index, epoch: Epoch) -> Self;
    fn index(&self) -> Index;
    fn epoch(&self) -> Epoch;
}

impl<T> GenericId<T> {
    fn raw(self) -> Id {
        self.0
    }
}

impl<T> TypedId for GenericId<T> {
    fn new(index: Index, epoch: Epoch) -> Self {
        Self(Id(index, epoch), PhantomData)
    }

    fn index(&self) -> Index {
        (self.raw()).0
    }

    fn epoch(&self) -> Epoch {
        (self.raw()).1
    }
}

#[cfg(not(feature = "gfx-backend-gl"))]
pub type InstanceId = GenericId<InstanceHandle>;

#[cfg(feature = "gfx-backend-gl")]
pub type InstanceId = SurfaceId;

#[cfg(not(feature = "gfx-backend-gl"))]
pub type InstanceHandle = back::Instance;

pub type AdapterId = GenericId<AdapterHandle>;
pub type AdapterHandle = hal::Adapter<back::Backend>;

pub type DeviceId = GenericId<DeviceHandle>;
pub type DeviceHandle = crate::Device<back::Backend>;

pub type QueueId = DeviceId;

// Resource
pub type BufferId = GenericId<BufferHandle>;
pub type BufferHandle = crate::Buffer<back::Backend>;

pub type TextureViewId = GenericId<TextureViewHandle>;
pub type TextureViewHandle = crate::TextureView<back::Backend>;

pub type TextureId = GenericId<TextureHandle>;
pub type TextureHandle = crate::Texture<back::Backend>;

pub type SamplerId = GenericId<SamplerHandle>;
pub type SamplerHandle = crate::Sampler<back::Backend>;

// Binding model
pub type BindGroupLayoutId = GenericId<BindGroupLayoutHandle>;
pub type BindGroupLayoutHandle = crate::BindGroupLayout<back::Backend>;

pub type PipelineLayoutId = GenericId<PipelineLayoutHandle>;
pub type PipelineLayoutHandle = crate::PipelineLayout<back::Backend>;

pub type BindGroupId = GenericId<BindGroupHandle>;
pub type BindGroupHandle = crate::BindGroup<back::Backend>;

// Pipeline
pub type InputStateId = GenericId<InputStateHandle>;
pub enum InputStateHandle {}

pub type ShaderModuleId = GenericId<ShaderModuleHandle>;
pub type ShaderModuleHandle = crate::ShaderModule<back::Backend>;

pub type RenderPipelineId = GenericId<RenderPipelineHandle>;
pub type RenderPipelineHandle = crate::RenderPipeline<back::Backend>;

pub type ComputePipelineId = GenericId<ComputePipelineHandle>;
pub type ComputePipelineHandle = crate::ComputePipeline<back::Backend>;

// Command
pub type CommandBufferId = GenericId<CommandBufferHandle>;
pub type CommandBufferHandle = crate::CommandBuffer<back::Backend>;

pub type CommandEncoderId = CommandBufferId;

pub type RenderBundleId = GenericId<RenderBundleHandle>;
pub enum RenderBundleHandle {}

pub type RenderPassId = GenericId<RenderPassHandle>;
pub type RenderPassHandle = crate::RenderPass<back::Backend>;

pub type ComputePassId = GenericId<ComputePassHandle>;
pub type ComputePassHandle = crate::ComputePass<back::Backend>;

// Swap chain
pub type SurfaceId = GenericId<SurfaceHandle>;
pub type SurfaceHandle = crate::Surface<back::Backend>;

pub type SwapChainId = SurfaceId;
