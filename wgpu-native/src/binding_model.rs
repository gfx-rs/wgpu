use crate::{
    track::TrackerSet,
    BindGroupLayoutId,
    BufferAddress,
    BufferId,
    DeviceId,
    LifeGuard,
    RefCount,
    SamplerId,
    Stored,
    TextureViewId,
};

use arrayvec::ArrayVec;
use bitflags::bitflags;
use rendy_descriptor::{DescriptorRanges, DescriptorSet};

use std::borrow::Borrow;

pub const MAX_BIND_GROUPS: usize = 4;

bitflags! {
    #[repr(transparent)]
    pub struct ShaderStage: u32 {
        const NONE = 0;
        const VERTEX = 1;
        const FRAGMENT = 2;
        const COMPUTE = 4;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum BindingType {
    UniformBuffer = 0,
    Sampler = 1,
    SampledTexture = 2,
    StorageBuffer = 3,
    UniformBufferDynamic = 4,
    StorageBufferDynamic = 5,
    StorageTexture = 10,
}

#[repr(C)]
#[derive(Clone, Debug, Hash)]
pub struct BindGroupLayoutBinding {
    pub binding: u32,
    pub visibility: ShaderStage,
    pub ty: BindingType,
}

#[repr(C)]
#[derive(Debug)]
pub struct BindGroupLayoutDescriptor {
    pub bindings: *const BindGroupLayoutBinding,
    pub bindings_length: usize,
}

#[derive(Debug)]
pub struct BindGroupLayout<B: hal::Backend> {
    pub(crate) raw: B::DescriptorSetLayout,
    pub(crate) bindings: Vec<BindGroupLayoutBinding>,
    pub(crate) desc_ranges: DescriptorRanges,
    pub(crate) dynamic_count: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct PipelineLayoutDescriptor {
    pub bind_group_layouts: *const BindGroupLayoutId,
    pub bind_group_layouts_length: usize,
}

#[derive(Debug)]
pub struct PipelineLayout<B: hal::Backend> {
    pub(crate) raw: B::PipelineLayout,
    pub(crate) bind_group_layout_ids: ArrayVec<[BindGroupLayoutId; MAX_BIND_GROUPS]>,
}

#[repr(C)]
#[derive(Debug)]
pub struct BufferBinding {
    pub buffer: BufferId,
    pub offset: BufferAddress,
    pub size: BufferAddress,
}

#[repr(C)]
#[derive(Debug)]
pub enum BindingResource {
    Buffer(BufferBinding),
    Sampler(SamplerId),
    TextureView(TextureViewId),
}

#[repr(C)]
#[derive(Debug)]
pub struct BindGroupBinding {
    pub binding: u32,
    pub resource: BindingResource,
}

#[repr(C)]
#[derive(Debug)]
pub struct BindGroupDescriptor {
    pub layout: BindGroupLayoutId,
    pub bindings: *const BindGroupBinding,
    pub bindings_length: usize,
}

#[derive(Debug)]
pub struct BindGroup<B: hal::Backend> {
    pub(crate) raw: DescriptorSet<B>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) layout_id: BindGroupLayoutId,
    pub(crate) life_guard: LifeGuard,
    pub(crate) used: TrackerSet,
    pub(crate) dynamic_count: usize,
}

impl<B: hal::Backend> Borrow<RefCount> for BindGroup<B> {
    fn borrow(&self) -> &RefCount {
        &self.life_guard.ref_count
    }
}
