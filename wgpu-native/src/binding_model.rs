use crate::track::TrackerSet;
use crate::{BindGroupLayoutId, BufferId, LifeGuard, SamplerId, TextureViewId};

use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    pub struct ShaderStageFlags: u32 {
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
}

#[repr(C)]
pub struct BindGroupLayoutBinding {
    pub binding: u32,
    pub visibility: ShaderStageFlags,
    pub ty: BindingType,
}

#[repr(C)]
pub struct BindGroupLayoutDescriptor {
    pub bindings: *const BindGroupLayoutBinding,
    pub bindings_length: usize,
}

pub struct BindGroupLayout<B: hal::Backend> {
    pub(crate) raw: B::DescriptorSetLayout,
}

#[repr(C)]
pub struct PipelineLayoutDescriptor {
    pub bind_group_layouts: *const BindGroupLayoutId,
    pub bind_group_layouts_length: usize,
}

pub struct PipelineLayout<B: hal::Backend> {
    pub(crate) raw: B::PipelineLayout,
    pub(crate) bind_group_layout_ids: Vec<BindGroupLayoutId>,
}

#[repr(C)]
pub struct BufferBinding {
    pub buffer: BufferId,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub enum BindingResource {
    Buffer(BufferBinding),
    Sampler(SamplerId),
    TextureView(TextureViewId),
}

#[repr(C)]
pub struct Binding {
    pub binding: u32,
    pub resource: BindingResource,
}

#[repr(C)]
pub struct BindGroupDescriptor {
    pub layout: BindGroupLayoutId,
    pub bindings: *const Binding,
    pub bindings_length: usize,
}

pub struct BindGroup<B: hal::Backend> {
    pub(crate) raw: B::DescriptorSet,
    pub(crate) layout_id: BindGroupLayoutId,
    pub(crate) life_guard: LifeGuard,
    pub(crate) used: TrackerSet,
}
