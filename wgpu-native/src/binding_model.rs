use hal;

use {BindGroupLayoutId, BufferId, SamplerId, TextureViewId};

bitflags! {
    #[repr(transparent)]
    pub struct ShaderStageFlags: u32 {
        const NONE = 0;
        const VERTEX = 1;
        const FRAGMENT = 2;
        const COMPUTE = 4;
    }
}

#[repr(C)]
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
pub struct BindGroupLayoutDescriptor<'a> {
    pub bindings: &'a [BindGroupLayoutBinding],
}

pub struct BindGroupLayout {
    // TODO
}

#[repr(C)]
pub struct PipelineLayoutDescriptor<'a> {
    pub bind_group_layouts: &'a [BindGroupLayoutId],
}

pub struct PipelineLayout<B: hal::Backend> {
    raw: B::PipelineLayout,
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
pub struct BindGroupDescriptor<'a> {
    pub layout: BindGroupLayout,
    pub bindings: &'a [Binding],
}

pub struct BindGroup {
    // TODO
}
