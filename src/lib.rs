#[macro_use] extern crate bitflags;

#[cfg(feature = "gfx-backend-vulkan")]
extern crate gfx_backend_vulkan as back;
#[cfg(feature = "gfx-backend-dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "gfx-backend-metal")]
extern crate gfx_backend_metal as back;
#[cfg(not(any(feature = "gfx-backend-vulkan", feature = "gfx-backend-dx12", feature = "gfx-backend-metal")))]
extern crate gfx_backend_empty as back;

extern crate gfx_hal as hal;
extern crate gfx_memory as memory;

mod binding_model;
mod conv;
mod command;
mod device;
mod handle;
mod instance;
mod pipeline;
mod resource;

pub use self::binding_model::*;
pub use self::command::*;
pub use self::device::*;
pub use self::instance::*;
pub use self::pipeline::*;
pub use self::resource::*;

use back::Backend as B;
use handle::Handle;

#[repr(C)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[repr(C)]
pub struct Origin3d {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
pub struct Extent3d {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
}

pub type InstanceHandle = Handle<back::Instance>;
pub type AdapterHandle = Handle<hal::Adapter<B>>;
pub type DeviceHandle = Handle<Device<B>>;
pub type BufferHandle = Handle<Buffer<B>>;

// Resource
pub type TextureViewHandle = Handle<TextureView>;
pub type TextureHandle = Handle<Texture>;
pub type SamplerHandle = Handle<Sampler>;

// Binding model
pub type BindGroupLayoutHandle = Handle<BindGroupLayout>;
pub type PipelineLayoutHandle = Handle<PipelineLayout<B>>;

// Pipeline
pub type BlendStateHandle = Handle<BlendState>;
pub type DepthStencilStateHandle = Handle<DepthStencilState>;
pub type InputStateHandle = Handle<InputState>;
pub type ShaderModuleHandle = Handle<ShaderModule<B>>;
pub type AttachmentStateHandle = Handle<AttachmentState>;
pub type ComputePipelineHandle = Handle<ComputePipeline>;
pub type RenderPipelineHandle = Handle<RenderPipeline>;

pub type CommandBufferHandle = Handle<CommandBuffer<B>>;
pub type RenderPassHandle = Handle<RenderPass<B>>;
pub type ComputePassHandle = Handle<ComputePass<B>>;
