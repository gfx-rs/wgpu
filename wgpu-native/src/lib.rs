#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate lazy_static;
#[cfg(feature = "remote")]
extern crate parking_lot;

#[cfg(feature = "gfx-backend-dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(not(any(
    feature = "gfx-backend-vulkan",
    feature = "gfx-backend-dx12",
    feature = "gfx-backend-metal"
)))]
extern crate gfx_backend_empty as back;
#[cfg(feature = "gfx-backend-metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "gfx-backend-vulkan")]
extern crate gfx_backend_vulkan as back;

extern crate gfx_hal as hal;
extern crate gfx_memory as memory;

mod binding_model;
mod command;
mod conv;
mod device;
mod instance;
mod pipeline;
mod registry;
mod resource;

pub use self::binding_model::*;
pub use self::command::*;
pub use self::device::*;
pub use self::instance::*;
pub use self::pipeline::*;
pub use self::resource::*;

use back::Backend as B;
use registry::Id;


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

#[repr(C)]
pub struct ByteArray {
    pub bytes: *const u8,
    pub length: usize,
}

pub type InstanceId = Id;
type InstanceHandle = back::Instance;
pub type AdapterId = Id;
type AdapterHandle = hal::Adapter<B>;
pub type DeviceId = Id;
type DeviceHandle = Device<B>;
pub type BufferId = Id;

// Resource
pub type TextureViewId = Id;
pub type TextureId = Id;
pub type SamplerId = Id;

// Binding model
pub type BindGroupLayoutId = Id;
pub type PipelineLayoutId = Id;

// Pipeline
pub type BlendStateId = Id;
pub type DepthStencilStateId = Id;
pub type InputStateId = Id;
pub type ShaderModuleId = Id;
type ShaderModuleHandle = ShaderModule<B>;
pub type AttachmentStateId = Id;
pub type ComputePipelineId = Id;
pub type RenderPipelineId = Id;

pub type CommandBufferId = Id;
type CommandBufferHandle = CommandBuffer<B>;
pub type RenderPassId = Id;
pub type ComputePassId = Id;
