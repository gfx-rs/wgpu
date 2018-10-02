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

#[derive(Debug, PartialEq)]
struct Stored<T>(T);
#[cfg(not(feature = "remote"))]
unsafe impl<T> Sync for Stored<T> {}
#[cfg(not(feature = "remote"))]
unsafe impl<T> Send for Stored<T> {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const TRANSPARENT : Self = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
    pub const BLACK       : Self = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const WHITE       : Self = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED         : Self = Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN       : Self = Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE        : Self = Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Origin3d {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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
pub type QueueId = Id;
pub type BufferId = Id;

// Resource
pub type TextureViewId = Id;
pub type TextureId = Id;
pub type SamplerId = Id;

// Binding model
pub type BindGroupLayoutId = Id;
type BindGroupLayoutHandle = BindGroupLayout<B>;
pub type PipelineLayoutId = Id;
type PipelineLayoutHandle = PipelineLayout<B>;

// Pipeline
pub type BlendStateId = Id;
type BlendStateHandle = BlendState;
pub type DepthStencilStateId = Id;
type DepthStencilStateHandle = DepthStencilState;
pub type InputStateId = Id;
pub type ShaderModuleId = Id;
type ShaderModuleHandle = ShaderModule<B>;
pub type AttachmentStateId = Id;
type AttachmentStateHandle = AttachmentState<B>;
pub type ComputePipelineId = Id;
pub type RenderPipelineId = Id;
type RenderPipelineHandle = RenderPipeline<B>;

pub type CommandBufferId = Id;
type CommandBufferHandle = CommandBuffer<B>;
pub type RenderPassId = Id;
type RenderPassHandle = RenderPass<B>;
pub type ComputePassId = Id;
type ComputePassHandle = ComputePass<B>;
