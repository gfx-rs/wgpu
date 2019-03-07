#[cfg(all(feature = "local", feature = "window-winit"))]
pub extern crate winit;

#[cfg(feature = "gfx-backend-dx11")]
extern crate gfx_backend_dx11 as back;
#[cfg(feature = "gfx-backend-dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(not(any(
    feature = "gfx-backend-vulkan",
    feature = "gfx-backend-dx11",
    feature = "gfx-backend-dx12",
    feature = "gfx-backend-metal",
)))]
extern crate gfx_backend_empty as back;
#[cfg(feature = "gfx-backend-metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "gfx-backend-vulkan")]
extern crate gfx_backend_vulkan as back;

extern crate gfx_hal as hal;
//extern crate rendy_memory;

mod binding_model;
mod command;
mod conv;
mod device;
mod hub;
mod instance;
mod pipeline;
mod resource;
mod swap_chain;
mod track;

pub use self::binding_model::*;
pub use self::command::*;
pub use self::device::*;
#[cfg(feature = "remote")]
pub use self::hub::{Id, IdentityManager, Registry, HUB};
pub use self::instance::*;
pub use self::pipeline::*;
pub use self::resource::*;
pub use self::swap_chain::*;

use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

type SubmissionIndex = usize;

//TODO: make it private. Currently used for swapchain creation impl.
#[derive(Debug)]
pub struct RefCount(ptr::NonNull<AtomicUsize>);

unsafe impl Send for RefCount {}
unsafe impl Sync for RefCount {}

impl RefCount {
    const MAX: usize = 1 << 24;

    fn load(&self) -> usize {
        unsafe { self.0.as_ref() }.load(Ordering::Acquire)
    }
}

impl Clone for RefCount {
    fn clone(&self) -> Self {
        let old_size = unsafe { self.0.as_ref() }.fetch_add(1, Ordering::Relaxed);
        assert!(old_size < Self::MAX);
        RefCount(self.0)
    }
}

impl Drop for RefCount {
    fn drop(&mut self) {
        if unsafe { self.0.as_ref() }.fetch_sub(1, Ordering::Relaxed) == 1 {
            let _ = unsafe { Box::from_raw(self.0.as_ptr()) };
        }
    }
}

struct LifeGuard {
    ref_count: RefCount,
    submission_index: AtomicUsize,
}

impl LifeGuard {
    fn new() -> Self {
        let bx = Box::new(AtomicUsize::new(1));
        LifeGuard {
            ref_count: RefCount(ptr::NonNull::new(Box::into_raw(bx)).unwrap()),
            submission_index: AtomicUsize::new(0),
        }
    }
}

#[derive(Clone, Debug)]
struct Stored<T> {
    value: T,
    ref_count: RefCount,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const TRANSPARENT: Self = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    pub const BLACK: Self = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Self = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const RED: Self = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const GREEN: Self = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const BLUE: Self = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
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
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
pub struct ByteArray {
    pub bytes: *const u8,
    pub length: usize,
}

pub type InstanceId = hub::Id;
type InstanceHandle = back::Instance;
pub type AdapterId = hub::Id;
type AdapterHandle = hal::Adapter<back::Backend>;
pub type DeviceId = hub::Id;
type DeviceHandle = Device<back::Backend>;
pub type QueueId = DeviceId;
pub type BufferId = hub::Id;
type BufferHandle = Buffer<back::Backend>;
// Resource
pub type TextureViewId = hub::Id;
type TextureViewHandle = TextureView<back::Backend>;
pub type TextureId = hub::Id;
type TextureHandle = Texture<back::Backend>;
pub type SamplerId = hub::Id;
type SamplerHandle = Sampler<back::Backend>;
// Binding model
pub type BindGroupLayoutId = hub::Id;
type BindGroupLayoutHandle = BindGroupLayout<back::Backend>;
pub type PipelineLayoutId = hub::Id;
type PipelineLayoutHandle = PipelineLayout<back::Backend>;
pub type BindGroupId = hub::Id;
type BindGroupHandle = BindGroup<back::Backend>;
// Pipeline
pub type InputStateId = hub::Id;
pub type ShaderModuleId = hub::Id;
type ShaderModuleHandle = ShaderModule<back::Backend>;
pub type RenderPipelineId = hub::Id;
type RenderPipelineHandle = RenderPipeline<back::Backend>;
pub type ComputePipelineId = hub::Id;
type ComputePipelineHandle = ComputePipeline<back::Backend>;
// Command
pub type CommandBufferId = hub::Id;
type CommandBufferHandle = CommandBuffer<back::Backend>;
pub type CommandEncoderId = CommandBufferId;
pub type RenderPassId = hub::Id;
type RenderPassHandle = RenderPass<back::Backend>;
pub type ComputePassId = hub::Id;
type ComputePassHandle = ComputePass<back::Backend>;
// Swap chain
pub type SurfaceId = hub::Id;
type SurfaceHandle = Surface<back::Backend>;
pub type SwapChainId = SurfaceId;
