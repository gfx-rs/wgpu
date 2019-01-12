#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
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
//extern crate rendy_memory;

mod binding_model;
mod command;
mod conv;
mod device;
mod instance;
mod pipeline;
mod registry;
mod resource;
mod track;

pub use self::binding_model::*;
pub use self::command::*;
pub use self::device::*;
pub use self::instance::*;
pub use self::pipeline::*;
pub use self::resource::*;

use back::Backend as B;
pub use crate::registry::Id;

use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

type SubmissionIndex = usize;

#[derive(Debug)]
struct RefCount(ptr::NonNull<AtomicUsize>);

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

//TODO: reconsider this
unsafe impl Send for LifeGuard {}
unsafe impl Sync for LifeGuard {}

impl LifeGuard {
    fn new() -> Self {
        let bx = Box::new(AtomicUsize::new(1));
        LifeGuard {
            ref_count: RefCount(ptr::NonNull::new(Box::into_raw(bx)).unwrap()),
            submission_index: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
struct Stored<T> {
    value: T,
    ref_count: RefCount,
}

unsafe impl<T> Send for Stored<T> {}
unsafe impl<T> Sync for Stored<T> {}

#[derive(Debug, Hash, PartialEq, Eq)]
struct WeaklyStored<T>(T);

unsafe impl<T> Send for WeaklyStored<T> {}
unsafe impl<T> Sync for WeaklyStored<T> {}

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

pub type InstanceId = Id;
type InstanceHandle = back::Instance;
pub type AdapterId = Id;
type AdapterHandle = hal::Adapter<B>;
pub type DeviceId = Id;
type DeviceHandle = Device<B>;
pub type QueueId = Id;
pub type BufferId = Id;
type BufferHandle = Buffer<B>;

// Resource
pub type TextureViewId = Id;
type TextureViewHandle = TextureView<B>;
pub type TextureId = Id;
type TextureHandle = Texture<B>;
pub type SamplerId = Id;

// Binding model
pub type BindGroupLayoutId = Id;
type BindGroupLayoutHandle = BindGroupLayout<B>;
pub type PipelineLayoutId = Id;
type PipelineLayoutHandle = PipelineLayout<B>;
pub type BindGroupId = Id;
type BindGroupHandle = BindGroup<B>;

// Pipeline
pub type BlendStateId = Id;
type BlendStateHandle = BlendState;
pub type DepthStencilStateId = Id;
type DepthStencilStateHandle = DepthStencilState;
pub type InputStateId = Id;
pub type ShaderModuleId = Id;
type ShaderModuleHandle = ShaderModule<B>;
pub type RenderPipelineId = Id;
type RenderPipelineHandle = RenderPipeline<B>;
pub type ComputePipelineId = Id;
type ComputePipelineHandle = ComputePipeline<B>;

pub type CommandBufferId = Id;
type CommandBufferHandle = CommandBuffer<B>;
pub type RenderPassId = Id;
type RenderPassHandle = RenderPass<B>;
pub type ComputePassId = Id;
type ComputePassHandle = ComputePass<B>;
