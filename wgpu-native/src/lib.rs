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
pub use self::hub::{IdentityManager, Registry, HUB};
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

macro_rules! transparent {
    ($i:item) => (
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Hash, PartialEq)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        $i
    )
}

pub trait ToId {
    fn id(&self) -> hub::Id;
}

macro_rules! to_id {
    ($i:ident) => (
        impl ToId for $i {
            fn id(&self) -> hub::Id {
                self.0
            }
        }
    )
}

macro_rules! from_id {
    ($i:ident) => (
        impl From<hub::Id> for $i {
            fn from(id:hub::Id) -> $i {
                $i(id)
            }
        }
    )
}

use hub::{Index, Epoch, NewId, Id};
macro_rules! new_id {
    ($i:ident) => (
        impl NewId for $i {
            fn new(index: Index, epoch: Epoch) -> Self {
                let id = Id::new(index, epoch);
                $i(id)
            }

            fn index(&self) -> Index {
                (self.id()).index()
            }

            fn epoch(&self) -> Epoch {
               (self.id()).epoch()
            }
        }
    )
}

transparent!(pub struct InstanceId(hub::Id););
to_id!(InstanceId);
from_id!(InstanceId);
type InstanceHandle = back::Instance;

transparent!(pub struct AdapterId(hub::Id););
to_id!(AdapterId);
from_id!(AdapterId);
type AdapterHandle = hal::Adapter<back::Backend>;

transparent!(pub struct DeviceId(hub::Id););
to_id!(DeviceId);
from_id!(DeviceId);
type DeviceHandle = Device<back::Backend>;
//transparent!(pub struct QueueId(DeviceId););
pub type QueueId = DeviceId;

transparent!(pub struct BufferId(hub::Id););
to_id!(BufferId);
from_id!(BufferId);
new_id!(BufferId);
type BufferHandle = Buffer<back::Backend>;

// Resource
transparent!(pub struct TextureViewId(hub::Id););
to_id!(TextureViewId);
from_id!(TextureViewId);
new_id!(TextureViewId);
type TextureViewHandle = TextureView<back::Backend>;

transparent!(pub struct TextureId(hub::Id););
to_id!(TextureId);
from_id!(TextureId);
new_id!(TextureId);
type TextureHandle = Texture<back::Backend>;

transparent!(pub struct SamplerId(hub::Id););
to_id!(SamplerId);
from_id!(SamplerId);
type SamplerHandle = Sampler<back::Backend>;

// Binding model
transparent!(pub struct BindGroupLayoutId(hub::Id););
to_id!(BindGroupLayoutId);
from_id!(BindGroupLayoutId);
type BindGroupLayoutHandle = BindGroupLayout<back::Backend>;

transparent!(pub struct PipelineLayoutId(hub::Id););
to_id!(PipelineLayoutId);
from_id!(PipelineLayoutId);
type PipelineLayoutHandle = PipelineLayout<back::Backend>;

transparent!(pub struct BindGroupId(hub::Id););
to_id!(BindGroupId);
from_id!(BindGroupId);
type BindGroupHandle = BindGroup<back::Backend>;

// Pipeline
transparent!(pub struct InputStateId(hub::Id););
to_id!(InputStateId);
from_id!(InputStateId);
transparent!(pub struct ShaderModuleId(hub::Id););
to_id!(ShaderModuleId);
from_id!(ShaderModuleId);
type ShaderModuleHandle = ShaderModule<back::Backend>;

transparent!(pub struct RenderPipelineId(hub::Id););
to_id!(RenderPipelineId);
from_id!(RenderPipelineId);
type RenderPipelineHandle = RenderPipeline<back::Backend>;

transparent!(pub struct ComputePipelineId(hub::Id););
to_id!(ComputePipelineId);
from_id!(ComputePipelineId);
type ComputePipelineHandle = ComputePipeline<back::Backend>;

// Command
transparent!(pub struct CommandBufferId(hub::Id););
to_id!(CommandBufferId);
from_id!(CommandBufferId);
type CommandBufferHandle = CommandBuffer<back::Backend>;
//transparent!(pub struct CommandEncoderId(CommandBufferId););
pub type CommandEncoderId = CommandBufferId;

transparent!(pub struct RenderPassId(hub::Id););
to_id!(RenderPassId);
from_id!(RenderPassId);
type RenderPassHandle = RenderPass<back::Backend>;

transparent!(pub struct ComputePassId(hub::Id););
to_id!(ComputePassId);
from_id!(ComputePassId);
type ComputePassHandle = ComputePass<back::Backend>;

// Swap chain
transparent!(pub struct SurfaceId(hub::Id););
to_id!(SurfaceId);
from_id!(SurfaceId);
type SurfaceHandle = Surface<back::Backend>;
//transparent!(pub struct SwapChainId(SurfaceId););
pub type SwapChainId = SurfaceId;
