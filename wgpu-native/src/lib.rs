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
    feature = "gfx-backend-gl",
)))]
extern crate gfx_backend_empty as back;
#[cfg(feature = "gfx-backend-metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "gfx-backend-vulkan")]
extern crate gfx_backend_vulkan as back;
#[cfg(feature = "gfx-backend-gl")]
extern crate gfx_backend_gl as back;

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
#[cfg(not(feature = "local"))]
pub use self::hub::{Access, IdentityManager, Registry, Token, HUB};
pub use self::instance::*;
pub use self::pipeline::*;
pub use self::resource::*;
pub use self::swap_chain::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "gfx-backend-gl")]
pub use back::glutin;

use std::{
    mem,
    os::raw::c_char,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

type SubmissionIndex = usize;
pub(crate) type Index = u32;
pub(crate) type Epoch = u32;

pub type BufferAddress = u64;
pub type RawString = *const c_char;

/// A trait for plain-old-data types.
///
/// A POD type does not have invalid bit patterns and can be safely
/// created from arbitrary bit pattern.
/// The `Pod` trait is implemented for standard integer and floating point numbers as well as
/// common arrays of them (for example `[f32; 2]`).
pub unsafe trait Pod: Copy {}

macro_rules! impl_pod {
    ( ty = $($ty:ty)* ) => { $( unsafe impl Pod for $ty {} )* };
    ( ar = $($tt:expr)* ) => { $( unsafe impl<T: Pod> Pod for [T; $tt] {} )* };
}

impl_pod! { ty = isize usize i8 u8 i16 u16 i32 u32 i64 u64 f32 f64 }
impl_pod! { ar =
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
}

unsafe impl<T: Pod, U: Pod> Pod for (T, U) {}

/// Cast a slice from one POD type to another.
pub fn cast_slice<A: Pod, B: Pod>(slice: &[A]) -> &[B] {
    use std::slice;

    let raw_len = mem::size_of::<A>().wrapping_mul(slice.len());
    let len = raw_len / mem::size_of::<B>();
    assert_eq!(raw_len, mem::size_of::<B>().wrapping_mul(len));
    unsafe { slice::from_raw_parts(slice.as_ptr() as *const B, len) }
}

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

#[derive(Debug)]
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

impl Origin3d {
    pub const ZERO: Self = Origin3d {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct ByteArray {
    pub bytes: *const u8,
    pub length: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct Id(Index, Epoch);

pub trait TypedId {
    fn new(index: Index, epoch: Epoch) -> Self;
    fn index(&self) -> Index;
    fn epoch(&self) -> Epoch;
}

macro_rules! define_id {
    ($i:ident) => {
        transparent!($i);
        typed_id!($i);
    };
}

macro_rules! transparent {
    ($i:ident) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Hash, PartialEq)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        pub struct $i(Id);
    };
}

macro_rules! typed_id {
    ($i:ident) => {
        impl $i {
            fn raw(&self) -> Id {
                self.0
            }
        }
        impl TypedId for $i {
            fn new(index: Index, epoch: Epoch) -> Self {
                let id = Id(index, epoch);
                $i(id)
            }

            fn index(&self) -> Index {
                (self.raw()).0
            }

            fn epoch(&self) -> Epoch {
                (self.raw()).1
            }
        }
    };
}

#[cfg(not(feature = "gfx-backend-gl"))]
define_id!(InstanceId);
#[cfg(not(feature = "gfx-backend-gl"))]
type InstanceHandle = back::Instance;
#[cfg(feature = "gfx-backend-gl")]
pub type InstanceId = SurfaceId;

define_id!(AdapterId);
type AdapterHandle = hal::Adapter<back::Backend>;

define_id!(DeviceId);
type DeviceHandle = Device<back::Backend>;
pub type QueueId = DeviceId;

define_id!(BufferId);
type BufferHandle = Buffer<back::Backend>;

// Resource
define_id!(TextureViewId);
type TextureViewHandle = TextureView<back::Backend>;

define_id!(TextureId);
type TextureHandle = Texture<back::Backend>;

define_id!(SamplerId);
type SamplerHandle = Sampler<back::Backend>;

// Binding model
define_id!(BindGroupLayoutId);
type BindGroupLayoutHandle = BindGroupLayout<back::Backend>;

define_id!(PipelineLayoutId);
type PipelineLayoutHandle = PipelineLayout<back::Backend>;

define_id!(BindGroupId);
type BindGroupHandle = BindGroup<back::Backend>;

// Pipeline
define_id!(InputStateId);
define_id!(ShaderModuleId);
type ShaderModuleHandle = ShaderModule<back::Backend>;

define_id!(RenderPipelineId);
type RenderPipelineHandle = RenderPipeline<back::Backend>;

define_id!(ComputePipelineId);
type ComputePipelineHandle = ComputePipeline<back::Backend>;

// Command
define_id!(CommandBufferId);
type CommandBufferHandle = CommandBuffer<back::Backend>;
pub type CommandEncoderId = CommandBufferId;

define_id!(RenderPassId);
type RenderPassHandle = RenderPass<back::Backend>;

define_id!(ComputePassId);
type ComputePassHandle = ComputePass<back::Backend>;

// Swap chain
define_id!(SurfaceId);
type SurfaceHandle = Surface<back::Backend>;
pub type SwapChainId = SurfaceId;
