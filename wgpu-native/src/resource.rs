use crate::{
    Extent3d, LifeGuard, RefCount, Stored,
    DeviceId, TextureId,
    BufferMapReadCallback, BufferMapWriteCallback,
};
use crate::swap_chain::{SwapChainLink, SwapImageEpoch};

use bitflags::bitflags;
use hal;
use parking_lot::Mutex;

use std::borrow::Borrow;

bitflags! {
    #[repr(transparent)]
    pub struct BufferUsageFlags: u32 {
        const MAP_READ = 1;
        const MAP_WRITE = 2;
        const TRANSFER_SRC = 4;
        const TRANSFER_DST = 8;
        const INDEX = 16;
        const VERTEX = 32;
        const UNIFORM = 64;
        const STORAGE = 128;
        const NONE = 0;
        const WRITE_ALL = 2 + 8 + 128;
    }
}

#[repr(C)]
pub struct BufferDescriptor {
    pub size: u32,
    pub usage: BufferUsageFlags,
}

pub enum BufferMapAsyncStatus {
    Success,
    Error,
    Unknown,
    ContextLost,
}

pub(crate) enum BufferMapOperation {
    Read(std::ops::Range<u64>, BufferMapReadCallback, *mut u8),
    Write(std::ops::Range<u64>, BufferMapWriteCallback, *mut u8),
}

unsafe impl Send for BufferMapOperation {}
unsafe impl Sync for BufferMapOperation {}

pub struct Buffer<B: hal::Backend> {
    pub(crate) raw: B::Buffer,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) memory_properties: hal::memory::Properties,
    pub(crate) memory: B::Memory,
    pub(crate) mapped_write_ranges: Vec<std::ops::Range<u64>>,
    pub(crate) pending_map_operation: Option<BufferMapOperation>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Borrow<RefCount> for Buffer<B> {
    fn borrow(&self) -> &RefCount {
        &self.life_guard.ref_count
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum TextureFormat {
    R8g8b8a8Unorm = 0,
    R8g8b8a8Uint = 1,
    B8g8r8a8Unorm = 2,
    D32Float = 3,
    D32FloatS8Uint = 4,
}

bitflags! {
    #[repr(transparent)]
    pub struct TextureUsageFlags: u32 {
        const TRANSFER_SRC = 1;
        const TRANSFER_DST = 2;
        const SAMPLED = 4;
        const STORAGE = 8;
        const OUTPUT_ATTACHMENT = 16;
        const NONE = 0;
        const WRITE_ALL = 2 + 8 + 16;
        const UNINITIALIZED = 0xFFFF;
    }
}

#[repr(C)]
pub struct TextureDescriptor {
    pub size: Extent3d,
    pub array_size: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsageFlags,
}

pub(crate) enum TexturePlacement<B: hal::Backend> {
    SwapChain(SwapChainLink<Mutex<SwapImageEpoch>>),
    Memory(B::Memory),
    Void,
}

impl<B: hal::Backend> TexturePlacement<B> {
    pub fn as_swap_chain(&self) -> &SwapChainLink<Mutex<SwapImageEpoch>> {
        match *self {
            TexturePlacement::SwapChain(ref link) => link,
            TexturePlacement::Memory(_) |
            TexturePlacement::Void => panic!("Expected swap chain link!"),
        }
    }
}

pub struct Texture<B: hal::Backend> {
    pub(crate) raw: B::Image,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) kind: hal::image::Kind,
    pub(crate) format: TextureFormat,
    pub(crate) full_range: hal::image::SubresourceRange,
    pub(crate) placement: TexturePlacement<B>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Borrow<RefCount> for Texture<B> {
    fn borrow(&self) -> &RefCount {
        &self.life_guard.ref_count
    }
}

bitflags! {
    #[repr(transparent)]
    pub struct TextureAspectFlags: u32 {
        const COLOR = 1;
        const DEPTH = 2;
        const STENCIL = 4;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[repr(C)]
pub struct TextureViewDescriptor {
    pub format: TextureFormat,
    pub dimension: TextureViewDimension,
    pub aspect: TextureAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub array_count: u32,
}

pub struct TextureView<B: hal::Backend> {
    pub(crate) raw: B::ImageView,
    pub(crate) texture_id: Stored<TextureId>,
    //TODO: store device_id for quick access?
    pub(crate) format: TextureFormat,
    pub(crate) extent: hal::image::Extent,
    pub(crate) samples: hal::image::NumSamples,
    pub(crate) is_owned_by_swap_chain: bool,
    #[cfg_attr(not(feature = "local"), allow(dead_code))]
    pub(crate) life_guard: LifeGuard,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum AddressMode {
    ClampToEdge = 0,
    Repeat = 1,
    MirrorRepeat = 2,
    ClampToBorderColor = 3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum FilterMode {
    Nearest = 0,
    Linear = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum CompareFunction {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum BorderColor {
    TransparentBlack = 0,
    OpaqueBlack = 1,
    OpaqueWhite = 2,
}

#[repr(C)]
pub struct SamplerDescriptor {
    pub r_address_mode: AddressMode,
    pub s_address_mode: AddressMode,
    pub t_address_mode: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub max_anisotropy: u32,
    pub compare_function: CompareFunction,
    pub border_color: BorderColor,
}

pub struct Sampler<B: hal::Backend> {
    pub(crate) raw: B::Sampler,
}
