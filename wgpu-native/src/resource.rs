use {Extent3d};

use hal;


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

pub struct Buffer<B: hal::Backend> {
    pub(crate) raw: B::UnboundBuffer,
    pub(crate) memory_properties: hal::memory::Properties,
    // TODO: mapping, unmap()
}

pub struct TextureView {
    // TODO
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
    D32FloatS8Uint = 3,
}

bitflags! {
    #[repr(transparent)]
    pub struct TextureUsageFlags: u32 {
        const TRANSFER_SRC = 1;
        const TRANSFER_DST = 2;
        const SAMPLED = 4;
        const STORAGE = 8;
        const OUTPUT_ATTACHMENT = 16;
        const PRESENT = 32;
        const NONE = 0;
        const WRITE_ALL = 2 + 8 + 16;
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

pub struct Texture<B: hal::Backend> {
    pub(crate) raw: B::Image,
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
    pub mipmap_filer: FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub max_anisotropy: u32,
    pub compare_function: CompareFunction,
    pub border_color: BorderColor,
}

pub struct Sampler {
    // TODO
}
