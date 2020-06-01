/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "peek-poke")]
use peek_poke::PeekPoke;
#[cfg(feature = "replay")]
use serde::Deserialize;
#[cfg(feature = "trace")]
use serde::Serialize;
use std::{io, ptr, slice};

/// Buffer-Texture copies on command encoders have to have the `bytes_per_row`
/// aligned to this number.
///
/// This doesn't apply to `Queue::write_texture`.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
/// Bound uniform/storage buffer offsets must be aligned to this number.
pub const BIND_BUFFER_ALIGNMENT: u64 = 256;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "peek-poke", derive(PeekPoke))]
#[cfg_attr(
    feature = "trace",
    derive(serde::Serialize),
    serde(into = "SerBufferSize")
)]
#[cfg_attr(
    feature = "replay",
    derive(serde::Deserialize),
    serde(from = "SerBufferSize")
)]
pub struct BufferSize(pub u64);

impl BufferSize {
    pub const WHOLE: BufferSize = BufferSize(!0u64);
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum Backend {
    Empty = 0,
    Vulkan = 1,
    Metal = 2,
    Dx12 = 3,
    Dx11 = 4,
    Gl = 5,
    BrowserWebGpu = 6,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum PowerPreference {
    Default = 0,
    LowPower = 1,
    HighPerformance = 2,
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
    #[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct BackendBit: u32 {
        const VULKAN = 1 << Backend::Vulkan as u32;
        const GL = 1 << Backend::Gl as u32;
        const METAL = 1 << Backend::Metal as u32;
        const DX12 = 1 << Backend::Dx12 as u32;
        const DX11 = 1 << Backend::Dx11 as u32;
        const BROWSER_WEBGPU = 1 << Backend::BrowserWebGpu as u32;
        /// Vulkan + Metal + DX12 + Browser WebGPU
        const PRIMARY = Self::VULKAN.bits
            | Self::METAL.bits
            | Self::DX12.bits
            | Self::BROWSER_WEBGPU.bits;
        /// OpenGL + DX11
        const SECONDARY = Self::GL.bits | Self::DX11.bits;
    }
}

impl From<Backend> for BackendBit {
    fn from(backend: Backend) -> Self {
        BackendBit::from_bits(1 << backend as u32).unwrap()
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct Extensions {
    /// This is a native only extension. Support is planned to be added to webgpu,
    /// but it is not yet implemented.
    ///
    /// https://github.com/gpuweb/gpuweb/issues/696
    pub anisotropic_filtering: bool,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct Limits {
    pub max_bind_groups: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_bind_groups: 4 }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct DeviceDescriptor {
    pub extensions: Extensions,
    pub limits: Limits,
}

// TODO: This is copy/pasted from gfx-hal, so we need to find a new place to put
// this function
pub fn read_spirv<R: io::Read + io::Seek>(mut x: R) -> io::Result<Vec<u32>> {
    let size = x.seek(io::SeekFrom::End(0))?;
    if size % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input length not divisible by 4",
        ));
    }
    if size > usize::max_value() as u64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "input too long"));
    }
    let words = (size / 4) as usize;
    let mut result = Vec::<u32>::with_capacity(words);
    x.seek(io::SeekFrom::Start(0))?;
    unsafe {
        // Writing all bytes through a pointer with less strict alignment when our type has no
        // invalid bitpatterns is safe.
        x.read_exact(slice::from_raw_parts_mut(
            result.as_mut_ptr() as *mut u8,
            words * 4,
        ))?;
        result.set_len(words);
    }
    const MAGIC_NUMBER: u32 = 0x0723_0203;
    if !result.is_empty() && result[0] == MAGIC_NUMBER.swap_bytes() {
        for word in &mut result {
            *word = word.swap_bytes();
        }
    }
    if result.is_empty() || result[0] != MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input missing SPIR-V magic number",
        ));
    }
    Ok(result)
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct ShaderStage: u32 {
        const NONE = 0;
        const VERTEX = 1;
        const FRAGMENT = 2;
        const COMPUTE = 4;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

pub type BufferAddress = u64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BlendFactor {
    Zero = 0,
    One = 1,
    SrcColor = 2,
    OneMinusSrcColor = 3,
    SrcAlpha = 4,
    OneMinusSrcAlpha = 5,
    DstColor = 6,
    OneMinusDstColor = 7,
    DstAlpha = 8,
    OneMinusDstAlpha = 9,
    SrcAlphaSaturated = 10,
    BlendColor = 11,
    OneMinusBlendColor = 12,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BlendOperation {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
}

impl Default for BlendOperation {
    fn default() -> Self {
        BlendOperation::Add
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BlendDescriptor {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl BlendDescriptor {
    pub const REPLACE: Self = BlendDescriptor {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };

    pub fn uses_color(&self) -> bool {
        match (self.src_factor, self.dst_factor) {
            (BlendFactor::BlendColor, _)
            | (BlendFactor::OneMinusBlendColor, _)
            | (_, BlendFactor::BlendColor)
            | (_, BlendFactor::OneMinusBlendColor) => true,
            (_, _) => false,
        }
    }
}

impl Default for BlendDescriptor {
    fn default() -> Self {
        BlendDescriptor::REPLACE
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ColorStateDescriptor {
    pub format: TextureFormat,
    pub alpha_blend: BlendDescriptor,
    pub color_blend: BlendDescriptor,
    pub write_mask: ColorWrite,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum PrimitiveTopology {
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum FrontFace {
    Ccw = 0,
    Cw = 1,
}

impl Default for FrontFace {
    fn default() -> Self {
        FrontFace::Ccw
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum CullMode {
    None = 0,
    Front = 1,
    Back = 2,
}

impl Default for CullMode {
    fn default() -> Self {
        CullMode::None
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RasterizationStateDescriptor {
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub depth_bias: i32,
    pub depth_bias_slope_scale: f32,
    pub depth_bias_clamp: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureFormat {
    // Normal 8 bit formats
    R8Unorm = 0,
    R8Snorm = 1,
    R8Uint = 2,
    R8Sint = 3,

    // Normal 16 bit formats
    R16Uint = 4,
    R16Sint = 5,
    R16Float = 6,
    Rg8Unorm = 7,
    Rg8Snorm = 8,
    Rg8Uint = 9,
    Rg8Sint = 10,

    // Normal 32 bit formats
    R32Uint = 11,
    R32Sint = 12,
    R32Float = 13,
    Rg16Uint = 14,
    Rg16Sint = 15,
    Rg16Float = 16,
    Rgba8Unorm = 17,
    Rgba8UnormSrgb = 18,
    Rgba8Snorm = 19,
    Rgba8Uint = 20,
    Rgba8Sint = 21,
    Bgra8Unorm = 22,
    Bgra8UnormSrgb = 23,

    // Packed 32 bit formats
    Rgb10a2Unorm = 24,
    Rg11b10Float = 25,

    // Normal 64 bit formats
    Rg32Uint = 26,
    Rg32Sint = 27,
    Rg32Float = 28,
    Rgba16Uint = 29,
    Rgba16Sint = 30,
    Rgba16Float = 31,

    // Normal 128 bit formats
    Rgba32Uint = 32,
    Rgba32Sint = 33,
    Rgba32Float = 34,

    // Depth and stencil formats
    Depth32Float = 35,
    Depth24Plus = 36,
    Depth24PlusStencil8 = 37,
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct ColorWrite: u32 {
        const RED = 1;
        const GREEN = 2;
        const BLUE = 4;
        const ALPHA = 8;
        const COLOR = 7;
        const ALL = 15;
    }
}

impl Default for ColorWrite {
    fn default() -> Self {
        ColorWrite::ALL
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct DepthStencilStateDescriptor {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
    pub stencil_front: StencilStateFaceDescriptor,
    pub stencil_back: StencilStateFaceDescriptor,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
}

impl DepthStencilStateDescriptor {
    pub fn needs_stencil_reference(&self) -> bool {
        !self.stencil_front.compare.is_trivial() || !self.stencil_back.compare.is_trivial()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum IndexFormat {
    Uint16 = 0,
    Uint32 = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum StencilOperation {
    Keep = 0,
    Zero = 1,
    Replace = 2,
    Invert = 3,
    IncrementClamp = 4,
    DecrementClamp = 5,
    IncrementWrap = 6,
    DecrementWrap = 7,
}

impl Default for StencilOperation {
    fn default() -> Self {
        StencilOperation::Keep
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct StencilStateFaceDescriptor {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

impl StencilStateFaceDescriptor {
    pub const IGNORE: Self = StencilStateFaceDescriptor {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };
}

impl Default for StencilStateFaceDescriptor {
    fn default() -> Self {
        StencilStateFaceDescriptor::IGNORE
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum CompareFunction {
    Undefined = 0,
    Never = 1,
    Less = 2,
    Equal = 3,
    LessEqual = 4,
    Greater = 5,
    NotEqual = 6,
    GreaterEqual = 7,
    Always = 8,
}

impl CompareFunction {
    pub fn is_trivial(self) -> bool {
        match self {
            CompareFunction::Never | CompareFunction::Always => true,
            _ => false,
        }
    }
}

pub type ShaderLocation = u32;

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum InputStepMode {
    Vertex = 0,
    Instance = 1,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct VertexAttributeDescriptor {
    pub offset: BufferAddress,
    pub format: VertexFormat,
    pub shader_location: ShaderLocation,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum VertexFormat {
    Uchar2 = 0,
    Uchar4 = 1,
    Char2 = 2,
    Char4 = 3,
    Uchar2Norm = 4,
    Uchar4Norm = 5,
    Char2Norm = 6,
    Char4Norm = 7,
    Ushort2 = 8,
    Ushort4 = 9,
    Short2 = 10,
    Short4 = 11,
    Ushort2Norm = 12,
    Ushort4Norm = 13,
    Short2Norm = 14,
    Short4Norm = 15,
    Half2 = 16,
    Half4 = 17,
    Float = 18,
    Float2 = 19,
    Float3 = 20,
    Float4 = 21,
    Uint = 22,
    Uint2 = 23,
    Uint3 = 24,
    Uint4 = 25,
    Int = 26,
    Int2 = 27,
    Int3 = 28,
    Int4 = 29,
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
    #[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct BufferUsage: u32 {
        const MAP_READ = 1;
        const MAP_WRITE = 2;
        const COPY_SRC = 4;
        const COPY_DST = 8;
        const INDEX = 16;
        const VERTEX = 32;
        const UNIFORM = 64;
        const STORAGE = 128;
        const INDIRECT = 256;
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BufferDescriptor<L> {
    pub label: L,
    pub size: BufferAddress,
    pub usage: BufferUsage,
}

impl<L> BufferDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> BufferDescriptor<K> {
        BufferDescriptor {
            label: fun(&self.label),
            size: self.size,
            usage: self.usage,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommandEncoderDescriptor {
    // MSVC doesn't allow zero-sized structs
    // We can remove this when we actually have a field
    // pub todo: u32,
    pub label: *const std::os::raw::c_char,
}

impl Default for CommandEncoderDescriptor {
    fn default() -> CommandEncoderDescriptor {
        CommandEncoderDescriptor { label: ptr::null() }
    }
}

pub type DynamicOffset = u32;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum PresentMode {
    /// The presentation engine does **not** wait for a vertical blanking period and
    /// the request is presented immediately. This is a low-latency presentation mode,
    /// but visible tearing may be observed. Will fallback to `Fifo` if unavailable on the
    /// selected  platform and backend. Not optimal for mobile.
    Immediate = 0,
    /// The presentation engine waits for the next vertical blanking period to update
    /// the current image, but frames may be submitted without delay. This is a low-latency
    /// presentation mode and visible tearing will **not** be observed. Will fallback to `Fifo`
    /// if unavailable on the selected platform and backend. Not optimal for mobile.
    Mailbox = 1,
    /// The presentation engine waits for the next vertical blanking period to update
    /// the current image. The framerate will be capped at the display refresh rate,
    /// corresponding to the `VSync`. Tearing cannot be observed. Optimal for mobile.
    Fifo = 2,
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct TextureUsage: u32 {
        const COPY_SRC = 1;
        const COPY_DST = 2;
        const SAMPLED = 4;
        const STORAGE = 8;
        const OUTPUT_ATTACHMENT = 16;
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct SwapChainDescriptor {
    pub usage: TextureUsage,
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
    pub present_mode: PresentMode,
}

#[repr(C)]
#[derive(Debug)]
pub enum SwapChainStatus {
    Good,
    Suboptimal,
    Timeout,
    Outdated,
    Lost,
    OutOfMemory,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "peek-poke", derive(PeekPoke))]
pub enum LoadOp {
    Clear = 0,
    Load = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "peek-poke", derive(PeekPoke))]
pub enum StoreOp {
    Clear = 0,
    Store = 1,
}

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "peek-poke", derive(PeekPoke))]
pub struct RenderPassColorAttachmentDescriptorBase<T> {
    pub attachment: T,
    pub resolve_target: Option<T>,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: Color,
}

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "peek-poke", derive(PeekPoke))]
pub struct RenderPassDepthStencilAttachmentDescriptorBase<T> {
    pub attachment: T,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub clear_depth: f32,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
    pub clear_stencil: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "peek-poke", derive(PeekPoke))]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
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
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct Origin3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Origin3d {
    pub const ZERO: Self = Origin3d { x: 0, y: 0, z: 0 };
}

impl Default for Origin3d {
    fn default() -> Self {
        Origin3d::ZERO
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct TextureDescriptor<L> {
    pub label: L,
    pub size: Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsage,
}

impl<L> TextureDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> TextureDescriptor<K> {
        TextureDescriptor {
            label: fun(&self.label),
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: self.sample_count,
            dimension: self.dimension,
            format: self.format,
            usage: self.usage,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureAspect {
    All,
    StencilOnly,
    DepthOnly,
}

impl Default for TextureAspect {
    fn default() -> Self {
        TextureAspect::All
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct TextureViewDescriptor<L> {
    pub label: L,
    pub format: TextureFormat,
    pub dimension: TextureViewDimension,
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
}

impl<L> TextureViewDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> TextureViewDescriptor<K> {
        TextureViewDescriptor {
            label: fun(&self.label),
            format: self.format,
            dimension: self.dimension,
            aspect: self.aspect,
            base_mip_level: self.base_mip_level,
            level_count: self.level_count,
            base_array_layer: self.base_array_layer,
            array_layer_count: self.array_layer_count,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum AddressMode {
    ClampToEdge = 0,
    Repeat = 1,
    MirrorRepeat = 2,
}

impl Default for AddressMode {
    fn default() -> Self {
        AddressMode::ClampToEdge
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum FilterMode {
    Nearest = 0,
    Linear = 1,
}

impl Default for FilterMode {
    fn default() -> Self {
        FilterMode::Nearest
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct SamplerDescriptor<L> {
    pub label: L,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: CompareFunction,
    /// Anisotropic filtering extension must be enabled if this value is
    /// anything other than 0 and 1.
    ///
    /// Valid values are 0, 1, 2, 4, 8, and 16.
    pub anisotropy_clamp: u8,
}

impl<L> SamplerDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> SamplerDescriptor<K> {
        SamplerDescriptor {
            label: fun(&self.label),
            address_mode_u: self.address_mode_u,
            address_mode_v: self.address_mode_v,
            address_mode_w: self.address_mode_w,
            mag_filter: self.mag_filter,
            min_filter: self.min_filter,
            mipmap_filter: self.mipmap_filter,
            lod_min_clamp: self.lod_min_clamp,
            lod_max_clamp: self.lod_max_clamp,
            compare: self.compare,
            anisotropy_clamp: self.anisotropy_clamp,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct CommandBufferDescriptor {
    pub todo: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureComponentType {
    Float,
    Sint,
    Uint,
}

impl From<TextureFormat> for TextureComponentType {
    fn from(format: TextureFormat) -> Self {
        match format {
            TextureFormat::R8Uint
            | TextureFormat::R16Uint
            | TextureFormat::Rg8Uint
            | TextureFormat::R32Uint
            | TextureFormat::Rg16Uint
            | TextureFormat::Rgba8Uint
            | TextureFormat::Rg32Uint
            | TextureFormat::Rgba16Uint
            | TextureFormat::Rgba32Uint => Self::Uint,

            TextureFormat::R8Sint
            | TextureFormat::R16Sint
            | TextureFormat::Rg8Sint
            | TextureFormat::R32Sint
            | TextureFormat::Rg16Sint
            | TextureFormat::Rgba8Sint
            | TextureFormat::Rg32Sint
            | TextureFormat::Rgba16Sint
            | TextureFormat::Rgba32Sint => Self::Sint,

            TextureFormat::R8Unorm
            | TextureFormat::R8Snorm
            | TextureFormat::R16Float
            | TextureFormat::R32Float
            | TextureFormat::Rg8Unorm
            | TextureFormat::Rg8Snorm
            | TextureFormat::Rg16Float
            | TextureFormat::Rg11b10Float
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba8Snorm
            | TextureFormat::Rgba16Float
            | TextureFormat::Rgba32Float
            | TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgb10a2Unorm
            | TextureFormat::Depth32Float
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8 => Self::Float,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct TextureDataLayout {
    pub offset: BufferAddress,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
}

/// This type allows us to make the serialized representation of a BufferSize more human-readable
#[allow(dead_code)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
enum SerBufferSize {
    Size(u64),
    Whole,
}

#[cfg(feature = "trace")]
impl From<BufferSize> for SerBufferSize {
    fn from(buffer_size: BufferSize) -> Self {
        if buffer_size == BufferSize::WHOLE {
            Self::Whole
        } else {
            Self::Size(buffer_size.0)
        }
    }
}

#[cfg(feature = "replay")]
impl From<SerBufferSize> for BufferSize {
    fn from(ser_buffer_size: SerBufferSize) -> Self {
        match ser_buffer_size {
            SerBufferSize::Size(size) => BufferSize(size),
            SerBufferSize::Whole => BufferSize::WHOLE,
        }
    }
}
