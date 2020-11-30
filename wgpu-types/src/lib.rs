/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// The intra doc links to the wgpu crate in this crate actually succesfully link to the types in the wgpu crate, when built from the wgpu crate.
// However when building from both the wgpu crate or this crate cargo doc will claim all the links cannot be resolved
// despite the fact that it works fine when it needs to.
// So we just disable those warnings.
#![allow(broken_intra_doc_links)]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU32, ops::Range};

/// Integral type used for buffer offsets.
pub type BufferAddress = u64;
/// Integral type used for buffer slice sizes.
pub type BufferSize = std::num::NonZeroU64;
/// Integral type used for binding locations in shaders.
pub type ShaderLocation = u32;
/// Integral type used for dynamic bind group offsets.
pub type DynamicOffset = u32;

/// Buffer-Texture copies must have [`bytes_per_row`] aligned to this number.
///
/// This doesn't apply to [`Queue::write_texture`].
///
/// [`bytes_per_row`]: TextureDataLayout::bytes_per_row
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
/// Bound uniform/storage buffer offsets must be aligned to this number.
pub const BIND_BUFFER_ALIGNMENT: BufferAddress = 256;
/// Buffer to buffer copy offsets and sizes must be aligned to this number.
pub const COPY_BUFFER_ALIGNMENT: BufferAddress = 4;
/// Vertex buffer strides have to be aligned to this number.
pub const VERTEX_STRIDE_ALIGNMENT: BufferAddress = 4;
/// Alignment all push constants need
pub const PUSH_CONSTANT_ALIGNMENT: u32 = 4;

/// Backends supported by wgpu.
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

/// Power Preference when choosing a physical adapter.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum PowerPreference {
    /// Adapter that uses the least possible power. This is often an integerated GPU.
    LowPower = 0,
    /// Adapter that has the highest performance. This is often a discrete GPU.
    HighPerformance = 1,
}

impl Default for PowerPreference {
    fn default() -> Self {
        Self::LowPower
    }
}

bitflags::bitflags! {
    /// Represents the backends that wgpu will use.
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
    #[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct BackendBit: u32 {
        /// Supported on Windows, Linux/Android, and macOS/iOS via Vulkan Portability (with the Vulkan feature enabled)
        const VULKAN = 1 << Backend::Vulkan as u32;
        /// Currently unsupported
        const GL = 1 << Backend::Gl as u32;
        /// Supported on macOS/iOS
        const METAL = 1 << Backend::Metal as u32;
        /// Supported on Windows 10
        const DX12 = 1 << Backend::Dx12 as u32;
        /// Supported on Windows 7+
        const DX11 = 1 << Backend::Dx11 as u32;
        /// Supported when targeting the web through webassembly
        const BROWSER_WEBGPU = 1 << Backend::BrowserWebGpu as u32;
        /// All the apis that wgpu offers first tier of support for.
        ///
        /// Vulkan + Metal + DX12 + Browser WebGPU
        const PRIMARY = Self::VULKAN.bits
            | Self::METAL.bits
            | Self::DX12.bits
            | Self::BROWSER_WEBGPU.bits;
        /// All the apis that wgpu offers second tier of support for. These may
        /// be unsupported/still experimental.
        ///
        /// OpenGL + DX11
        const SECONDARY = Self::GL.bits | Self::DX11.bits;
    }
}

impl From<Backend> for BackendBit {
    fn from(backend: Backend) -> Self {
        Self::from_bits(1 << backend as u32).unwrap()
    }
}

/// Options for requesting adapter.
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RequestAdapterOptions<S> {
    /// Power preference for the adapter.
    pub power_preference: PowerPreference,
    /// Surface that is required to be presentable with the requested adapter. This does not
    /// create the surface, only guarantees that the adapter can present to said surface.
    pub compatible_surface: Option<S>,
}

impl<S> Default for RequestAdapterOptions<S> {
    fn default() -> Self {
        Self {
            power_preference: PowerPreference::default(),
            compatible_surface: None,
        }
    }
}

bitflags::bitflags! {
    /// Features that are not guaranteed to be supported.
    ///
    /// These are either part of the webgpu standard, or are extension features supported by
    /// wgpu when targeting native.
    ///
    /// If you want to use a feature, you need to first verify that the adapter supports
    /// the feature. If the adapter does not support the feature, requesting a device with it enabled
    /// will panic.
    #[repr(transparent)]
    #[derive(Default)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
    #[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct Features: u64 {
        /// By default, polygon depth is clipped to 0-1 range. Anything outside of that range
        /// is rejected, and respective fragments are not touched.
        ///
        /// With this extension, we can force clamping of the polygon depth to 0-1. That allows
        /// shadow map occluders to be rendered into a tighter depth range.
        ///
        /// Supported platforms:
        /// - desktops
        /// - some mobile chips
        ///
        /// This is a web and native feature.
        const DEPTH_CLAMPING = 0x0000_0000_0000_0001;
        /// Enables BCn family of compressed textures. All BCn textures use 4x4 pixel blocks
        /// with 8 or 16 bytes per block.
        ///
        /// Compressed textures sacrifice some quality in exchange for signifigantly reduced
        /// bandwidth usage.
        ///
        /// Supported Platforms:
        /// - desktops
        ///
        /// This is a web and native feature.
        const TEXTURE_COMPRESSION_BC = 0x0000_0000_0000_0002;
        /// Webgpu only allows the MAP_READ and MAP_WRITE buffer usage to be matched with
        /// COPY_DST and COPY_SRC respectively. This removes this requirement.
        ///
        /// This is only beneficial on systems that share memory between CPU and GPU. If enabled
        /// on a system that doesn't, this can severely hinder performance. Only use if you understand
        /// the consequences.
        ///
        /// Supported platforms:
        /// - All
        ///
        /// This is a native only feature.
        const MAPPABLE_PRIMARY_BUFFERS = 0x0000_0000_0001_0000;
        /// Allows the user to create uniform arrays of sampled textures in shaders:
        ///
        /// eg. `uniform texture2D textures[10]`.
        ///
        /// This capability allows them to exist and to be indexed by compile time constant
        /// values.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan
        ///
        /// This is a native only feature.
        const SAMPLED_TEXTURE_BINDING_ARRAY = 0x0000_0000_0002_0000;
        /// Allows shaders to index sampled texture arrays with dynamically uniform values:
        ///
        /// eg. `texture_array[uniform_value]`
        ///
        /// This capability means the hardware will also support SAMPLED_TEXTURE_BINDING_ARRAY.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan's shaderSampledImageArrayDynamicIndexing feature
        ///
        /// This is a native only feature.
        const SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING = 0x0000_0000_0004_0000;
        /// Allows shaders to index sampled texture arrays with dynamically non-uniform values:
        ///
        /// eg. `texture_array[vertex_data]`
        ///
        /// In order to use this capability, the corresponding GLSL extension must be enabled like so:
        ///
        /// `#extension GL_EXT_nonuniform_qualifier : require`
        ///
        /// and then used either as `nonuniformEXT` qualifier in variable declaration:
        ///
        /// eg. `layout(location = 0) nonuniformEXT flat in int vertex_data;`
        ///
        /// or as `nonuniformEXT` constructor:
        ///
        /// eg. `texture_array[nonuniformEXT(vertex_data)]`
        ///
        /// HLSL does not need any extension.
        ///
        /// This capability means the hardware will also support SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING
        /// and SAMPLED_TEXTURE_BINDING_ARRAY.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan 1.2+ (or VK_EXT_descriptor_indexing)'s shaderSampledImageArrayNonUniformIndexing feature)
        ///
        /// This is a native only feature.
        const SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING = 0x0000_0000_0008_0000;
        /// Allows the user to create unsized uniform arrays of bindings:
        ///
        /// eg. `uniform texture2D textures[]`.
        ///
        /// If this capability is supported, SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING is very likely
        /// to also be supported
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan 1.2+ (or VK_EXT_descriptor_indexing)'s runtimeDescriptorArray feature
        ///
        /// This is a native only feature.
        const UNSIZED_BINDING_ARRAY = 0x0000_0000_0010_0000;
        /// Allows the user to call [`RenderPass::multi_draw_indirect`] and [`RenderPass::multi_draw_indexed_indirect`].
        ///
        /// Allows multiple indirect calls to be dispatched from a single buffer.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Metal
        /// - Vulkan
        ///
        /// This is a native only feature.
        const MULTI_DRAW_INDIRECT = 0x0000_0000_0020_0000;
        /// Allows the user to call [`RenderPass::multi_draw_indirect_count`] and [`RenderPass::multi_draw_indexed_indirect_count`].
        ///
        /// This allows the use of a buffer containing the actual number of draw calls.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan 1.2+ (or VK_KHR_draw_indirect_count)
        ///
        /// This is a native only feature.
        const MULTI_DRAW_INDIRECT_COUNT = 0x0000_0000_0040_0000;
        /// Allows the use of push constants: small, fast bits of memory that can be updated
        /// inside a [`RenderPass`].
        ///
        /// Allows the user to call [`RenderPass::set_push_constants`], provide a non-empty array
        /// to [`PipelineLayoutDescriptor`], and provide a non-zero limit to [`Limits::max_push_constant_size`].
        ///
        /// A block of push constants can be declared with `layout(push_constant) uniform Name {..}` in shaders.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        /// - Metal
        /// - DX11 (emulated with uniforms)
        /// - OpenGL (emulated with uniforms)
        ///
        /// This is a native only feature.
        const PUSH_CONSTANTS = 0x0000_0000_0080_0000;
        /// Allows the use of [`AddressMode::ClampToBorder`].
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        /// - Metal (macOS 10.12+ only)
        /// - DX11
        /// - OpenGL
        ///
        /// This is a web and native feature.
        const ADDRESS_MODE_CLAMP_TO_BORDER = 0x0000_0000_0100_0000;
        /// Allows the user to set a non-fill polygon mode in [`RasterizationStateDescriptor::polygon_mode`]
        ///
        /// This allows drawing polygons/triangles as lines (wireframe) or points instead of filled
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        ///
        /// This is a native only feature.
        const NON_FILL_POLYGON_MODE = 0x0000_0000_0200_0000;
        /// Features which are part of the upstream WebGPU standard.
        const ALL_WEBGPU = 0x0000_0000_0000_FFFF;
        /// Features that are only available when targeting native (not web).
        const ALL_NATIVE = 0xFFFF_FFFF_FFFF_0000;
    }
}

/// Represents the sets of limits an adapter/device supports.
///
/// Limits "better" than the default must be supported by the adapter and requested when requesting
/// a device. If limits "better" than the adapter supports are requested, requesting a device will panic.
/// Once a device is requested, you may only use resources up to the limits requested _even_ if the
/// adapter supports "better" limits.
///
/// Requesting limits that are "better" than you need may cause performance to decrease because the
/// implementation needs to support more than is needed. You should ideally only request exactly what
/// you need.
///
/// See also: https://gpuweb.github.io/gpuweb/#dictdef-gpulimits
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct Limits {
    /// Amount of bind groups that can be attached to a pipeline at the same time. Defaults to 4. Higher is "better".
    pub max_bind_groups: u32,
    /// Amount of uniform buffer bindings that can be dynamic in a single pipeline. Defaults to 8. Higher is "better".
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    /// Amount of storage buffer bindings that can be dynamic in a single pipeline. Defaults to 4. Higher is "better".
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    /// Amount of sampled textures visible in a single shader stage. Defaults to 16. Higher is "better".
    pub max_sampled_textures_per_shader_stage: u32,
    /// Amount of samplers visible in a single shader stage. Defaults to 16. Higher is "better".
    pub max_samplers_per_shader_stage: u32,
    /// Amount of storage buffers visible in a single shader stage. Defaults to 4. Higher is "better".
    pub max_storage_buffers_per_shader_stage: u32,
    /// Amount of storage textures visible in a single shader stage. Defaults to 4. Higher is "better".
    pub max_storage_textures_per_shader_stage: u32,
    /// Amount of uniform buffers visible in a single shader stage. Defaults to 12. Higher is "better".
    pub max_uniform_buffers_per_shader_stage: u32,
    /// Maximum size in bytes of a binding to a uniform buffer. Defaults to 16384. Higher is "better".
    pub max_uniform_buffer_binding_size: u32,
    /// Amount of storage available for push constants in bytes. Defaults to 0. Higher is "better".
    /// Requesting more than 0 during device creation requires [`Features::PUSH_CONSTANTS`] to be enabled.
    ///
    /// Expect the size to be:
    /// - Vulkan: 128-256 bytes
    /// - DX12: 256 bytes
    /// - Metal: 4096 bytes
    /// - DX11 & OpenGL don't natively support push constants, and are emulated with uniforms,
    ///   so this number is less useful.
    pub max_push_constant_size: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bind_groups: 4,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 4,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 4,
            max_storage_textures_per_shader_stage: 4,
            max_uniform_buffers_per_shader_stage: 12,
            max_uniform_buffer_binding_size: 16384,
            max_push_constant_size: 0,
        }
    }
}

/// Describes a [`Device`].
#[repr(C)]
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct DeviceDescriptor<L> {
    /// Debug label for the device.
    pub label: L,
    /// Features that the device should support. If any feature is not supported by
    /// the adapter, creating a device will panic.
    pub features: Features,
    /// Limits that the device should support. If any limit is "better" than the limit exposed by
    /// the adapter, creating a device will panic.
    pub limits: Limits,
    /// Switch shader validation on/off. This is a temporary field
    /// that will be removed once our validation logic is complete.
    pub shader_validation: bool,
}

impl<L> DeviceDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> DeviceDescriptor<K> {
        DeviceDescriptor {
            label: fun(&self.label),
            features: self.features,
            limits: self.limits.clone(),
            shader_validation: self.shader_validation,
        }
    }
}

bitflags::bitflags! {
    /// Describes the shader stages that a binding will be visible from.
    ///
    /// These can be combined so something that is visible from both vertex and fragment shaders can be defined as:
    ///
    /// `ShaderStage::VERTEX | ShaderStage::FRAGMENT`
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
    pub struct ShaderStage: u32 {
        /// Binding is not visible from any shader stage.
        const NONE = 0;
        /// Binding is visible from the vertex shader of a render pipeline.
        const VERTEX = 1;
        /// Binding is visible from the fragment shader of a render pipeline.
        const FRAGMENT = 2;
        /// Binding is visible from the compute shader of a compute pipeline.
        const COMPUTE = 4;
    }
}

/// Dimensions of a particular texture view.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureViewDimension {
    /// A one dimensional texture. `texture1D` in glsl shaders.
    D1,
    /// A two dimensional texture. `texture2D` in glsl shaders.
    D2,
    /// A two dimensional array texture. `texture2DArray` in glsl shaders.
    D2Array,
    /// A cubemap texture. `textureCube` in glsl shaders.
    Cube,
    /// A cubemap array texture. `textureCubeArray` in glsl shaders.
    CubeArray,
    /// A three dimensional texture. `texture3D` in glsl shaders.
    D3,
}

impl Default for TextureViewDimension {
    fn default() -> Self {
        Self::D2
    }
}

impl TextureViewDimension {
    /// Get the texture dimension required fo this texture view dimension.
    pub fn compatible_texture_dimension(self) -> TextureDimension {
        match self {
            Self::D1 => TextureDimension::D1,
            Self::D2 | Self::D2Array | Self::Cube | Self::CubeArray => TextureDimension::D2,
            Self::D3 => TextureDimension::D3,
        }
    }
}

/// Alpha blend factor.
///
/// Alpha blending is very complicated: see the OpenGL or Vulkan spec for more information.
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

/// Alpha blend operation.
///
/// Alpha blending is very complicated: see the OpenGL or Vulkan spec for more information.
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
        Self::Add
    }
}

/// Describes the blend state of a pipeline.
///
/// Alpha blending is very complicated: see the OpenGL or Vulkan spec for more information.
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
        Self::REPLACE
    }
}

/// Describes the color state of a render pipeline.
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ColorStateDescriptor {
    /// The [`TextureFormat`] of the image that this pipeline will render to. Must match the the format
    /// of the corresponding color attachment in [`CommandEncoder::begin_render_pass`].
    pub format: TextureFormat,
    /// The alpha blending that is used for this pipeline.
    pub alpha_blend: BlendDescriptor,
    /// The color blending that is used for this pipeline.
    pub color_blend: BlendDescriptor,
    /// Mask which enables/disables writes to different color/alpha channel.
    pub write_mask: ColorWrite,
}

impl From<TextureFormat> for ColorStateDescriptor {
    fn from(format: TextureFormat) -> Self {
        Self {
            format,
            alpha_blend: BlendDescriptor::REPLACE,
            color_blend: BlendDescriptor::REPLACE,
            write_mask: ColorWrite::ALL,
        }
    }
}

/// Primitive type the input mesh is composed of.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum PrimitiveTopology {
    /// Vertex data is a list of points. Each vertex is a new point.
    PointList = 0,
    /// Vertex data is a list of lines. Each pair of vertices composes a new line.
    ///
    /// Vertices `0 1 2 3` create two lines `0 1` and `2 3`
    LineList = 1,
    /// Vertex data is a strip of lines. Each set of two adjacent vertices form a line.
    ///
    /// Vertices `0 1 2 3` create three lines `0 1`, `1 2`, and `2 3`.
    LineStrip = 2,
    /// Vertex data is a list of triangles. Each set of 3 vertices composes a new triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create two triangles `0 1 2` and `3 4 5`
    TriangleList = 3,
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` creates four triangles `0 1 2`, `2 1 3`, `3 2 4`, and `4 3 5`
    TriangleStrip = 4,
}

/// Winding order which classifies the "front" face.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum FrontFace {
    /// Triangles with vertices in counter clockwise order are considered the front face.
    ///
    /// This is the default with right handed coordinate spaces.
    Ccw = 0,
    /// Triangles with vertices in clockwise order are considered the front face.
    ///
    /// This is the default with left handed coordinate spaces.
    Cw = 1,
}

impl Default for FrontFace {
    fn default() -> Self {
        Self::Ccw
    }
}

/// Type of faces to be culled.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum CullMode {
    /// No faces should be culled
    None = 0,
    /// Front faces should be culled
    Front = 1,
    /// Back faces should be culled
    Back = 2,
}

impl Default for CullMode {
    fn default() -> Self {
        Self::None
    }
}

/// Type of drawing mode for polygons
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum PolygonMode {
    /// Polygons are filled
    Fill = 0,
    /// Polygons are draw as line segments
    Line = 1,
    /// Polygons are draw as points
    Point = 2,
}

impl Default for PolygonMode {
    fn default() -> Self {
        Self::Fill
    }
}

/// Describes the state of the rasterizer in a render pipeline.
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RasterizationStateDescriptor {
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    /// Controls the way each polygon is rasterized. Can be either `Fill` (default), `Line` or `Point`
    ///
    /// Setting this to something other than `Fill` requires `Features::NON_FILL_POLYGON_MODE` to be enabled.
    pub polygon_mode: PolygonMode,
    /// If enabled polygon depth is clamped to 0-1 range instead of being clipped.
    ///
    /// Requires `Features::DEPTH_CLAMPING` enabled.
    pub clamp_depth: bool,
    pub depth_bias: i32,
    pub depth_bias_slope_scale: f32,
    pub depth_bias_clamp: f32,
}

/// Underlying texture data format.
///
/// If there is a conversion in the format (such as srgb -> linear), The conversion listed is for
/// loading from texture in a shader. When writing to the texture, the opposite conversion takes place.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum TextureFormat {
    // Normal 8 bit formats
    /// Red channel only. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    R8Unorm = 0,
    /// Red channel only. 8 bit integer per channel. [-127, 127] converted to/from float [-1, 1] in shader.
    R8Snorm = 1,
    /// Red channel only. 8 bit integer per channel. Unsigned in shader.
    R8Uint = 2,
    /// Red channel only. 8 bit integer per channel. Signed in shader.
    R8Sint = 3,

    // Normal 16 bit formats
    /// Red channel only. 16 bit integer per channel. Unsigned in shader.
    R16Uint = 4,
    /// Red channel only. 16 bit integer per channel. Signed in shader.
    R16Sint = 5,
    /// Red channel only. 16 bit float per channel. Float in shader.
    R16Float = 6,
    /// Red and green channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    Rg8Unorm = 7,
    /// Red and green channels. 8 bit integer per channel. [-127, 127] converted to/from float [-1, 1] in shader.
    Rg8Snorm = 8,
    /// Red and green channels. 8 bit integer per channel. Unsigned in shader.
    Rg8Uint = 9,
    /// Red and green channel s. 8 bit integer per channel. Signed in shader.
    Rg8Sint = 10,

    // Normal 32 bit formats
    /// Red channel only. 32 bit integer per channel. Unsigned in shader.
    R32Uint = 11,
    /// Red channel only. 32 bit integer per channel. Signed in shader.
    R32Sint = 12,
    /// Red channel only. 32 bit float per channel. Float in shader.
    R32Float = 13,
    /// Red and green channels. 16 bit integer per channel. Unsigned in shader.
    Rg16Uint = 14,
    /// Red and green channels. 16 bit integer per channel. Signed in shader.
    Rg16Sint = 15,
    /// Red and green channels. 16 bit float per channel. Float in shader.
    Rg16Float = 16,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    Rgba8Unorm = 17,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    Rgba8UnormSrgb = 18,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. [-127, 127] converted to/from float [-1, 1] in shader.
    Rgba8Snorm = 19,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Unsigned in shader.
    Rgba8Uint = 20,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Signed in shader.
    Rgba8Sint = 21,
    /// Blue, green, red, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    Bgra8Unorm = 22,
    /// Blue, green, red, and alpha channels. 8 bit integer per channel. Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    Bgra8UnormSrgb = 23,

    // Packed 32 bit formats
    /// Red, green, blue, and alpha channels. 10 bit integer for RGB channels, 2 bit integer for alpha channel. [0, 1023] ([0, 3] for alpha) converted to/from float [0, 1] in shader.
    Rgb10a2Unorm = 24,
    /// Red, green, and blue channels. 11 bit float with no sign bit for RG channels. 10 bit float with no sign bit for blue channel. Float in shader.
    Rg11b10Float = 25,

    // Normal 64 bit formats
    /// Red and green channels. 32 bit integer per channel. Unsigned in shader.
    Rg32Uint = 26,
    /// Red and green channels. 32 bit integer per channel. Signed in shader.
    Rg32Sint = 27,
    /// Red and green channels. 32 bit float per channel. Float in shader.
    Rg32Float = 28,
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. Unsigned in shader.
    Rgba16Uint = 29,
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. Signed in shader.
    Rgba16Sint = 30,
    /// Red, green, blue, and alpha channels. 16 bit float per channel. Float in shader.
    Rgba16Float = 31,

    // Normal 128 bit formats
    /// Red, green, blue, and alpha channels. 32 bit integer per channel. Unsigned in shader.
    Rgba32Uint = 32,
    /// Red, green, blue, and alpha channels. 32 bit integer per channel. Signed in shader.
    Rgba32Sint = 33,
    /// Red, green, blue, and alpha channels. 32 bit float per channel. Float in shader.
    Rgba32Float = 34,

    // Depth and stencil formats
    /// Special depth format with 32 bit floating point depth.
    Depth32Float = 35,
    /// Special depth format with at least 24 bit integer depth.
    Depth24Plus = 36,
    /// Special depth/stencil format with at least 24 bit integer depth and 8 bits integer stencil.
    Depth24PlusStencil8 = 37,

    // Compressed textures usable with `TEXTURE_COMPRESSION_BC` feature.
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 4 color + alpha pallet. 5 bit R + 6 bit G + 5 bit B + 1 bit alpha.
    /// [0, 64] ([0, 1] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc1RgbaUnorm = 38,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 4 color + alpha pallet. 5 bit R + 6 bit G + 5 bit B + 1 bit alpha.
    /// Srgb-color [0, 64] ([0, 16] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc1RgbaUnormSrgb = 39,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet. 5 bit R + 6 bit G + 5 bit B + 4 bit alpha.
    /// [0, 64] ([0, 16] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT3.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc2RgbaUnorm = 40,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet. 5 bit R + 6 bit G + 5 bit B + 4 bit alpha.
    /// Srgb-color [0, 64] ([0, 256] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT3.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc2RgbaUnormSrgb = 41,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet + 8 alpha pallet. 5 bit R + 6 bit G + 5 bit B + 8 bit alpha.
    /// [0, 64] ([0, 256] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT5.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc3RgbaUnorm = 42,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet + 8 alpha pallet. 5 bit R + 6 bit G + 5 bit B + 8 bit alpha.
    /// Srgb-color [0, 64] ([0, 256] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT5.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc3RgbaUnormSrgb = 43,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 8 color pallet. 8 bit R.
    /// [0, 256] converted to/from float [0, 1] in shader.
    ///
    /// Also known as RGTC1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc4RUnorm = 44,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 8 color pallet. 8 bit R.
    /// [-127, 127] converted to/from float [-1, 1] in shader.
    ///
    /// Also known as RGTC1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc4RSnorm = 45,
    /// 4x4 block compressed texture. 16 bytes per block (16 bit/px). 8 color red pallet + 8 color green pallet. 8 bit RG.
    /// [0, 256] converted to/from float [0, 1] in shader.
    ///
    /// Also known as RGTC2.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc5RgUnorm = 46,
    /// 4x4 block compressed texture. 16 bytes per block (16 bit/px). 8 color red pallet + 8 color green pallet. 8 bit RG.
    /// [-127, 127] converted to/from float [-1, 1] in shader.
    ///
    /// Also known as RGTC2.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc5RgSnorm = 47,
    /// 4x4 block compressed texture. 16 bytes per block (16 bit/px). Variable sized pallet. 16 bit unsigned float RGB. Float in shader.
    ///
    /// Also known as BPTC (float).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc6hRgbUfloat = 48,
    /// 4x4 block compressed texture. 16 bytes per block (16 bit/px). Variable sized pallet. 16 bit signed float RGB. Float in shader.
    ///
    /// Also known as BPTC (float).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc6hRgbSfloat = 49,
    /// 4x4 block compressed texture. 16 bytes per block (16 bit/px). Variable sized pallet. 8 bit integer RGBA.
    /// [0, 256] converted to/from float [0, 1] in shader.
    ///
    /// Also known as BPTC (unorm).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc7RgbaUnorm = 50,
    /// 4x4 block compressed texture. 16 bytes per block (16 bit/px). Variable sized pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as BPTC (unorm).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    Bc7RgbaUnormSrgb = 51,
}

bitflags::bitflags! {
    /// Color write mask. Disabled color channels will not be written to.
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
    #[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct ColorWrite: u32 {
        /// Enable red channel writes
        const RED = 1;
        /// Enable green channel writes
        const GREEN = 2;
        /// Enable blue channel writes
        const BLUE = 4;
        /// Enable alpha channel writes
        const ALPHA = 8;
        /// Enable red, green, and blue channel writes
        const COLOR = 7;
        /// Enable writes to all channels.
        const ALL = 15;
    }
}

impl Default for ColorWrite {
    fn default() -> Self {
        Self::ALL
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct StencilStateDescriptor {
    /// Front face mode.
    pub front: StencilStateFaceDescriptor,
    /// Back face mode.
    pub back: StencilStateFaceDescriptor,
    /// Stencil values are AND'd with this mask when reading and writing from the stencil buffer. Only low 8 bits are used.
    pub read_mask: u32,
    /// Stencil values are AND'd with this mask when writing to the stencil buffer. Only low 8 bits are used.
    pub write_mask: u32,
}

impl StencilStateDescriptor {
    pub fn is_enabled(&self) -> bool {
        (self.front != StencilStateFaceDescriptor::IGNORE
            || self.back != StencilStateFaceDescriptor::IGNORE)
            && (self.read_mask != 0 || self.write_mask != 0)
    }
    pub fn is_read_only(&self) -> bool {
        self.write_mask == 0
    }
    pub fn needs_ref_value(&self) -> bool {
        self.front.compare.needs_ref_value() || self.back.compare.needs_ref_value()
    }
}

/// Describes the depth/stencil state in a render pipeline.
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct DepthStencilStateDescriptor {
    /// Format of the depth/stencil buffer, must be special depth format. Must match the the format
    /// of the depth/stencil attachment in [`CommandEncoder::begin_render_pass`].
    pub format: TextureFormat,
    /// If disabled, depth will not be written to.
    pub depth_write_enabled: bool,
    /// Comparison function used to compare depth values in the depth test.
    pub depth_compare: CompareFunction,
    pub stencil: StencilStateDescriptor,
}

impl DepthStencilStateDescriptor {
    pub fn is_depth_enabled(&self) -> bool {
        self.depth_compare != CompareFunction::Always || self.depth_write_enabled
    }
    pub fn is_read_only(&self) -> bool {
        !self.depth_write_enabled && self.stencil.is_read_only()
    }
}

/// Format of indices used with pipeline.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum IndexFormat {
    /// Indices are 16 bit unsigned integers.
    Uint16 = 0,
    /// Indices are 32 bit unsigned integers.
    Uint32 = 1,
}

impl Default for IndexFormat {
    fn default() -> Self {
        Self::Uint32
    }
}

/// Operation to perform on the stencil value.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum StencilOperation {
    /// Keep stencil value unchanged.
    Keep = 0,
    /// Set stencil value to zero.
    Zero = 1,
    /// Replace stencil value with value provided in most recent call to [`RenderPass::set_stencil_reference`].
    Replace = 2,
    /// Bitwise inverts stencil value.
    Invert = 3,
    /// Increments stencil value by one, clamping on overflow.
    IncrementClamp = 4,
    /// Decrements stencil value by one, clamping on underflow.
    DecrementClamp = 5,
    /// Increments stencil value by one, wrapping on overflow.
    IncrementWrap = 6,
    /// Decrements stencil value by one, wrapping on underflow.
    DecrementWrap = 7,
}

impl Default for StencilOperation {
    fn default() -> Self {
        Self::Keep
    }
}

/// Describes stencil state in a render pipeline.
///
/// If you are not using stencil state, set this to [`StencilStateFaceDescriptor::IGNORE`].
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct StencilStateFaceDescriptor {
    /// Comparison function that determines if the fail_op or pass_op is used on the stencil buffer.
    pub compare: CompareFunction,
    /// Operation that is preformed when stencil test fails.
    pub fail_op: StencilOperation,
    /// Operation that is performed when depth test fails but stencil test succeeds.
    pub depth_fail_op: StencilOperation,
    /// Operation that is performed when stencil test success.
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
        Self::IGNORE
    }
}

/// Comparison function used for depth and stencil operations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum CompareFunction {
    /// Function never passes
    Never = 1,
    /// Function passes if new value less than existing value
    Less = 2,
    /// Function passes if new value is equal to existing value
    Equal = 3,
    /// Function passes if new value is less than or equal to existing value
    LessEqual = 4,
    /// Function passes if new value is greater than existing value
    Greater = 5,
    /// Function passes if new value is not equal to existing value
    NotEqual = 6,
    /// Function passes if new value is greater than or equal to existing value
    GreaterEqual = 7,
    /// Function always passes
    Always = 8,
}

impl CompareFunction {
    pub fn needs_ref_value(self) -> bool {
        match self {
            Self::Never | Self::Always => false,
            _ => true,
        }
    }
}

/// Rate that determines when vertex data is advanced.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum InputStepMode {
    /// Input data is advanced every vertex. This is the standard value for vertex data.
    Vertex = 0,
    /// Input data is advanced every instance.
    Instance = 1,
}

/// Vertex inputs (attributes) to shaders.
///
/// Arrays of these can be made with the [`vertex_attr_array`] macro. Vertex attributes are assumed to be tightly packed.
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct VertexAttributeDescriptor {
    /// Byte offset of the start of the input
    pub offset: BufferAddress,
    /// Format of the input
    pub format: VertexFormat,
    /// Location for this input. Must match the location in the shader.
    pub shader_location: ShaderLocation,
}

/// Vertex Format for a Vertex Attribute (input).
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum VertexFormat {
    /// Two unsigned bytes (u8). `uvec2` in shaders.
    Uchar2 = 0,
    /// Four unsigned bytes (u8). `uvec4` in shaders.
    Uchar4 = 1,
    /// Two signed bytes (i8). `ivec2` in shaders.
    Char2 = 2,
    /// Four signed bytes (i8). `ivec4` in shaders.
    Char4 = 3,
    /// Two unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec2` in shaders.
    Uchar2Norm = 4,
    /// Four unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec4` in shaders.
    Uchar4Norm = 5,
    /// Two signed bytes (i8). [-127, 127] converted to float [-1, 1] `vec2` in shaders.
    Char2Norm = 6,
    /// Four signed bytes (i8). [-127, 127] converted to float [-1, 1] `vec4` in shaders.
    Char4Norm = 7,
    /// Two unsigned shorts (u16). `uvec2` in shaders.
    Ushort2 = 8,
    /// Four unsigned shorts (u16). `uvec4` in shaders.
    Ushort4 = 9,
    /// Two signed shorts (i16). `ivec2` in shaders.
    Short2 = 10,
    /// Four signed shorts (i16). `ivec4` in shaders.
    Short4 = 11,
    /// Two unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec2` in shaders.
    Ushort2Norm = 12,
    /// Four unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec4` in shaders.
    Ushort4Norm = 13,
    /// Two signed shorts (i16). [-32767, 32767] converted to float [-1, 1] `vec2` in shaders.
    Short2Norm = 14,
    /// Four signed shorts (i16). [-32767, 32767] converted to float [-1, 1] `vec4` in shaders.
    Short4Norm = 15,
    /// Two half-precision floats (no Rust equiv). `vec2` in shaders.
    Half2 = 16,
    /// Four half-precision floats (no Rust equiv). `vec4` in shaders.
    Half4 = 17,
    /// One single-precision float (f32). `float` in shaders.
    Float = 18,
    /// Two single-precision floats (f32). `vec2` in shaders.
    Float2 = 19,
    /// Three single-precision floats (f32). `vec3` in shaders.
    Float3 = 20,
    /// Four single-precision floats (f32). `vec4` in shaders.
    Float4 = 21,
    /// One unsigned int (u32). `uint` in shaders.
    Uint = 22,
    /// Two unsigned ints (u32). `uvec2` in shaders.
    Uint2 = 23,
    /// Three unsigned ints (u32). `uvec3` in shaders.
    Uint3 = 24,
    /// Four unsigned ints (u32). `uvec4` in shaders.
    Uint4 = 25,
    /// One signed int (i32). `int` in shaders.
    Int = 26,
    /// Two signed ints (i32). `ivec2` in shaders.
    Int2 = 27,
    /// Three signed ints (i32). `ivec3` in shaders.
    Int3 = 28,
    /// Four signed ints (i32). `ivec4` in shaders.
    Int4 = 29,
}

impl VertexFormat {
    pub const fn size(&self) -> u64 {
        match self {
            Self::Uchar2 | Self::Char2 | Self::Uchar2Norm | Self::Char2Norm => 2,
            Self::Uchar4
            | Self::Char4
            | Self::Uchar4Norm
            | Self::Char4Norm
            | Self::Ushort2
            | Self::Short2
            | Self::Ushort2Norm
            | Self::Short2Norm
            | Self::Half2
            | Self::Float
            | Self::Uint
            | Self::Int => 4,
            Self::Ushort4
            | Self::Short4
            | Self::Ushort4Norm
            | Self::Short4Norm
            | Self::Half4
            | Self::Float2
            | Self::Uint2
            | Self::Int2 => 8,
            Self::Float3 | Self::Uint3 | Self::Int3 => 12,
            Self::Float4 | Self::Uint4 | Self::Int4 => 16,
        }
    }
}

bitflags::bitflags! {
    /// Different ways that you can use a buffer.
    ///
    /// The usages determine what kind of memory the buffer is allocated from and what
    /// actions the buffer can partake in.
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
    #[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct BufferUsage: u32 {
        /// Allow a buffer to be mapped for reading using [`Buffer::map_async`] + [`Buffer::get_mapped_range`].
        /// This does not include creating a buffer with [`BufferDescriptor::mapped_at_creation`] set.
        ///
        /// If [`Features::MAPPABLE_PRIMARY_BUFFERS`] isn't enabled, the only other usage a buffer
        /// may have is COPY_DST.
        const MAP_READ = 1;
        /// Allow a buffer to be mapped for writing using [`Buffer::map_async`] + [`Buffer::get_mapped_range_mut`].
        /// This does not include creating a buffer with `mapped_at_creation` set.
        ///
        /// If [`Features::MAPPABLE_PRIMARY_BUFFERS`] feature isn't enabled, the only other usage a buffer
        /// may have is COPY_SRC.
        const MAP_WRITE = 2;
        /// Allow a buffer to be the source buffer for a [`CommandEncoder::copy_buffer_to_buffer`] or [`CommandEncoder::copy_buffer_to_texture`]
        /// operation.
        const COPY_SRC = 4;
        /// Allow a buffer to be the destination buffer for a [`CommandEncoder::copy_buffer_to_buffer`], [`CommandEncoder::copy_texture_to_buffer`],
        /// or [`Queue::write_buffer`] operation.
        const COPY_DST = 8;
        /// Allow a buffer to be the index buffer in a draw operation.
        const INDEX = 16;
        /// Allow a buffer to be the vertex buffer in a draw operation.
        const VERTEX = 32;
        /// Allow a buffer to be a [`BufferBindingType::Uniform`] inside a bind group.
        const UNIFORM = 64;
        /// Allow a buffer to be a [`BufferBindingType::Storage`] inside a bind group.
        const STORAGE = 128;
        /// Allow a buffer to be the indirect buffer in an indirect draw call.
        const INDIRECT = 256;
    }
}

/// Describes a [`Buffer`].
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BufferDescriptor<L> {
    /// Debug label of a buffer. This will show up in graphics debuggers for easy identification.
    pub label: L,
    /// Size of a buffer.
    pub size: BufferAddress,
    /// Usages of a buffer. If the buffer is used in any way that isn't specified here, the operation
    /// will panic.
    pub usage: BufferUsage,
    /// Allows a buffer to be mapped immediately after they are made. It does not have to be [`BufferUsage::MAP_READ`] or
    /// [`BufferUsage::MAP_WRITE`], all buffers are allowed to be mapped at creation.
    pub mapped_at_creation: bool,
}

impl<L> BufferDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> BufferDescriptor<K> {
        BufferDescriptor {
            label: fun(&self.label),
            size: self.size,
            usage: self.usage,
            mapped_at_creation: self.mapped_at_creation,
        }
    }
}

/// Describes a [`CommandEncoder`].
#[repr(C)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommandEncoderDescriptor<L> {
    /// Debug label for the command encoder. This will show up in graphics debuggers for easy identification.
    pub label: L,
}

impl<L> CommandEncoderDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CommandEncoderDescriptor<K> {
        CommandEncoderDescriptor {
            label: fun(&self.label),
        }
    }
}

impl<T> Default for CommandEncoderDescriptor<Option<T>> {
    fn default() -> Self {
        Self { label: None }
    }
}

/// Behavior of the presentation engine based on frame rate.
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
    /// Different ways that you can use a texture.
    ///
    /// The usages determine what kind of memory the texture is allocated from and what
    /// actions the texture can partake in.
    #[repr(transparent)]
    #[cfg_attr(feature = "trace", derive(Serialize))]
    #[cfg_attr(feature = "replay", derive(Deserialize))]
    pub struct TextureUsage: u32 {
        /// Allows a texture to be the source in a [`CommandEncoder::copy_texture_to_buffer`] or
        /// [`CommandEncoder::copy_texture_to_texture`] operation.
        const COPY_SRC = 1;
        /// Allows a texture to be the destination in a  [`CommandEncoder::copy_texture_to_buffer`],
        /// [`CommandEncoder::copy_texture_to_texture`], or [`Queue::write_texture`] operation.
        const COPY_DST = 2;
        /// Allows a texture to be a [`BindingType::Texture`] in a bind group.
        const SAMPLED = 4;
        /// Allows a texture to be a [`BindingType::StorageTexture`] in a bind group.
        const STORAGE = 8;
        /// Allows a texture to be an output attachment of a renderpass.
        const RENDER_ATTACHMENT = 16;
    }
}

/// Describes a [`SwapChain`].
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct SwapChainDescriptor {
    /// The usage of the swap chain. The only supported usage is `RENDER_ATTACHMENT`.
    pub usage: TextureUsage,
    /// The texture format of the swap chain. The only formats that are guaranteed are
    /// `Bgra8Unorm` and `Bgra8UnormSrgb`
    pub format: TextureFormat,
    /// Width of the swap chain. Must be the same size as the surface.
    pub width: u32,
    /// Height of the swap chain. Must be the same size as the surface.
    pub height: u32,
    /// Presentation mode of the swap chain. FIFO is the only guaranteed to be supported, though
    /// other formats will automatically fall back to FIFO.
    pub present_mode: PresentMode,
}

/// Status of the recieved swapchain image.
#[repr(C)]
#[derive(Debug)]
pub enum SwapChainStatus {
    Good,
    Suboptimal,
    Timeout,
    Outdated,
    Lost,
}

/// RGBA double precision color.
///
/// This is not to be used as a generic color type, only for specific wgpu interfaces.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const RED: Self = Self {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const GREEN: Self = Self {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const BLUE: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
}

/// Dimensionality of a texture.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureDimension {
    /// 1D texture
    D1,
    /// 2D texture
    D2,
    /// 3D texture
    D3,
}

/// Origin of a copy to/from a texture.
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
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };
}

impl Default for Origin3d {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Extent of a texture related operation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Default for Extent3d {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            depth: 1,
        }
    }
}

/// Describes a [`Texture`].
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct TextureDescriptor<L> {
    /// Debug label of the texture. This will show up in graphics debuggers for easy identification.
    pub label: L,
    /// Size of the texture. For a regular 1D/2D texture, the unused sizes will be 1. For 2DArray textures, Z is the
    /// number of 2D textures in that array.
    pub size: Extent3d,
    /// Mip count of texture. For a texture with no extra mips, this must be 1.
    pub mip_level_count: u32,
    /// Sample count of texture. If this is not 1, texture must have [`BindingType::Texture::multisampled`] set to true.
    pub sample_count: u32,
    /// Dimensions of the texture.
    pub dimension: TextureDimension,
    /// Format of the texture.
    pub format: TextureFormat,
    /// Allowed usages of the texture. If used in other ways, the operation will panic.
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

/// Kind of data the texture holds.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureAspect {
    /// Depth, Stencil, and Color.
    All,
    /// Stencil.
    StencilOnly,
    /// Depth.
    DepthOnly,
}

impl Default for TextureAspect {
    fn default() -> Self {
        Self::All
    }
}

/// How edges should be handled in texture addressing.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum AddressMode {
    /// Clamp the value to the edge of the texture
    ///
    /// -0.25 -> 0.0
    /// 1.25  -> 1.0
    ClampToEdge = 0,
    /// Repeat the texture in a tiling fashion
    ///
    /// -0.25 -> 0.75
    /// 1.25 -> 0.25
    Repeat = 1,
    /// Repeat the texture, mirroring it every repeat
    ///
    /// -0.25 -> 0.25
    /// 1.25 -> 0.75
    MirrorRepeat = 2,
    /// Clamp the value to the border of the texture
    /// Requires feature [`Features::ADDRESS_MODE_CLAMP_TO_BORDER`]
    ///
    /// -0.25 -> border
    /// 1.25 -> border
    ClampToBorder = 3,
}

impl Default for AddressMode {
    fn default() -> Self {
        Self::ClampToEdge
    }
}

/// Texel mixing mode when sampling between texels.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum FilterMode {
    /// Nearest neighbor sampling.
    ///
    /// This creates a pixelated effect when used as a mag filter
    Nearest = 0,
    /// Linear Interpolation
    ///
    /// This makes textures smooth but blurry when used as a mag filter.
    Linear = 1,
}

impl Default for FilterMode {
    fn default() -> Self {
        Self::Nearest
    }
}

/// A range of push constant memory to pass to a shader stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PushConstantRange {
    /// Stage push constant range is visible from. Each stage can only be served by at most one range.
    /// One range can serve multiple stages however.
    pub stages: ShaderStage,
    /// Range in push constant memory to use for the stage. Must be less than [`Limits::max_push_constant_size`].
    /// Start and end must be aligned to the 4s.
    pub range: Range<u32>,
}

/// Describes a [`CommandBuffer`].
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct CommandBufferDescriptor<L> {
    pub label: L,
}

impl<L> CommandBufferDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CommandBufferDescriptor<K> {
        CommandBufferDescriptor {
            label: fun(&self.label),
        }
    }
}

/// Describes a [`RenderBundle`].
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderBundleDescriptor<L> {
    /// Debug label of the render bundle encoder. This will show up in graphics debuggers for easy identification.
    pub label: L,
}

impl<L> RenderBundleDescriptor<L> {
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> RenderBundleDescriptor<K> {
        RenderBundleDescriptor {
            label: fun(&self.label),
        }
    }
}

impl<T> Default for RenderBundleDescriptor<Option<T>> {
    fn default() -> Self {
        Self { label: None }
    }
}

/// Layout of a texture in a buffer's memory.
#[repr(C)]
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct TextureDataLayout {
    /// Offset into the buffer that is the start of the texture. Must be a multiple of texture block size.
    /// For non-compressed textures, this is 1.
    pub offset: BufferAddress,
    /// Bytes per "row" of the image. This represents one row of pixels in the x direction. Compressed
    /// textures include multiple rows of pixels in each "row". May be 0 for 1D texture copies.
    ///
    /// Must be a multiple of 256 for [`CommandEncoder::copy_buffer_to_texture`] and [`CommandEncoder::copy_texture_to_buffer`].
    /// [`Queue::write_texture`] does not have this requirement.
    ///
    /// Must be a multiple of the texture block size. For non-compressed textures, this is 1.
    pub bytes_per_row: u32,
    /// Rows that make up a single "image". Each "image" is one layer in the z direction of a 3D image. May be larger
    /// than `copy_size.y`.
    ///
    /// May be 0 for 2D texture copies.
    pub rows_per_image: u32,
}

/// Specific type of a buffer binding.
///
/// WebGPU spec: https://gpuweb.github.io/gpuweb/#enumdef-gpubufferbindingtype
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BufferBindingType {
    /// A buffer for uniform values.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(std140, binding = 0)
    /// uniform Globals {
    ///     vec2 aUniform;
    ///     vec2 anotherUniform;
    /// };
    /// ```
    Uniform,
    /// A storage buffer.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout (set=0, binding=0) buffer myStorageBuffer {
    ///     vec4 myElement[];
    /// };
    /// ```
    Storage {
        /// If `true`, the buffer can only be read in the shader,
        /// and it must be annotated with `readonly`.
        ///
        /// Example GLSL syntax:
        /// ```cpp,ignore
        /// layout (set=0, binding=0) readonly buffer myStorageBuffer {
        ///     vec4 myElement[];
        /// };
        /// ```
        read_only: bool,
    },
}

impl Default for BufferBindingType {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Specific type of a sample in a texture binding.
///
/// WebGPU spec: https://gpuweb.github.io/gpuweb/#enumdef-gputexturesampletype
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureSampleType {
    /// Sampling returns floats.
    ///
    /// If `filterable` is false, the texture can't be sampled with
    /// a filtering sampler.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    Float { filterable: bool },
    /// Sampling does the depth reference comparison.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2DShadow t;
    /// ```
    Depth,
    /// Sampling returns signed integers.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform itexture2D t;
    /// ```
    Sint,
    /// Sampling returns unsigned integers.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform utexture2D t;
    /// ```
    Uint,
}

impl Default for TextureSampleType {
    fn default() -> Self {
        Self::Float { filterable: true }
    }
}

impl From<TextureFormat> for TextureSampleType {
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

            TextureFormat::R32Float | TextureFormat::Rg32Float | TextureFormat::Rgba32Float => {
                Self::Float { filterable: false }
            }

            TextureFormat::R8Unorm
            | TextureFormat::R8Snorm
            | TextureFormat::R16Float
            | TextureFormat::Rg8Unorm
            | TextureFormat::Rg8Snorm
            | TextureFormat::Rg16Float
            | TextureFormat::Rg11b10Float
            | TextureFormat::Rgba8Snorm
            | TextureFormat::Rgba16Float
            | TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgb10a2Unorm
            | TextureFormat::Bc1RgbaUnorm
            | TextureFormat::Bc1RgbaUnormSrgb
            | TextureFormat::Bc2RgbaUnorm
            | TextureFormat::Bc2RgbaUnormSrgb
            | TextureFormat::Bc3RgbaUnorm
            | TextureFormat::Bc3RgbaUnormSrgb
            | TextureFormat::Bc4RUnorm
            | TextureFormat::Bc4RSnorm
            | TextureFormat::Bc5RgUnorm
            | TextureFormat::Bc5RgSnorm
            | TextureFormat::Bc6hRgbSfloat
            | TextureFormat::Bc6hRgbUfloat
            | TextureFormat::Bc7RgbaUnorm
            | TextureFormat::Bc7RgbaUnormSrgb => Self::Float { filterable: true },

            TextureFormat::Depth32Float
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8 => Self::Depth,
        }
    }
}

/// Specific type of a sample in a texture binding.
///
/// WebGPU spec: https://gpuweb.github.io/gpuweb/#enumdef-gpustoragetextureaccess
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum StorageTextureAccess {
    /// The texture can only be read in the shader and it must be annotated with `readonly`.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) readonly uniform image2D myStorageImage;
    /// ```
    ReadOnly,
    /// The texture can only be read in the shader and it must be annotated with `writeonly`.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) writeonly uniform image2D myStorageImage;
    /// ```
    WriteOnly,
}

/// Specific type of a binding.
///
/// WebGPU spec: the enum of
/// - https://gpuweb.github.io/gpuweb/#dictdef-gpubufferbindinglayout
/// - https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerbindinglayout
/// - https://gpuweb.github.io/gpuweb/#dictdef-gputexturebindinglayout
/// - https://gpuweb.github.io/gpuweb/#dictdef-gpustoragetexturebindinglayout
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BindingType {
    /// A buffer binding.
    Buffer {
        ty: BufferBindingType,
        /// Indicates that the binding has a dynamic offset.
        /// One offset must be passed to [`RenderPass::set_bind_group`] for each dynamic binding in increasing order of binding number.
        #[cfg_attr(any(feature = "replay", feature = "trace"), serde(default))]
        has_dynamic_offset: bool,
        /// Minimum size of the corresponding `BufferBinding` required to match this entry.
        /// When pipeline is created, the size has to cover at least the corresponding structure in the shader
        /// plus one element of the unbound array, which can only be last in the structure.
        /// If `None`, the check is performed at draw call time instead of pipeline and bind group creation.
        #[cfg_attr(any(feature = "replay", feature = "trace"), serde(default))]
        min_binding_size: Option<BufferSize>,
    },
    /// A sampler that can be used to sample a texture.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform sampler s;
    /// ```
    Sampler {
        /// The sampling result is produced based on more than a single color sample from a texture,
        /// e.g. when bilinear interpolation is enabled.
        ///
        /// A filtering sampler can only be used with a filterable texture.
        filtering: bool,
        /// Use as a comparison sampler instead of a normal sampler.
        /// For more info take a look at the analogous functionality in OpenGL: https://www.khronos.org/opengl/wiki/Sampler_Object#Comparison_mode.
        comparison: bool,
    },
    /// A texture binding.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    Texture {
        /// Sample type of the texture binding.
        sample_type: TextureSampleType,
        /// Dimension of the texture view that is going to be sampled.
        view_dimension: TextureViewDimension,
        /// True if the texture has a sample count greater than 1. If this is true,
        /// the texture must be read from shaders with `texture1DMS`, `texture2DMS`, or `texture3DMS`,
        /// depending on `dimension`.
        multisampled: bool,
    },
    /// A storage texture.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) uniform image2D myStorageImage;
    /// ```
    /// Note that the texture format must be specified in the shader as well.
    /// A list of valid formats can be found in the specification here: https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.html#layout-qualifiers
    StorageTexture {
        /// Allowed access to this texture.
        access: StorageTextureAccess,
        /// Format of the texture.
        format: TextureFormat,
        /// Dimension of the texture view that is going to be sampled.
        view_dimension: TextureViewDimension,
    },
}

impl BindingType {
    pub fn has_dynamic_offset(&self) -> bool {
        match *self {
            Self::Buffer {
                has_dynamic_offset, ..
            } => has_dynamic_offset,
            _ => false,
        }
    }
}

/// Describes a single binding inside a bind group.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BindGroupLayoutEntry {
    /// Binding index. Must match shader index and be unique inside a BindGroupLayout. A binding
    /// of index 1, would be described as `layout(set = 0, binding = 1) uniform` in shaders.
    pub binding: u32,
    /// Which shader stages can see this binding.
    pub visibility: ShaderStage,
    /// The type of the binding
    pub ty: BindingType,
    /// If this value is Some, indicates this entry is an array. Array size must be 1 or greater.
    ///
    /// If this value is Some and `ty` is `BindingType::Texture`, [`Features::SAMPLED_TEXTURE_BINDING_ARRAY`] must be supported.
    ///
    /// If this value is Some and `ty` is any other variant, bind group creation will fail.
    #[cfg_attr(any(feature = "replay", feature = "trace"), serde(default))]
    pub count: Option<NonZeroU32>,
}

/// View of a buffer which can be used to copy to/from a texture.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct BufferCopyView<B> {
    /// The buffer to be copied to/from.
    pub buffer: B,
    /// The layout of the texture data in this buffer.
    pub layout: TextureDataLayout,
}

/// View of a texture which can be used to copy to/from a buffer/texture.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct TextureCopyView<T> {
    /// The texture to be copied to/from.
    pub texture: T,
    /// The target mip level of the texture.
    pub mip_level: u32,
    /// The base texel of the texture in the selected `mip_level`.
    #[cfg_attr(any(feature = "replay", feature = "trace"), serde(default))]
    pub origin: Origin3d,
}

/// Color variation to use when sampler addressing mode is [`AddressMode::ClampToBorder`]
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum SamplerBorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}
