/*! This library describes the API surface of WebGPU that is agnostic of the backend.
 *  This API is used for targeting both Web and Native.
 */

#![allow(
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
)]
#![warn(missing_docs)]

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
/// [`bytes_per_row`]: ImageDataLayout::bytes_per_row
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
/// An offset into the query resolve buffer has to be aligned to this.
pub const QUERY_RESOLVE_BUFFER_ALIGNMENT: BufferAddress = 256;
/// Buffer to buffer copy as well as buffer clear offsets and sizes must be aligned to this number.
pub const COPY_BUFFER_ALIGNMENT: BufferAddress = 4;
/// Size to align mappings.
pub const MAP_ALIGNMENT: BufferAddress = 8;
/// Vertex buffer strides have to be aligned to this number.
pub const VERTEX_STRIDE_ALIGNMENT: BufferAddress = 4;
/// Alignment all push constants need
pub const PUSH_CONSTANT_ALIGNMENT: u32 = 4;
/// Maximum queries in a query set
pub const QUERY_SET_MAX_QUERIES: u32 = 8192;
/// Size of a single piece of query data.
pub const QUERY_SIZE: u32 = 8;

/// Backends supported by wgpu.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum Backend {
    /// Dummy backend, used for testing.
    Empty = 0,
    /// Vulkan API
    Vulkan = 1,
    /// Metal API (Apple platforms)
    Metal = 2,
    /// Direct3D-12 (Windows)
    Dx12 = 3,
    /// Direct3D-11 (Windows)
    Dx11 = 4,
    /// OpenGL ES-3 (Linux, Android)
    Gl = 5,
    /// WebGPU in the browser
    BrowserWebGpu = 6,
}

/// Power Preference when choosing a physical adapter.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum PowerPreference {
    /// Adapter that uses the least possible power. This is often an integrated GPU.
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
    pub struct Backends: u32 {
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

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(Backends);

impl From<Backend> for Backends {
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
    /// Indicates that only a fallback adapter can be returned. This is generally a "software"
    /// implementation on the system.
    pub force_fallback_adapter: bool,
    /// Surface that is required to be presentable with the requested adapter. This does not
    /// create the surface, only guarantees that the adapter can present to said surface.
    pub compatible_surface: Option<S>,
}

impl<S> Default for RequestAdapterOptions<S> {
    fn default() -> Self {
        Self {
            power_preference: PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }
    }
}

//TODO: make robust resource access configurable

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
    pub struct Features: u64 {
        /// By default, polygon depth is clipped to 0-1 range before/during rasterization.
        /// Anything outside of that range is rejected, and respective fragments are not touched.
        ///
        /// With this extension, we can disabling clipping. That allows
        /// shadow map occluders to be rendered into a tighter depth range.
        ///
        /// Supported platforms:
        /// - desktops
        /// - some mobile chips
        ///
        /// This is a web and native feature.
        const DEPTH_CLIP_CONTROL = 1 << 0;
        /// Enables BCn family of compressed textures. All BCn textures use 4x4 pixel blocks
        /// with 8 or 16 bytes per block.
        ///
        /// Compressed textures sacrifice some quality in exchange for significantly reduced
        /// bandwidth usage.
        ///
        /// Support for this feature guarantees availability of [`TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING`] for BCn formats.
        /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] may enable additional usages.
        ///
        /// Supported Platforms:
        /// - desktops
        ///
        /// This is a web and native feature.
        const TEXTURE_COMPRESSION_BC = 1 << 1;
        /// Allows non-zero value for the "first instance" in indirect draw calls.
        ///
        /// Supported Platforms:
        /// - Vulkan (mostly)
        /// - DX12
        /// - Metal
        ///
        /// This is a web and native feature.
        const INDIRECT_FIRST_INSTANCE = 1 << 2;
        /// Enables use of Timestamp Queries. These queries tell the current gpu timestamp when
        /// all work before the query is finished. Call [`CommandEncoder::write_timestamp`],
        /// [`RenderPassEncoder::write_timestamp`], or [`ComputePassEncoder::write_timestamp`] to
        /// write out a timestamp.
        ///
        /// They must be resolved using [`CommandEncoder::resolve_query_sets`] into a buffer,
        /// then the result must be multiplied by the timestamp period [`Device::get_timestamp_period`]
        /// to get the timestamp in nanoseconds. Multiple timestamps can then be diffed to get the
        /// time for operations between them to finish.
        ///
        /// Due to wgpu-hal limitations, this is only supported on vulkan for now.
        ///
        /// Supported Platforms:
        /// - Vulkan (works)
        /// - DX12 (works)
        ///
        /// This is a web and native feature.
        const TIMESTAMP_QUERY = 1 << 3;
        /// Enables use of Pipeline Statistics Queries. These queries tell the count of various operations
        /// performed between the start and stop call. Call [`RenderPassEncoder::begin_pipeline_statistics_query`] to start
        /// a query, then call [`RenderPassEncoder::end_pipeline_statistics_query`] to stop one.
        ///
        /// They must be resolved using [`CommandEncoder::resolve_query_sets`] into a buffer.
        /// The rules on how these resolve into buffers are detailed in the documentation for [`PipelineStatisticsTypes`].
        ///
        /// Due to wgpu-hal limitations, this is only supported on vulkan for now.
        ///
        /// Supported Platforms:
        /// - Vulkan (works)
        /// - DX12 (works)
        ///
        /// This is a web and native feature.
        const PIPELINE_STATISTICS_QUERY = 1 << 4;
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
        const MAPPABLE_PRIMARY_BUFFERS = 1 << 16;
        /// Allows the user to create uniform arrays of textures in shaders:
        ///
        /// eg. `uniform texture2D textures[10]`.
        ///
        /// If [`Features::STORAGE_RESOURCE_BINDING_ARRAY`] is supported as well as this, the user
        /// may also create uniform arrays of storage textures.
        ///
        /// eg. `uniform image2D textures[10]`.
        ///
        /// This capability allows them to exist and to be indexed by dynamically uniform
        /// values.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan
        ///
        /// This is a native only feature.
        const TEXTURE_BINDING_ARRAY = 1 << 17;
        /// Allows the user to create arrays of buffers in shaders:
        ///
        /// eg. `uniform myBuffer { .... } buffer_array[10]`.
        ///
        /// This capability allows them to exist and to be indexed by dynamically uniform
        /// values.
        ///
        /// If [`Features::STORAGE_RESOURCE_BINDING_ARRAY`] is supported as well as this, the user
        /// may also create arrays of storage buffers.
        ///
        /// eg. `buffer myBuffer { ... } buffer_array[10]`
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        ///
        /// This is a native only feature.
        const BUFFER_BINDING_ARRAY = 1 << 18;
        /// Allows the user to create uniform arrays of storage buffers or textures in shaders,
        /// if resp. [`Features::BUFFER_BINDING_ARRAY`] or [`Features::TEXTURE_BINDING_ARRAY`]
        /// is supported.
        ///
        /// This capability allows them to exist and to be indexed by dynamically uniform
        /// values.
        ///
        /// Supported platforms:
        /// - Metal (with MSL 2.2+ on macOS 10.13+)
        /// - Vulkan
        ///
        /// This is a native only feature.
        const STORAGE_RESOURCE_BINDING_ARRAY = 1 << 19;
        /// Allows shaders to index sampled texture and storage buffer resource arrays with dynamically non-uniform values:
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
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan 1.2+ (or VK_EXT_descriptor_indexing)'s shaderSampledImageArrayNonUniformIndexing & shaderStorageBufferArrayNonUniformIndexing feature)
        ///
        /// This is a native only feature.
        const SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING = 1 << 20;
        /// Allows shaders to index uniform buffer and storage texture resource arrays with dynamically non-uniform values:
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
        /// Supported platforms:
        /// - DX12
        /// - Metal (with MSL 2.0+ on macOS 10.13+)
        /// - Vulkan 1.2+ (or VK_EXT_descriptor_indexing)'s shaderUniformBufferArrayNonUniformIndexing & shaderStorageTextureArrayNonUniformIndexing feature)
        ///
        /// This is a native only feature.
        const UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING = 1 << 21;
        /// Allows the user to create bind groups continaing arrays with less bindings than the BindGroupLayout.
        ///
        /// This is a native only feature.
        const PARTIALLY_BOUND_BINDING_ARRAY = 1 << 22;
        /// Allows the user to create unsized uniform arrays of bindings:
        ///
        /// eg. `uniform texture2D textures[]`.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan 1.2+ (or VK_EXT_descriptor_indexing)'s runtimeDescriptorArray feature
        ///
        /// This is a native only feature.
        const UNSIZED_BINDING_ARRAY = 1 << 23;
        /// Allows the user to call [`RenderPass::multi_draw_indirect`] and [`RenderPass::multi_draw_indexed_indirect`].
        ///
        /// Allows multiple indirect calls to be dispatched from a single buffer.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        ///
        /// This is a native only feature.
        const MULTI_DRAW_INDIRECT = 1 << 24;
        /// Allows the user to call [`RenderPass::multi_draw_indirect_count`] and [`RenderPass::multi_draw_indexed_indirect_count`].
        ///
        /// This allows the use of a buffer containing the actual number of draw calls.
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan 1.2+ (or VK_KHR_draw_indirect_count)
        ///
        /// This is a native only feature.
        const MULTI_DRAW_INDIRECT_COUNT = 1 << 25;
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
        const PUSH_CONSTANTS = 1 << 26;
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
        const ADDRESS_MODE_CLAMP_TO_BORDER = 1 << 27;
        /// Allows the user to set [`PolygonMode::Line`] in [`PrimitiveState::polygon_mode`]
        ///
        /// This allows drawing polygons/triangles as lines (wireframe) instead of filled
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        /// - Metal
        ///
        /// This is a native only feature.
        const POLYGON_MODE_LINE= 1 << 28;
        /// Allows the user to set [`PolygonMode::Point`] in [`PrimitiveState::polygon_mode`]
        ///
        /// This allows only drawing the vertices of polygons/triangles instead of filled
        ///
        /// Supported platforms:
        /// - DX12
        /// - Vulkan
        ///
        /// This is a native only feature.
        const POLYGON_MODE_POINT = 1 << 29;
        /// Enables ETC family of compressed textures. All ETC textures use 4x4 pixel blocks.
        /// ETC2 RGB and RGBA1 are 8 bytes per block. RTC2 RGBA8 and EAC are 16 bytes per block.
        ///
        /// Compressed textures sacrifice some quality in exchange for significantly reduced
        /// bandwidth usage.
        ///
        /// Support for this feature guarantees availability of [`TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING`] for ETC2 formats.
        /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] may enable additional usages.
        ///
        /// Supported Platforms:
        /// - Intel/Vulkan
        /// - Mobile (some)
        ///
        /// This is a native-only feature.
        const TEXTURE_COMPRESSION_ETC2 = 1 << 30;
        /// Enables ASTC family of compressed textures. ASTC textures use pixel blocks varying from 4x4 to 12x12.
        /// Blocks are always 16 bytes.
        ///
        /// Compressed textures sacrifice some quality in exchange for significantly reduced
        /// bandwidth usage.
        ///
        /// Support for this feature guarantees availability of [`TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING`] for ASTC formats.
        /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] may enable additional usages.
        ///
        /// Supported Platforms:
        /// - Intel/Vulkan
        /// - Mobile (some)
        ///
        /// This is a native-only feature.
        const TEXTURE_COMPRESSION_ASTC_LDR = 1 << 31;
        /// Enables device specific texture format features.
        ///
        /// See `TextureFormatFeatures` for a listing of the features in question.
        ///
        /// By default only texture format properties as defined by the WebGPU specification are allowed.
        /// Enabling this feature flag extends the features of each format to the ones supported by the current device.
        /// Note that without this flag, read/write storage access is not allowed at all.
        ///
        /// This extension does not enable additional formats.
        ///
        /// This is a native-only feature.
        const TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES = 1 << 32;
        /// Enables 64-bit floating point types in SPIR-V shaders.
        ///
        /// Note: even when supported by GPU hardware, 64-bit floating point operations are
        /// frequently between 16 and 64 _times_ slower than equivalent operations on 32-bit floats.
        ///
        /// Supported Platforms:
        /// - Vulkan
        ///
        /// This is a native-only feature.
        const SHADER_FLOAT64 = 1 << 33;
        /// Enables using 64-bit types for vertex attributes.
        ///
        /// Requires SHADER_FLOAT64.
        ///
        /// Supported Platforms: N/A
        ///
        /// This is a native-only feature.
        const VERTEX_ATTRIBUTE_64BIT = 1 << 34;
        /// Allows the user to set a overestimation-conservative-rasterization in [`PrimitiveState::conservative`]
        ///
        /// Processing of degenerate triangles/lines is hardware specific.
        /// Only triangles are supported.
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        const CONSERVATIVE_RASTERIZATION = 1 << 35;
        /// Enables bindings of writable storage buffers and textures visible to vertex shaders.
        ///
        /// Note: some (tiled-based) platforms do not support vertex shaders with any side-effects.
        ///
        /// Supported Platforms:
        /// - All
        ///
        /// This is a native-only feature.
        const VERTEX_WRITABLE_STORAGE = 1 << 36;
        /// Enables clear to zero for buffers & textures.
        ///
        /// Supported platforms:
        /// - All
        ///
        /// This is a native only feature.
        const CLEAR_COMMANDS = 1 << 37;
        /// Enables creating shader modules from SPIR-V binary data (unsafe).
        ///
        /// SPIR-V data is not parsed or interpreted in any way; you can use
        /// [`wgpu::make_spirv_raw!`] to check for alignment and magic number when converting from
        /// raw bytes.
        ///
        /// Supported platforms:
        /// - Vulkan, in case shader's requested capabilities and extensions agree with
        /// Vulkan implementation.
        ///
        /// This is a native only feature.
        const SPIRV_SHADER_PASSTHROUGH = 1 << 38;
        /// Enables `builtin(primitive_index)` in fragment shaders.
        ///
        /// Note: enables geometry processing for pipelines using the builtin.
        /// This may come with a significant performance impact on some hardware.
        /// Other pipelines are not affected.
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        const SHADER_PRIMITIVE_INDEX = 1 << 39;
        /// Enables multiview render passes and `builtin(view_index)` in vertex shaders.
        ///
        /// Supported platforms:
        /// - Vulkan
        ///
        /// This is a native only feature.
        const MULTIVIEW = 1 << 40;
        /// Enables normalized `16-bit` texture formats.
        ///
        /// Supported platforms:
        /// - Vulkan
        /// - DX12
        /// - Metal
        ///
        /// This is a native only feature.
        const TEXTURE_FORMAT_16BIT_NORM = 1 << 41;
    }
}

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(Features);

impl Features {
    /// Mask of all features which are part of the upstream WebGPU standard.
    pub const fn all_webgpu_mask() -> Self {
        Self::from_bits_truncate(0x0000_0000_0000_FFFF)
    }

    /// Mask of all features that are only available when targeting native (not web).
    pub const fn all_native_mask() -> Self {
        Self::from_bits_truncate(0xFFFF_FFFF_FFFF_0000)
    }
}

/// Represents the sets of limits an adapter/device supports.
///
/// We provide three different defaults.
/// - [`Limits::downlevel_defaults()`]. This is a set of limits that is guarenteed to work on almost
///   all backends, including "downlevel" backends such as OpenGL and D3D11, other than WebGL. For
///   most applications we recommend using these limits, assuming they are high enough for your
///   application, and you do not intent to support WebGL.
/// - [`Limits::downlevel_webgl2_defaults()`] This is a set of limits that is lower even than the
///   [`downlevel_defaults()`], configured to be low enough to support running in the browser using
///   WebGL2.
/// - [`Limits::default()`]. This is the set of limits that is guarenteed to work on all modern
///   backends and is guarenteed to be supported by WebGPU. Applications needing more modern
///   features can use this as a reasonable set of limits if they are targetting only desktop and
///   modern mobile devices.
///
/// We recommend starting with the most restrictive limits you can and manually increasing the
/// limits you need boosted. This will let you stay running on all hardware that supports the limits
/// you need.
///
/// Limits "better" than the default must be supported by the adapter and requested when requesting
/// a device. If limits "better" than the adapter supports are requested, requesting a device will
/// panic. Once a device is requested, you may only use resources up to the limits requested _even_
/// if the adapter supports "better" limits.
///
/// Requesting limits that are "better" than you need may cause performance to decrease because the
/// implementation needs to support more than is needed. You should ideally only request exactly
/// what you need.
///
/// See also: <https://gpuweb.github.io/gpuweb/#dictdef-gpulimits>
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Limits {
    /// Maximum allowed value for the `size.width` of a texture created with `TextureDimension::D1`.
    /// Defaults to 8192. Higher is "better".
    pub max_texture_dimension_1d: u32,
    /// Maximum allowed value for the `size.width` and `size.height` of a texture created with `TextureDimension::D2`.
    /// Defaults to 8192. Higher is "better".
    pub max_texture_dimension_2d: u32,
    /// Maximum allowed value for the `size.width`, `size.height`, and `size.depth_or_array_layers`
    /// of a texture created with `TextureDimension::D3`.
    /// Defaults to 2048. Higher is "better".
    pub max_texture_dimension_3d: u32,
    /// Maximum allowed value for the `size.depth_or_array_layers` of a texture created with
    /// `TextureDimension::D1` or `TextureDimension::D2`.
    /// Defaults to 256. Higher is "better".
    pub max_texture_array_layers: u32,
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
    /// Maximum size in bytes of a binding to a uniform buffer. Defaults to 64 KB. Higher is "better".
    pub max_uniform_buffer_binding_size: u32,
    /// Maximum size in bytes of a binding to a storage buffer. Defaults to 128 MB. Higher is "better".
    pub max_storage_buffer_binding_size: u32,
    /// Maximum length of `VertexState::buffers` when creating a `RenderPipeline`.
    /// Defaults to 8. Higher is "better".
    pub max_vertex_buffers: u32,
    /// Maximum length of `VertexBufferLayout::attributes`, summed over all `VertexState::buffers`,
    /// when creating a `RenderPipeline`.
    /// Defaults to 16. Higher is "better".
    pub max_vertex_attributes: u32,
    /// Maximum value for `VertexBufferLayout::array_stride` when creating a `RenderPipeline`.
    /// Defaults to 2048. Higher is "better".
    pub max_vertex_buffer_array_stride: u32,
    /// Amount of storage available for push constants in bytes. Defaults to 0. Higher is "better".
    /// Requesting more than 0 during device creation requires [`Features::PUSH_CONSTANTS`] to be enabled.
    ///
    /// Expect the size to be:
    /// - Vulkan: 128-256 bytes
    /// - DX12: 256 bytes
    /// - Metal: 4096 bytes
    /// - DX11 & OpenGL don't natively support push constants, and are emulated with uniforms,
    ///   so this number is less useful but likely 256.
    pub max_push_constant_size: u32,
    /// Required `BufferBindingType::Uniform` alignment for `BufferBinding::offset`
    /// when creating a `BindGroup`, or for `set_bind_group` `dynamicOffsets`.
    /// Defaults to 256. Lower is "better".
    pub min_uniform_buffer_offset_alignment: u32,
    /// Required `BufferBindingType::Storage` alignment for `BufferBinding::offset`
    /// when creating a `BindGroup`, or for `set_bind_group` `dynamicOffsets`.
    /// Defaults to 256. Lower is "better".
    pub min_storage_buffer_offset_alignment: u32,
    /// Maximum allowed number of components (scalars) of input or output locations for
    /// inter-stage communication (vertex outputs to fragment inputs).
    pub max_inter_stage_shader_components: u32,
    /// Maximum number of bytes used for workgroup memory in a compute entry point.
    pub max_compute_workgroup_storage_size: u32,
    /// Maximum value of the product of the `workgroup_size` dimensions for a compute entry-point.
    pub max_compute_invocations_per_workgroup: u32,
    /// The maximum value of the workgroup_size X dimension for a compute stage `ShaderModule` entry-point.
    /// Defaults to 256.
    pub max_compute_workgroup_size_x: u32,
    /// The maximum value of the workgroup_size Y dimension for a compute stage `ShaderModule` entry-point.
    /// Defaults to 256.
    pub max_compute_workgroup_size_y: u32,
    /// The maximum value of the workgroup_size Z dimension for a compute stage `ShaderModule` entry-point.
    /// Defaults to 256.
    pub max_compute_workgroup_size_z: u32,
    /// The maximum value for each dimension of a `ComputePass::dispatch(x, y, z)` operation.
    /// Defaults to 65535.
    pub max_compute_workgroups_per_dimension: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_bind_groups: 4,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 4,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_textures_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_uniform_buffer_binding_size: 64 << 10,
            max_storage_buffer_binding_size: 128 << 20,
            max_vertex_buffers: 8,
            max_vertex_attributes: 16,
            max_vertex_buffer_array_stride: 2048,
            max_push_constant_size: 0,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_inter_stage_shader_components: 60,
            max_compute_workgroup_storage_size: 16352,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65535,
        }
    }
}

impl Limits {
    /// These default limits are guarenteed to be compatible with GLES-3.1, and D3D11
    pub fn downlevel_defaults() -> Self {
        Self {
            max_texture_dimension_1d: 2048,
            max_texture_dimension_2d: 2048,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 256,
            max_bind_groups: 4,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 4,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 4,
            max_storage_textures_per_shader_stage: 4,
            max_uniform_buffers_per_shader_stage: 12,
            max_uniform_buffer_binding_size: 16 << 10,
            max_storage_buffer_binding_size: 128 << 20,
            max_vertex_buffers: 8,
            max_vertex_attributes: 16,
            max_vertex_buffer_array_stride: 2048,
            max_push_constant_size: 0,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_inter_stage_shader_components: 60,
            max_compute_workgroup_storage_size: 16352,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65535,
        }
    }

    /// These default limits are guarenteed to be compatible with GLES-3.0, and D3D11, and WebGL2
    pub fn downlevel_webgl2_defaults() -> Self {
        Self {
            max_uniform_buffers_per_shader_stage: 11,
            max_storage_buffers_per_shader_stage: 0,
            max_storage_textures_per_shader_stage: 0,
            max_dynamic_storage_buffers_per_pipeline_layout: 0,
            max_storage_buffer_binding_size: 0,
            max_vertex_buffer_array_stride: 255,
            max_compute_workgroup_storage_size: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroup_size_y: 0,
            max_compute_workgroup_size_z: 0,
            max_compute_workgroups_per_dimension: 0,

            // Most of the values should be the same as the downlevel defaults
            ..Self::downlevel_defaults()
        }
    }

    /// Modify the current limits to use the resolution limits of the other.
    ///
    /// This is useful because the swapchain might need to be larger than any other image in the application.
    ///
    /// If your application only needs 512x512, you might be running on a 4k display and need extremely high resolution limits.
    pub fn using_resolution(self, other: Self) -> Self {
        Self {
            max_texture_dimension_1d: other.max_texture_dimension_1d,
            max_texture_dimension_2d: other.max_texture_dimension_2d,
            max_texture_dimension_3d: other.max_texture_dimension_3d,
            ..self
        }
    }

    /// Modify the current limits to use the buffer alignment limits of the adapter.
    ///
    /// This is useful for when you'd like to dynamically use the "best" supported buffer alignments.
    pub fn using_alignment(self, other: Self) -> Self {
        Self {
            min_uniform_buffer_offset_alignment: other.min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment: other.min_storage_buffer_offset_alignment,
            ..self
        }
    }
}

/// Represents the sets of additional limits on an adapter,
/// which take place when running on downlevel backends.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DownlevelLimits {}

#[allow(unknown_lints)] // derivable_impls is nightly only currently
#[allow(clippy::derivable_impls)]
impl Default for DownlevelLimits {
    fn default() -> Self {
        DownlevelLimits {}
    }
}

/// Lists various ways the underlying platform does not conform to the WebGPU standard.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DownlevelCapabilities {
    /// Combined boolean flags.
    pub flags: DownlevelFlags,
    /// Additional limits
    pub limits: DownlevelLimits,
    /// Which collections of features shaders support. Defined in terms of D3D's shader models.
    pub shader_model: ShaderModel,
}

impl Default for DownlevelCapabilities {
    fn default() -> Self {
        Self {
            flags: DownlevelFlags::compliant(),
            limits: DownlevelLimits::default(),
            shader_model: ShaderModel::Sm5,
        }
    }
}

impl DownlevelCapabilities {
    /// Returns true if the underlying platform offers complete support of the baseline WebGPU standard.
    ///
    /// If this returns false, some parts of the API will result in validation errors where they would not normally.
    /// These parts can be determined by the values in this structure.
    pub fn is_webgpu_compliant(&self) -> bool {
        self.flags.contains(DownlevelFlags::compliant())
            && self.limits == DownlevelLimits::default()
            && self.shader_model >= ShaderModel::Sm5
    }
}

bitflags::bitflags! {
    /// Binary flags listing features that may or may not be present on downlevel adapters.
    ///
    /// A downlevel adapter is a GPU adapter that WGPU supports, but with potentially limited
    /// features, due to the lack of hardware feature support.
    ///
    /// Flags that are **not** present for a downlevel adapter or device usually indicates
    /// non-compliance with the WebGPU specification, but not always.
    ///
    /// You can check whether a set of flags is compliant through the
    /// [`DownlevelCapabilities::is_webgpu_compliant()`] function.
    pub struct DownlevelFlags: u32 {
        /// The device supports compiling and using compute shaders.
        const COMPUTE_SHADERS = 1 << 0;
        /// Supports binding storage buffers and textures to fragment shaders.
        const FRAGMENT_WRITABLE_STORAGE = 1 << 1;
        /// Supports indirect drawing and dispatching.
        const INDIRECT_EXECUTION = 1 << 2;
        /// Supports non-zero `base_vertex` parameter to indexed draw calls.
        const BASE_VERTEX = 1 << 3;
        /// Supports reading from a depth/stencil buffer while using as a read-only depth/stencil
        /// attachment.
        const READ_ONLY_DEPTH_STENCIL = 1 << 4;
        /// Supports:
        /// - copy_image_to_image
        /// - copy_buffer_to_image and copy_image_to_buffer with a buffer without a MAP_* usage
        const DEVICE_LOCAL_IMAGE_COPIES = 1 << 5;
        /// Supports textures with mipmaps which have a non power of two size.
        const NON_POWER_OF_TWO_MIPMAPPED_TEXTURES = 1 << 6;
        /// Supports textures that are cube arrays.
        const CUBE_ARRAY_TEXTURES = 1 << 7;
        /// Supports comparison samplers.
        const COMPARISON_SAMPLERS = 1 << 8;
        /// Supports different blending modes per color target.
        const INDEPENDENT_BLENDING = 1 << 9;
        /// Supports storage buffers in vertex shaders.
        const VERTEX_STORAGE = 1 << 10;


        /// Supports samplers with anisotropic filtering. Note this isn't actually required by
        /// WebGPU, the implementation is allowed to completely ignore aniso clamp. This flag is
        /// here for native backends so they can comunicate to the user of aniso is enabled.
        const ANISOTROPIC_FILTERING = 1 << 11;

        /// Supports storage buffers in fragment shaders.
        const FRAGMENT_STORAGE = 1 << 12;
    }
}

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(DownlevelFlags);

impl DownlevelFlags {
    /// All flags that indicate if the backend is WebGPU compliant
    pub const fn compliant() -> Self {
        // We use manual bit twiddling to make this a const fn as `Sub` and `.remove` aren't const

        // WebGPU doesn't actually require aniso
        Self::from_bits_truncate(Self::all().bits() & !Self::ANISOTROPIC_FILTERING.bits)
    }
}

/// Collections of shader features a device supports if they support less than WebGPU normally allows.
// TODO: Fill out the differences between shader models more completely
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderModel {
    /// Extremely limited shaders, including a total instruction limit.
    Sm2,
    /// Missing minor features and storage images.
    Sm4,
    /// WebGPU supports shader module 5.
    Sm5,
}

/// Supported physical device types.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum DeviceType {
    /// Other.
    Other,
    /// Integrated GPU with shared CPU/GPU memory.
    IntegratedGpu,
    /// Discrete GPU with separate CPU/GPU memory.
    DiscreteGpu,
    /// Virtual / Hosted.
    VirtualGpu,
    /// Cpu / Software Rendering.
    Cpu,
}

//TODO: convert `vendor` and `device` to `u32`

/// Information about an adapter.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct AdapterInfo {
    /// Adapter name
    pub name: String,
    /// Vendor PCI id of the adapter
    pub vendor: usize,
    /// PCI id of the adapter
    pub device: usize,
    /// Type of device
    pub device_type: DeviceType,
    /// Backend used for device
    pub backend: Backend,
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
}

impl<L> DeviceDescriptor<L> {
    ///
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> DeviceDescriptor<K> {
        DeviceDescriptor {
            label: fun(&self.label),
            features: self.features,
            limits: self.limits.clone(),
        }
    }
}

bitflags::bitflags! {
    /// Describes the shader stages that a binding will be visible from.
    ///
    /// These can be combined so something that is visible from both vertex and fragment shaders can be defined as:
    ///
    /// `ShaderStages::VERTEX | ShaderStages::FRAGMENT`
    #[repr(transparent)]
    pub struct ShaderStages: u32 {
        /// Binding is not visible from any shader stage.
        const NONE = 0;
        /// Binding is visible from the vertex shader of a render pipeline.
        const VERTEX = 1 << 0;
        /// Binding is visible from the fragment shader of a render pipeline.
        const FRAGMENT = 1 << 1;
        /// Binding is visible from the compute shader of a compute pipeline.
        const COMPUTE = 1 << 2;
        /// Binding is visible from the vertex and fragment shaders of a render pipeline.
        const VERTEX_FRAGMENT = Self::VERTEX.bits | Self::FRAGMENT.bits;
    }
}

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(ShaderStages);

/// Dimensions of a particular texture view.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureViewDimension {
    /// A one dimensional texture. `texture1D` in glsl shaders.
    #[cfg_attr(feature = "serde", serde(rename = "1d"))]
    D1,
    /// A two dimensional texture. `texture2D` in glsl shaders.
    #[cfg_attr(feature = "serde", serde(rename = "2d"))]
    D2,
    /// A two dimensional array texture. `texture2DArray` in glsl shaders.
    #[cfg_attr(feature = "serde", serde(rename = "2d-array"))]
    D2Array,
    /// A cubemap texture. `textureCube` in glsl shaders.
    #[cfg_attr(feature = "serde", serde(rename = "cube"))]
    Cube,
    /// A cubemap array texture. `textureCubeArray` in glsl shaders.
    #[cfg_attr(feature = "serde", serde(rename = "cube-array"))]
    CubeArray,
    /// A three dimensional texture. `texture3D` in glsl shaders.
    #[cfg_attr(feature = "serde", serde(rename = "3d"))]
    D3,
}

impl Default for TextureViewDimension {
    fn default() -> Self {
        Self::D2
    }
}

impl TextureViewDimension {
    /// Get the texture dimension required of this texture view dimension.
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
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum BlendFactor {
    /// 0.0
    Zero = 0,
    /// 1.0
    One = 1,
    /// S.component
    Src = 2,
    /// 1.0 - S.component
    OneMinusSrc = 3,
    /// S.alpha
    SrcAlpha = 4,
    /// 1.0 - S.alpha
    OneMinusSrcAlpha = 5,
    /// D.component
    Dst = 6,
    /// 1.0 - D.component
    OneMinusDst = 7,
    /// D.alpha
    DstAlpha = 8,
    /// 1.0 - D.alpha
    OneMinusDstAlpha = 9,
    /// min(S.alpha, 1.0 - D.alpha)
    SrcAlphaSaturated = 10,
    /// Constant
    Constant = 11,
    /// 1.0 - Constant
    OneMinusConstant = 12,
}

/// Alpha blend operation.
///
/// Alpha blending is very complicated: see the OpenGL or Vulkan spec for more information.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum BlendOperation {
    /// Src + Dst
    Add = 0,
    /// Src - Dst
    Subtract = 1,
    /// Dst - Src
    ReverseSubtract = 2,
    /// min(Src, Dst)
    Min = 3,
    /// max(Src, Dst)
    Max = 4,
}

impl Default for BlendOperation {
    fn default() -> Self {
        Self::Add
    }
}

/// Describes the blend component of a pipeline.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct BlendComponent {
    /// Multiplier for the source, which is produced by the fragment shader.
    pub src_factor: BlendFactor,
    /// Multiplier for the destination, which is stored in the target.
    pub dst_factor: BlendFactor,
    /// The binary operation applied to the source and destination,
    /// multiplied by their respective factors.
    pub operation: BlendOperation,
}

impl BlendComponent {
    /// Default blending state that replaces destination with the source.
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };

    /// Blend state of (1 * src) + ((1 - src_alpha) * dst)
    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };

    /// Returns true if the state relies on the constant color, which is
    /// set independently on a render command encoder.
    pub fn uses_constant(&self) -> bool {
        match (self.src_factor, self.dst_factor) {
            (BlendFactor::Constant, _)
            | (BlendFactor::OneMinusConstant, _)
            | (_, BlendFactor::Constant)
            | (_, BlendFactor::OneMinusConstant) => true,
            (_, _) => false,
        }
    }
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self::REPLACE
    }
}

/// Describe the blend state of a render pipeline.
///
/// See the OpenGL or Vulkan spec for more information.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct BlendState {
    /// Color equation.
    pub color: BlendComponent,
    /// Alpha equation.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// Blend mode that does no color blending, just overwrites the output with the contents of the shader.
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };

    /// Blend mode that does standard alpha blending with non-premultiplied alpha.
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::OVER,
    };

    /// Blend mode that does standard alpha blending with premultiplied alpha.
    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        color: BlendComponent::OVER,
        alpha: BlendComponent::OVER,
    };
}

/// Describes the color state of a render pipeline.
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ColorTargetState {
    /// The [`TextureFormat`] of the image that this pipeline will render to. Must match the the format
    /// of the corresponding color attachment in [`CommandEncoder::begin_render_pass`].
    pub format: TextureFormat,
    /// The blending that is used for this pipeline.
    #[cfg_attr(feature = "serde", serde(default))]
    pub blend: Option<BlendState>,
    /// Mask which enables/disables writes to different color/alpha channel.
    #[cfg_attr(feature = "serde", serde(default))]
    pub write_mask: ColorWrites,
}

impl From<TextureFormat> for ColorTargetState {
    fn from(format: TextureFormat) -> Self {
        Self {
            format,
            blend: None,
            write_mask: ColorWrites::ALL,
        }
    }
}

/// Primitive type the input mesh is composed of.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
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
    /// Vertices `0 1 2 3 4 5` creates four triangles `0 1 2`, `2 1 3`, `2 3 4`, and `4 3 5`
    TriangleStrip = 4,
}

impl Default for PrimitiveTopology {
    fn default() -> Self {
        PrimitiveTopology::TriangleList
    }
}

impl PrimitiveTopology {
    /// Returns true for strip topologies.
    pub fn is_strip(&self) -> bool {
        match *self {
            Self::PointList | Self::LineList | Self::TriangleList => false,
            Self::LineStrip | Self::TriangleStrip => true,
        }
    }
}

/// Winding order which classifies the "front" face.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
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

/// Face of a vertex.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum Face {
    /// Front face
    Front = 0,
    /// Back face
    Back = 1,
}

/// Type of drawing mode for polygons
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum PolygonMode {
    /// Polygons are filled
    Fill = 0,
    /// Polygons are drawn as line segments
    Line = 1,
    /// Polygons are drawn as points
    Point = 2,
}

impl Default for PolygonMode {
    fn default() -> Self {
        Self::Fill
    }
}

/// Describes the state of primitive assembly and rasterization in a render pipeline.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct PrimitiveState {
    /// The primitive topology used to interpret vertices.
    pub topology: PrimitiveTopology,
    /// When drawing strip topologies with indices, this is the required format for the index buffer.
    /// This has no effect on non-indexed or non-strip draws.
    #[cfg_attr(feature = "serde", serde(default))]
    pub strip_index_format: Option<IndexFormat>,
    /// The face to consider the front for the purpose of culling and stencil operations.
    #[cfg_attr(feature = "serde", serde(default))]
    pub front_face: FrontFace,
    /// The face culling mode.
    #[cfg_attr(feature = "serde", serde(default))]
    pub cull_mode: Option<Face>,
    /// If set to true, the polygon depth is not clipped to 0-1 before rasterization.
    ///
    /// Enabling this requires `Features::DEPTH_CLIP_CONTROL` to be enabled.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unclipped_depth: bool,
    /// Controls the way each polygon is rasterized. Can be either `Fill` (default), `Line` or `Point`
    ///
    /// Setting this to `Line` requires `Features::POLYGON_MODE_LINE` to be enabled.
    ///
    /// Setting this to `Point` requires `Features::POLYGON_MODE_POINT` to be enabled.
    #[cfg_attr(feature = "serde", serde(default))]
    pub polygon_mode: PolygonMode,
    /// If set to true, the primitives are rendered with conservative overestimation. I.e. any rastered pixel touched by it is filled.
    /// Only valid for PolygonMode::Fill!
    ///
    /// Enabling this requires `Features::CONSERVATIVE_RASTERIZATION` to be enabled.
    pub conservative: bool,
}

/// Describes the multi-sampling state of a render pipeline.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct MultisampleState {
    /// The number of samples calculated per pixel (for MSAA). For non-multisampled textures,
    /// this should be `1`
    pub count: u32,
    /// Bitmask that restricts the samples of a pixel modified by this pipeline. All samples
    /// can be enabled using the value `!0`
    pub mask: u64,
    /// When enabled, produces another sample mask per pixel based on the alpha output value, that
    /// is ANDed with the sample_mask and the primitive coverage to restrict the set of samples
    /// affected by a primitive.
    ///
    /// The implicit mask produced for alpha of zero is guaranteed to be zero, and for alpha of one
    /// is guaranteed to be all 1-s.
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

bitflags::bitflags! {
    /// Feature flags for a texture format.
    #[repr(transparent)]
    pub struct TextureFormatFeatureFlags: u32 {
        /// When used as a STORAGE texture, then a texture with this format can be bound with
        /// [`StorageTextureAccess::ReadOnly`] or [`StorageTextureAccess::ReadWrite`].
        const STORAGE_READ_WRITE = 1 << 0;
        /// When used as a STORAGE texture, then a texture with this format can be written to with atomics.
        // TODO: No access flag exposed as of writing
        const STORAGE_ATOMICS = 1 << 1;
    }
}

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(TextureFormatFeatureFlags);

/// Features supported by a given texture format
///
/// Features are defined by WebGPU specification unless `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` is enabled.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct TextureFormatFeatures {
    /// Valid bits for `TextureDescriptor::Usage` provided for format creation.
    pub allowed_usages: TextureUsages,
    /// Additional property flags for the format.
    pub flags: TextureFormatFeatureFlags,
    /// If `filterable` is false, the texture can't be sampled with a filtering sampler.
    /// This may overwrite TextureSampleType::Float.filterable
    pub filterable: bool,
}

/// Information about a texture format.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct TextureFormatInfo {
    /// Features required (if any) to use the texture.
    pub required_features: Features,
    /// Type of sampling that is valid for the texture.
    pub sample_type: TextureSampleType,
    /// Dimension of a "block" of texels. This is always (1, 1) on uncompressed textures.
    pub block_dimensions: (u8, u8),
    /// Size in bytes of a "block" of texels. This is the size per pixel on uncompressed textures.
    pub block_size: u8,
    /// Count of components in the texture. This determines which components there will be actual data in the shader for.
    pub components: u8,
    /// Format will have colors be converted from srgb to linear on read and from linear to srgb on write.
    pub srgb: bool,
    /// Format features guaranteed by the WebGPU spec. Additional features are available if `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` is enabled.
    pub guaranteed_format_features: TextureFormatFeatures,
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
    #[cfg_attr(feature = "serde", serde(rename = "r8unorm"))]
    R8Unorm,
    /// Red channel only. 8 bit integer per channel. [-127, 127] converted to/from float [-1, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r8snorm"))]
    R8Snorm,
    /// Red channel only. 8 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r8uint"))]
    R8Uint,
    /// Red channel only. 8 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r8sint"))]
    R8Sint,

    // Normal 16 bit formats
    /// Red channel only. 16 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r16uint"))]
    R16Uint,
    /// Red channel only. 16 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r16sint"))]
    R16Sint,
    /// Red channel only. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "r16unorm"))]
    R16Unorm,
    /// Red channel only. 16 bit integer per channel. [0, 65535] converted to/from float [-1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "r16snorm"))]
    R16Snorm,
    /// Red channel only. 16 bit float per channel. Float in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r16float"))]
    R16Float,
    /// Red and green channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg8unorm"))]
    Rg8Unorm,
    /// Red and green channels. 8 bit integer per channel. [-127, 127] converted to/from float [-1, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg8snorm"))]
    Rg8Snorm,
    /// Red and green channels. 8 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg8uint"))]
    Rg8Uint,
    /// Red and green channels. 8 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg8sint"))]
    Rg8Sint,

    // Normal 32 bit formats
    /// Red channel only. 32 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r32uint"))]
    R32Uint,
    /// Red channel only. 32 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r32sint"))]
    R32Sint,
    /// Red channel only. 32 bit float per channel. Float in shader.
    #[cfg_attr(feature = "serde", serde(rename = "r32float"))]
    R32Float,
    /// Red and green channels. 16 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg16uint"))]
    Rg16Uint,
    /// Red and green channels. 16 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg16sint"))]
    Rg16Sint,
    /// Red and green channels. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "rg16unorm"))]
    Rg16Unorm,
    /// Red and green channels. 16 bit integer per channel. [0, 65535] converted to/from float [-1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "rg16snorm"))]
    Rg16Snorm,
    /// Red and green channels. 16 bit float per channel. Float in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg16float"))]
    Rg16Float,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba8unorm"))]
    Rgba8Unorm,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba8unorm-srgb"))]
    Rgba8UnormSrgb,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. [-127, 127] converted to/from float [-1, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba8snorm"))]
    Rgba8Snorm,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba8uint"))]
    Rgba8Uint,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba8sint"))]
    Rgba8Sint,
    /// Blue, green, red, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "bgra8unorm"))]
    Bgra8Unorm,
    /// Blue, green, red, and alpha channels. 8 bit integer per channel. Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "bgra8unorm-srgb"))]
    Bgra8UnormSrgb,

    // Packed 32 bit formats
    /// Red, green, blue, and alpha channels. 10 bit integer for RGB channels, 2 bit integer for alpha channel. [0, 1023] ([0, 3] for alpha) converted to/from float [0, 1] in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgb10a2unorm"))]
    Rgb10a2Unorm,
    /// Red, green, and blue channels. 11 bit float with no sign bit for RG channels. 10 bit float with no sign bit for blue channel. Float in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg11b10ufloat"))]
    Rg11b10Float,

    // Normal 64 bit formats
    /// Red and green channels. 32 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg32uint"))]
    Rg32Uint,
    /// Red and green channels. 32 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg32sint"))]
    Rg32Sint,
    /// Red and green channels. 32 bit float per channel. Float in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rg32float"))]
    Rg32Float,
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba16uint"))]
    Rgba16Uint,
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba16sint"))]
    Rgba16Sint,
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "rgba16unorm"))]
    Rgba16Unorm,
    /// Red, green, blue, and alpha. 16 bit integer per channel. [0, 65535] converted to/from float [-1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "rgba16snorm"))]
    Rgba16Snorm,
    /// Red, green, blue, and alpha channels. 16 bit float per channel. Float in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba16float"))]
    Rgba16Float,

    // Normal 128 bit formats
    /// Red, green, blue, and alpha channels. 32 bit integer per channel. Unsigned in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba32uint"))]
    Rgba32Uint,
    /// Red, green, blue, and alpha channels. 32 bit integer per channel. Signed in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba32sint"))]
    Rgba32Sint,
    /// Red, green, blue, and alpha channels. 32 bit float per channel. Float in shader.
    #[cfg_attr(feature = "serde", serde(rename = "rgba32float"))]
    Rgba32Float,

    // Depth and stencil formats
    /// Special depth format with 32 bit floating point depth.
    #[cfg_attr(feature = "serde", serde(rename = "depth32float"))]
    Depth32Float,
    /// Special depth format with at least 24 bit integer depth.
    #[cfg_attr(feature = "serde", serde(rename = "depth24plus"))]
    Depth24Plus,
    /// Special depth/stencil format with at least 24 bit integer depth and 8 bits integer stencil.
    #[cfg_attr(feature = "serde", serde(rename = "depth24plus-stencil8"))]
    Depth24PlusStencil8,

    // Packed uncompressed texture formats
    /// Packed unsigned float with 9 bits mantisa for each RGB component, then a common 5 bits exponent
    #[cfg_attr(feature = "serde", serde(rename = "rgb9e5ufloat"))]
    Rgb9e5Ufloat,

    // Compressed textures usable with `TEXTURE_COMPRESSION_BC` feature.
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 4 color + alpha pallet. 5 bit R + 6 bit G + 5 bit B + 1 bit alpha.
    /// [0, 63] ([0, 1] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc1-rgba-unorm"))]
    Bc1RgbaUnorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 4 color + alpha pallet. 5 bit R + 6 bit G + 5 bit B + 1 bit alpha.
    /// Srgb-color [0, 63] ([0, 1] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc1-rgba-unorm-srgb"))]
    Bc1RgbaUnormSrgb,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet. 5 bit R + 6 bit G + 5 bit B + 4 bit alpha.
    /// [0, 63] ([0, 15] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT3.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc2-rgba-unorm"))]
    Bc2RgbaUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet. 5 bit R + 6 bit G + 5 bit B + 4 bit alpha.
    /// Srgb-color [0, 63] ([0, 255] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT3.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc2-rgba-unorm-srgb"))]
    Bc2RgbaUnormSrgb,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet + 8 alpha pallet. 5 bit R + 6 bit G + 5 bit B + 8 bit alpha.
    /// [0, 63] ([0, 255] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT5.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc3-rgba-unorm"))]
    Bc3RgbaUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet + 8 alpha pallet. 5 bit R + 6 bit G + 5 bit B + 8 bit alpha.
    /// Srgb-color [0, 63] ([0, 255] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT5.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc3-rgba-unorm-srgb"))]
    Bc3RgbaUnormSrgb,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 8 color pallet. 8 bit R.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as RGTC1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc4-r-unorm"))]
    Bc4RUnorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 8 color pallet. 8 bit R.
    /// [-127, 127] converted to/from float [-1, 1] in shader.
    ///
    /// Also known as RGTC1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc4-r-snorm"))]
    Bc4RSnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 8 color red pallet + 8 color green pallet. 8 bit RG.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as RGTC2.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc5-rg-unorm"))]
    Bc5RgUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 8 color red pallet + 8 color green pallet. 8 bit RG.
    /// [-127, 127] converted to/from float [-1, 1] in shader.
    ///
    /// Also known as RGTC2.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc5-rg-snorm"))]
    Bc5RgSnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 16 bit unsigned float RGB. Float in shader.
    ///
    /// Also known as BPTC (float).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc6h-rgb-ufloat"))]
    Bc6hRgbUfloat,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 16 bit signed float RGB. Float in shader.
    ///
    /// Also known as BPTC (float).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc6h-rgb-float"))]
    Bc6hRgbSfloat,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as BPTC (unorm).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc7-rgba-unorm"))]
    Bc7RgbaUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as BPTC (unorm).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "bc7-rgba-unorm-srgb"))]
    Bc7RgbaUnormSrgb,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "etc2-rgb8unorm"))]
    Etc2Rgb8Unorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "etc2-rgb8unorm-srgb"))]
    Etc2Rgb8UnormSrgb,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB + 1 bit alpha.
    /// [0, 255] ([0, 1] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "etc2-rgb8a1unorm"))]
    Etc2Rgb8A1Unorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB + 1 bit alpha.
    /// Srgb-color [0, 255] ([0, 1] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "etc2-rgb8a1unorm-srgb"))]
    Etc2Rgb8A1UnormSrgb,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 8 bit integer RGB + 8 bit alpha.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "etc2-rgba8unorm"))]
    Etc2Rgba8Unorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 8 bit integer RGB + 8 bit alpha.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "etc2-rgba8unorm-srgb"))]
    Etc2Rgba8UnormSrgb,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 11 bit integer R.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "eac-r11unorm"))]
    EacR11Unorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 11 bit integer R.
    /// [-127, 127] converted to/from float [-1, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "eac-r11snorm"))]
    EacR11Snorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 11 bit integer R + 11 bit integer G.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "eac-rg11unorm"))]
    EacRg11Unorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 11 bit integer R + 11 bit integer G.
    /// [-127, 127] converted to/from float [-1, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "eac-rg11snorm"))]
    EacRg11Snorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-4x4-unorm"))]
    Astc4x4RgbaUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-4x4-unorm-srgb"))]
    Astc4x4RgbaUnormSrgb,
    /// 5x4 block compressed texture. 16 bytes per block (6.4 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-5x4-unorm"))]
    Astc5x4RgbaUnorm,
    /// 5x4 block compressed texture. 16 bytes per block (6.4 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-5x4-unorm-srgb"))]
    Astc5x4RgbaUnormSrgb,
    /// 5x5 block compressed texture. 16 bytes per block (5.12 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-5x5-unorm"))]
    Astc5x5RgbaUnorm,
    /// 5x5 block compressed texture. 16 bytes per block (5.12 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-5x5-unorm-srgb"))]
    Astc5x5RgbaUnormSrgb,
    /// 6x5 block compressed texture. 16 bytes per block (4.27 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-6x5-unorm"))]
    Astc6x5RgbaUnorm,
    /// 6x5 block compressed texture. 16 bytes per block (4.27 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-6x5-unorm-srgb"))]
    Astc6x5RgbaUnormSrgb,
    /// 6x6 block compressed texture. 16 bytes per block (3.56 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-6x6-unorm"))]
    Astc6x6RgbaUnorm,
    /// 6x6 block compressed texture. 16 bytes per block (3.56 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-6x6-unorm-srgb"))]
    Astc6x6RgbaUnormSrgb,
    /// 8x5 block compressed texture. 16 bytes per block (3.2 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-8x5-unorm"))]
    Astc8x5RgbaUnorm,
    /// 8x5 block compressed texture. 16 bytes per block (3.2 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-8x5-unorm-srgb"))]
    Astc8x5RgbaUnormSrgb,
    /// 8x6 block compressed texture. 16 bytes per block (2.67 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-8x6-unorm"))]
    Astc8x6RgbaUnorm,
    /// 8x6 block compressed texture. 16 bytes per block (2.67 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-8x6-unorm-srgb"))]
    Astc8x6RgbaUnormSrgb,
    /// 10x5 block compressed texture. 16 bytes per block (2.56 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-10x5-unorm"))]
    Astc10x5RgbaUnorm,
    /// 10x5 block compressed texture. 16 bytes per block (2.56 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-10x5-unorm-srgb"))]
    Astc10x5RgbaUnormSrgb,
    /// 10x6 block compressed texture. 16 bytes per block (2.13 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-10x6-unorm"))]
    Astc10x6RgbaUnorm,
    /// 10x6 block compressed texture. 16 bytes per block (2.13 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-10x6-unorm-srgb"))]
    Astc10x6RgbaUnormSrgb,
    /// 8x8 block compressed texture. 16 bytes per block (2 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-8x8-unorm"))]
    Astc8x8RgbaUnorm,
    /// 8x8 block compressed texture. 16 bytes per block (2 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-8x8-unorm-srgb"))]
    Astc8x8RgbaUnormSrgb,
    /// 10x8 block compressed texture. 16 bytes per block (1.6 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-10x8-unorm"))]
    Astc10x8RgbaUnorm,
    /// 10x8 block compressed texture. 16 bytes per block (1.6 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-10x8-unorm-srgb"))]
    Astc10x8RgbaUnormSrgb,
    /// 10x10 block compressed texture. 16 bytes per block (1.28 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-10x10-unorm"))]
    Astc10x10RgbaUnorm,
    /// 10x10 block compressed texture. 16 bytes per block (1.28 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-10x10-unorm-srgb"))]
    Astc10x10RgbaUnormSrgb,
    /// 12x10 block compressed texture. 16 bytes per block (1.07 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-12x10-unorm"))]
    Astc12x10RgbaUnorm,
    /// 12x10 block compressed texture. 16 bytes per block (1.07 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-12x10-unorm-srgb"))]
    Astc12x10RgbaUnormSrgb,
    /// 12x12 block compressed texture. 16 bytes per block (0.89 bit/px). Complex pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-12x12-unorm"))]
    Astc12x12RgbaUnorm,
    /// 12x12 block compressed texture. 16 bytes per block (0.89 bit/px). Complex pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_LDR`] must be enabled to use this texture format.
    #[cfg_attr(feature = "serde", serde(rename = "astc-12x12-unorm-srgb"))]
    Astc12x12RgbaUnormSrgb,
}

impl TextureFormat {
    /// Get useful information about the texture format.
    pub fn describe(&self) -> TextureFormatInfo {
        // Features
        let native = Features::empty();
        let bc = Features::TEXTURE_COMPRESSION_BC;
        let etc2 = Features::TEXTURE_COMPRESSION_ETC2;
        let astc_ldr = Features::TEXTURE_COMPRESSION_ASTC_LDR;
        let norm16bit = Features::TEXTURE_FORMAT_16BIT_NORM;

        // Sample Types
        let uint = TextureSampleType::Uint;
        let sint = TextureSampleType::Sint;
        let nearest = TextureSampleType::Float { filterable: false };
        let float = TextureSampleType::Float { filterable: true };
        let depth = TextureSampleType::Depth;

        // Color spaces
        let linear = false;
        let srgb = true;

        // Flags
        let basic =
            TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
        let attachment = basic | TextureUsages::RENDER_ATTACHMENT;
        let storage = basic | TextureUsages::STORAGE_BINDING;
        let all_flags = TextureUsages::all();

        // See <https://gpuweb.github.io/gpuweb/#texture-format-caps> for reference
        let (
            required_features,
            sample_type,
            srgb,
            block_dimensions,
            block_size,
            allowed_usages,
            components,
        ) = match self {
            // Normal 8 bit textures
            Self::R8Unorm => (native, float, linear, (1, 1), 1, attachment, 1),
            Self::R8Snorm => (native, float, linear, (1, 1), 1, basic, 1),
            Self::R8Uint => (native, uint, linear, (1, 1), 1, attachment, 1),
            Self::R8Sint => (native, sint, linear, (1, 1), 1, attachment, 1),

            // Normal 16 bit textures
            Self::R16Uint => (native, uint, linear, (1, 1), 2, attachment, 1),
            Self::R16Sint => (native, sint, linear, (1, 1), 2, attachment, 1),
            Self::R16Float => (native, float, linear, (1, 1), 2, attachment, 1),
            Self::Rg8Unorm => (native, float, linear, (1, 1), 2, attachment, 2),
            Self::Rg8Snorm => (native, float, linear, (1, 1), 2, attachment, 2),
            Self::Rg8Uint => (native, uint, linear, (1, 1), 2, attachment, 2),
            Self::Rg8Sint => (native, sint, linear, (1, 1), 2, basic, 2),

            // Normal 32 bit textures
            Self::R32Uint => (native, uint, linear, (1, 1), 4, all_flags, 1),
            Self::R32Sint => (native, sint, linear, (1, 1), 4, all_flags, 1),
            Self::R32Float => (native, nearest, linear, (1, 1), 4, all_flags, 1),
            Self::Rg16Uint => (native, uint, linear, (1, 1), 4, attachment, 2),
            Self::Rg16Sint => (native, sint, linear, (1, 1), 4, attachment, 2),
            Self::Rg16Float => (native, float, linear, (1, 1), 4, attachment, 2),
            Self::Rgba8Unorm => (native, float, linear, (1, 1), 4, all_flags, 4),
            Self::Rgba8UnormSrgb => (native, float, srgb, (1, 1), 4, attachment, 4),
            Self::Rgba8Snorm => (native, float, linear, (1, 1), 4, storage, 4),
            Self::Rgba8Uint => (native, uint, linear, (1, 1), 4, all_flags, 4),
            Self::Rgba8Sint => (native, sint, linear, (1, 1), 4, all_flags, 4),
            Self::Bgra8Unorm => (native, float, linear, (1, 1), 4, attachment, 4),
            Self::Bgra8UnormSrgb => (native, float, srgb, (1, 1), 4, attachment, 4),

            // Packed 32 bit textures
            Self::Rgb10a2Unorm => (native, float, linear, (1, 1), 4, attachment, 4),
            Self::Rg11b10Float => (native, float, linear, (1, 1), 4, basic, 3),

            // Packed 32 bit textures
            Self::Rg32Uint => (native, uint, linear, (1, 1), 8, all_flags, 2),
            Self::Rg32Sint => (native, sint, linear, (1, 1), 8, all_flags, 2),
            Self::Rg32Float => (native, nearest, linear, (1, 1), 8, all_flags, 2),
            Self::Rgba16Uint => (native, uint, linear, (1, 1), 8, all_flags, 4),
            Self::Rgba16Sint => (native, sint, linear, (1, 1), 8, all_flags, 4),
            Self::Rgba16Float => (native, float, linear, (1, 1), 8, all_flags, 4),

            // Packed 32 bit textures
            Self::Rgba32Uint => (native, uint, linear, (1, 1), 16, all_flags, 4),
            Self::Rgba32Sint => (native, sint, linear, (1, 1), 16, all_flags, 4),
            Self::Rgba32Float => (native, nearest, linear, (1, 1), 16, all_flags, 4),

            // Depth-stencil textures
            Self::Depth32Float => (native, depth, linear, (1, 1), 4, attachment, 1),
            Self::Depth24Plus => (native, depth, linear, (1, 1), 4, attachment, 1),
            Self::Depth24PlusStencil8 => (native, depth, linear, (1, 1), 4, attachment, 2),

            // Packed uncompressed
            Self::Rgb9e5Ufloat => (native, float, linear, (1, 1), 4, basic, 3),

            // BCn compressed textures
            Self::Bc1RgbaUnorm => (bc, float, linear, (4, 4), 8, basic, 4),
            Self::Bc1RgbaUnormSrgb => (bc, float, srgb, (4, 4), 8, basic, 4),
            Self::Bc2RgbaUnorm => (bc, float, linear, (4, 4), 16, basic, 4),
            Self::Bc2RgbaUnormSrgb => (bc, float, srgb, (4, 4), 16, basic, 4),
            Self::Bc3RgbaUnorm => (bc, float, linear, (4, 4), 16, basic, 4),
            Self::Bc3RgbaUnormSrgb => (bc, float, srgb, (4, 4), 16, basic, 4),
            Self::Bc4RUnorm => (bc, float, linear, (4, 4), 8, basic, 1),
            Self::Bc4RSnorm => (bc, float, linear, (4, 4), 8, basic, 1),
            Self::Bc5RgUnorm => (bc, float, linear, (4, 4), 16, basic, 2),
            Self::Bc5RgSnorm => (bc, float, linear, (4, 4), 16, basic, 2),
            Self::Bc6hRgbUfloat => (bc, float, linear, (4, 4), 16, basic, 3),
            Self::Bc6hRgbSfloat => (bc, float, linear, (4, 4), 16, basic, 3),
            Self::Bc7RgbaUnorm => (bc, float, linear, (4, 4), 16, basic, 4),
            Self::Bc7RgbaUnormSrgb => (bc, float, srgb, (4, 4), 16, basic, 4),

            // ETC compressed textures
            Self::Etc2Rgb8Unorm => (etc2, float, linear, (4, 4), 8, basic, 3),
            Self::Etc2Rgb8UnormSrgb => (etc2, float, srgb, (4, 4), 8, basic, 3),
            Self::Etc2Rgb8A1Unorm => (etc2, float, linear, (4, 4), 8, basic, 4),
            Self::Etc2Rgb8A1UnormSrgb => (etc2, float, srgb, (4, 4), 8, basic, 4),
            Self::Etc2Rgba8Unorm => (etc2, float, linear, (4, 4), 16, basic, 4),
            Self::Etc2Rgba8UnormSrgb => (etc2, float, srgb, (4, 4), 16, basic, 4),
            Self::EacR11Unorm => (etc2, float, linear, (4, 4), 8, basic, 1),
            Self::EacR11Snorm => (etc2, float, linear, (4, 4), 8, basic, 1),
            Self::EacRg11Unorm => (etc2, float, linear, (4, 4), 16, basic, 2),
            Self::EacRg11Snorm => (etc2, float, linear, (4, 4), 16, basic, 2),

            // ASTC compressed textures
            Self::Astc4x4RgbaUnorm => (astc_ldr, float, linear, (4, 4), 16, basic, 4),
            Self::Astc4x4RgbaUnormSrgb => (astc_ldr, float, srgb, (4, 4), 16, basic, 4),
            Self::Astc5x4RgbaUnorm => (astc_ldr, float, linear, (5, 4), 16, basic, 4),
            Self::Astc5x4RgbaUnormSrgb => (astc_ldr, float, srgb, (5, 4), 16, basic, 4),
            Self::Astc5x5RgbaUnorm => (astc_ldr, float, linear, (5, 5), 16, basic, 4),
            Self::Astc5x5RgbaUnormSrgb => (astc_ldr, float, srgb, (5, 5), 16, basic, 4),
            Self::Astc6x5RgbaUnorm => (astc_ldr, float, linear, (6, 5), 16, basic, 4),
            Self::Astc6x5RgbaUnormSrgb => (astc_ldr, float, srgb, (6, 5), 16, basic, 4),
            Self::Astc6x6RgbaUnorm => (astc_ldr, float, linear, (6, 6), 16, basic, 4),
            Self::Astc6x6RgbaUnormSrgb => (astc_ldr, float, srgb, (6, 6), 16, basic, 4),
            Self::Astc8x5RgbaUnorm => (astc_ldr, float, linear, (8, 5), 16, basic, 4),
            Self::Astc8x5RgbaUnormSrgb => (astc_ldr, float, srgb, (8, 5), 16, basic, 4),
            Self::Astc8x6RgbaUnorm => (astc_ldr, float, linear, (8, 6), 16, basic, 4),
            Self::Astc8x6RgbaUnormSrgb => (astc_ldr, float, srgb, (8, 6), 16, basic, 4),
            Self::Astc10x5RgbaUnorm => (astc_ldr, float, linear, (10, 5), 16, basic, 4),
            Self::Astc10x5RgbaUnormSrgb => (astc_ldr, float, srgb, (10, 5), 16, basic, 4),
            Self::Astc10x6RgbaUnorm => (astc_ldr, float, linear, (10, 6), 16, basic, 4),
            Self::Astc10x6RgbaUnormSrgb => (astc_ldr, float, srgb, (10, 6), 16, basic, 4),
            Self::Astc8x8RgbaUnorm => (astc_ldr, float, linear, (8, 8), 16, basic, 4),
            Self::Astc8x8RgbaUnormSrgb => (astc_ldr, float, srgb, (8, 8), 16, basic, 4),
            Self::Astc10x8RgbaUnorm => (astc_ldr, float, linear, (10, 8), 16, basic, 4),
            Self::Astc10x8RgbaUnormSrgb => (astc_ldr, float, srgb, (10, 8), 16, basic, 4),
            Self::Astc10x10RgbaUnorm => (astc_ldr, float, linear, (10, 10), 16, basic, 4),
            Self::Astc10x10RgbaUnormSrgb => (astc_ldr, float, srgb, (10, 10), 16, basic, 4),
            Self::Astc12x10RgbaUnorm => (astc_ldr, float, linear, (12, 10), 16, basic, 4),
            Self::Astc12x10RgbaUnormSrgb => (astc_ldr, float, srgb, (12, 10), 16, basic, 4),
            Self::Astc12x12RgbaUnorm => (astc_ldr, float, linear, (12, 12), 16, basic, 4),
            Self::Astc12x12RgbaUnormSrgb => (astc_ldr, float, srgb, (12, 12), 16, basic, 4),

            // Optional normalized 16-bit-per-channel formats
            Self::R16Unorm => (norm16bit, float, linear, (1, 1), 2, storage, 1),
            Self::R16Snorm => (norm16bit, float, linear, (1, 1), 2, storage, 1),
            Self::Rg16Unorm => (norm16bit, float, linear, (1, 1), 4, storage, 2),
            Self::Rg16Snorm => (norm16bit, float, linear, (1, 1), 4, storage, 2),
            Self::Rgba16Unorm => (norm16bit, float, linear, (1, 1), 8, storage, 4),
            Self::Rgba16Snorm => (norm16bit, float, linear, (1, 1), 8, storage, 4),
        };

        TextureFormatInfo {
            required_features,
            sample_type,
            block_dimensions,
            block_size,
            components,
            srgb,
            guaranteed_format_features: TextureFormatFeatures {
                allowed_usages,
                flags: TextureFormatFeatureFlags::empty(),
                filterable: sample_type == TextureSampleType::Float { filterable: true },
            },
        }
    }
}

bitflags::bitflags! {
    /// Color write mask. Disabled color channels will not be written to.
    #[repr(transparent)]
    pub struct ColorWrites: u32 {
        /// Enable red channel writes
        const RED = 1 << 0;
        /// Enable green channel writes
        const GREEN = 1 << 1;
        /// Enable blue channel writes
        const BLUE = 1 << 2;
        /// Enable alpha channel writes
        const ALPHA = 1 << 3;
        /// Enable red, green, and blue channel writes
        const COLOR = Self::RED.bits | Self::GREEN.bits | Self::BLUE.bits;
        /// Enable writes to all channels.
        const ALL = Self::RED.bits | Self::GREEN.bits | Self::BLUE.bits | Self::ALPHA.bits;
    }
}

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(ColorWrites);

impl Default for ColorWrites {
    fn default() -> Self {
        Self::ALL
    }
}

/// State of the stencil operation (fixed-pipeline stage).
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct StencilState {
    /// Front face mode.
    pub front: StencilFaceState,
    /// Back face mode.
    pub back: StencilFaceState,
    /// Stencil values are AND'd with this mask when reading and writing from the stencil buffer. Only low 8 bits are used.
    pub read_mask: u32,
    /// Stencil values are AND'd with this mask when writing to the stencil buffer. Only low 8 bits are used.
    pub write_mask: u32,
}

impl StencilState {
    /// Returns true if the stencil test is enabled.
    pub fn is_enabled(&self) -> bool {
        (self.front != StencilFaceState::IGNORE || self.back != StencilFaceState::IGNORE)
            && (self.read_mask != 0 || self.write_mask != 0)
    }
    /// Returns true if the state doesn't mutate the target values.
    pub fn is_read_only(&self) -> bool {
        self.write_mask == 0
    }
    /// Returns true if the stencil state uses the reference value for testing.
    pub fn needs_ref_value(&self) -> bool {
        self.front.needs_ref_value() || self.back.needs_ref_value()
    }
}

/// Describes the biasing setting for the depth target.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct DepthBiasState {
    /// Constant depth biasing factor, in basic units of the depth format.
    pub constant: i32,
    /// Slope depth biasing factor.
    pub slope_scale: f32,
    /// Depth bias clamp value (absolute).
    pub clamp: f32,
}

impl DepthBiasState {
    /// Returns true if the depth biasing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.constant != 0 || self.slope_scale != 0.0
    }
}

/// Describes the depth/stencil state in a render pipeline.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct DepthStencilState {
    /// Format of the depth/stencil buffer, must be special depth format. Must match the the format
    /// of the depth/stencil attachment in [`CommandEncoder::begin_render_pass`].
    pub format: TextureFormat,
    /// If disabled, depth will not be written to.
    pub depth_write_enabled: bool,
    /// Comparison function used to compare depth values in the depth test.
    pub depth_compare: CompareFunction,
    /// Stencil state.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
    pub stencil: StencilState,
    /// Depth bias state.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
    pub bias: DepthBiasState,
}

impl DepthStencilState {
    /// Returns true if the depth testing is enabled.
    pub fn is_depth_enabled(&self) -> bool {
        self.depth_compare != CompareFunction::Always || self.depth_write_enabled
    }
    /// Returns true if the state doesn't mutate either depth or stencil of the target.
    pub fn is_read_only(&self) -> bool {
        !self.depth_write_enabled && self.stencil.is_read_only()
    }
}

/// Format of indices used with pipeline.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
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
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
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
/// If you are not using stencil state, set this to [`StencilFaceState::IGNORE`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct StencilFaceState {
    /// Comparison function that determines if the fail_op or pass_op is used on the stencil buffer.
    pub compare: CompareFunction,
    /// Operation that is preformed when stencil test fails.
    pub fail_op: StencilOperation,
    /// Operation that is performed when depth test fails but stencil test succeeds.
    pub depth_fail_op: StencilOperation,
    /// Operation that is performed when stencil test success.
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    /// Ignore the stencil state for the face.
    pub const IGNORE: Self = StencilFaceState {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };

    /// Returns true if the face state uses the reference value for testing or operation.
    pub fn needs_ref_value(&self) -> bool {
        self.compare.needs_ref_value()
            || self.fail_op == StencilOperation::Replace
            || self.depth_fail_op == StencilOperation::Replace
            || self.pass_op == StencilOperation::Replace
    }
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self::IGNORE
    }
}

/// Comparison function used for depth and stencil operations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
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
    /// Returns true if the comparison depends on the reference value.
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
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum VertexStepMode {
    /// Vertex data is advanced every vertex.
    Vertex = 0,
    /// Vertex data is advanced every instance.
    Instance = 1,
}

impl Default for VertexStepMode {
    fn default() -> Self {
        VertexStepMode::Vertex
    }
}

/// Vertex inputs (attributes) to shaders.
///
/// Arrays of these can be made with the [`vertex_attr_array`] macro. Vertex attributes are assumed to be tightly packed.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct VertexAttribute {
    /// Format of the input
    pub format: VertexFormat,
    /// Byte offset of the start of the input
    pub offset: BufferAddress,
    /// Location for this input. Must match the location in the shader.
    pub shader_location: ShaderLocation,
}

/// Vertex Format for a Vertex Attribute (input).
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum VertexFormat {
    /// Two unsigned bytes (u8). `uvec2` in shaders.
    Uint8x2 = 0,
    /// Four unsigned bytes (u8). `uvec4` in shaders.
    Uint8x4 = 1,
    /// Two signed bytes (i8). `ivec2` in shaders.
    Sint8x2 = 2,
    /// Four signed bytes (i8). `ivec4` in shaders.
    Sint8x4 = 3,
    /// Two unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec2` in shaders.
    Unorm8x2 = 4,
    /// Four unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec4` in shaders.
    Unorm8x4 = 5,
    /// Two signed bytes (i8). [-127, 127] converted to float [-1, 1] `vec2` in shaders.
    Snorm8x2 = 6,
    /// Four signed bytes (i8). [-127, 127] converted to float [-1, 1] `vec4` in shaders.
    Snorm8x4 = 7,
    /// Two unsigned shorts (u16). `uvec2` in shaders.
    Uint16x2 = 8,
    /// Four unsigned shorts (u16). `uvec4` in shaders.
    Uint16x4 = 9,
    /// Two signed shorts (i16). `ivec2` in shaders.
    Sint16x2 = 10,
    /// Four signed shorts (i16). `ivec4` in shaders.
    Sint16x4 = 11,
    /// Two unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec2` in shaders.
    Unorm16x2 = 12,
    /// Four unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec4` in shaders.
    Unorm16x4 = 13,
    /// Two signed shorts (i16). [-32767, 32767] converted to float [-1, 1] `vec2` in shaders.
    Snorm16x2 = 14,
    /// Four signed shorts (i16). [-32767, 32767] converted to float [-1, 1] `vec4` in shaders.
    Snorm16x4 = 15,
    /// Two half-precision floats (no Rust equiv). `vec2` in shaders.
    Float16x2 = 16,
    /// Four half-precision floats (no Rust equiv). `vec4` in shaders.
    Float16x4 = 17,
    /// One single-precision float (f32). `float` in shaders.
    Float32 = 18,
    /// Two single-precision floats (f32). `vec2` in shaders.
    Float32x2 = 19,
    /// Three single-precision floats (f32). `vec3` in shaders.
    Float32x3 = 20,
    /// Four single-precision floats (f32). `vec4` in shaders.
    Float32x4 = 21,
    /// One unsigned int (u32). `uint` in shaders.
    Uint32 = 22,
    /// Two unsigned ints (u32). `uvec2` in shaders.
    Uint32x2 = 23,
    /// Three unsigned ints (u32). `uvec3` in shaders.
    Uint32x3 = 24,
    /// Four unsigned ints (u32). `uvec4` in shaders.
    Uint32x4 = 25,
    /// One signed int (i32). `int` in shaders.
    Sint32 = 26,
    /// Two signed ints (i32). `ivec2` in shaders.
    Sint32x2 = 27,
    /// Three signed ints (i32). `ivec3` in shaders.
    Sint32x3 = 28,
    /// Four signed ints (i32). `ivec4` in shaders.
    Sint32x4 = 29,
    /// One double-precision float (f64). `double` in shaders. Requires VERTEX_ATTRIBUTE_64BIT features.
    Float64 = 30,
    /// Two double-precision floats (f64). `dvec2` in shaders. Requires VERTEX_ATTRIBUTE_64BIT features.
    Float64x2 = 31,
    /// Three double-precision floats (f64). `dvec3` in shaders. Requires VERTEX_ATTRIBUTE_64BIT features.
    Float64x3 = 32,
    /// Four double-precision floats (f64). `dvec4` in shaders. Requires VERTEX_ATTRIBUTE_64BIT features.
    Float64x4 = 33,
}

impl VertexFormat {
    /// Returns the byte size of the format.
    pub const fn size(&self) -> u64 {
        match self {
            Self::Uint8x2 | Self::Sint8x2 | Self::Unorm8x2 | Self::Snorm8x2 => 2,
            Self::Uint8x4
            | Self::Sint8x4
            | Self::Unorm8x4
            | Self::Snorm8x4
            | Self::Uint16x2
            | Self::Sint16x2
            | Self::Unorm16x2
            | Self::Snorm16x2
            | Self::Float16x2
            | Self::Float32
            | Self::Uint32
            | Self::Sint32 => 4,
            Self::Uint16x4
            | Self::Sint16x4
            | Self::Unorm16x4
            | Self::Snorm16x4
            | Self::Float16x4
            | Self::Float32x2
            | Self::Uint32x2
            | Self::Sint32x2
            | Self::Float64 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 | Self::Float64x2 => 16,
            Self::Float64x3 => 24,
            Self::Float64x4 => 32,
        }
    }
}

bitflags::bitflags! {
    /// Different ways that you can use a buffer.
    ///
    /// The usages determine what kind of memory the buffer is allocated from and what
    /// actions the buffer can partake in.
    #[repr(transparent)]
    pub struct BufferUsages: u32 {
        /// Allow a buffer to be mapped for reading using [`Buffer::map_async`] + [`Buffer::get_mapped_range`].
        /// This does not include creating a buffer with [`BufferDescriptor::mapped_at_creation`] set.
        ///
        /// If [`Features::MAPPABLE_PRIMARY_BUFFERS`] isn't enabled, the only other usage a buffer
        /// may have is COPY_DST.
        const MAP_READ = 1 << 0;
        /// Allow a buffer to be mapped for writing using [`Buffer::map_async`] + [`Buffer::get_mapped_range_mut`].
        /// This does not include creating a buffer with `mapped_at_creation` set.
        ///
        /// If [`Features::MAPPABLE_PRIMARY_BUFFERS`] feature isn't enabled, the only other usage a buffer
        /// may have is COPY_SRC.
        const MAP_WRITE = 1 << 1;
        /// Allow a buffer to be the source buffer for a [`CommandEncoder::copy_buffer_to_buffer`] or [`CommandEncoder::copy_buffer_to_texture`]
        /// operation.
        const COPY_SRC = 1 << 2;
        /// Allow a buffer to be the destination buffer for a [`CommandEncoder::copy_buffer_to_buffer`], [`CommandEncoder::copy_texture_to_buffer`],
        /// [`CommandEncoder::clear_buffer`] or [`Queue::write_buffer`] operation.
        const COPY_DST = 1 << 3;
        /// Allow a buffer to be the index buffer in a draw operation.
        const INDEX = 1 << 4;
        /// Allow a buffer to be the vertex buffer in a draw operation.
        const VERTEX = 1 << 5;
        /// Allow a buffer to be a [`BufferBindingType::Uniform`] inside a bind group.
        const UNIFORM = 1 << 6;
        /// Allow a buffer to be a [`BufferBindingType::Storage`] inside a bind group.
        const STORAGE = 1 << 7;
        /// Allow a buffer to be the indirect buffer in an indirect draw call.
        const INDIRECT = 1 << 8;
    }
}

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(BufferUsages);

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
    pub usage: BufferUsages,
    /// Allows a buffer to be mapped immediately after they are made. It does not have to be [`BufferUsages::MAP_READ`] or
    /// [`BufferUsages::MAP_WRITE`], all buffers are allowed to be mapped at creation.
    pub mapped_at_creation: bool,
}

impl<L> BufferDescriptor<L> {
    ///
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
    ///
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
    pub struct TextureUsages: u32 {
        /// Allows a texture to be the source in a [`CommandEncoder::copy_texture_to_buffer`] or
        /// [`CommandEncoder::copy_texture_to_texture`] operation.
        const COPY_SRC = 1 << 0;
        /// Allows a texture to be the destination in a  [`CommandEncoder::copy_buffer_to_texture`],
        /// [`CommandEncoder::copy_texture_to_texture`], or [`Queue::write_texture`] operation.
        const COPY_DST = 1 << 1;
        /// Allows a texture to be a [`BindingType::Texture`] in a bind group.
        const TEXTURE_BINDING = 1 << 2;
        /// Allows a texture to be a [`BindingType::StorageTexture`] in a bind group.
        const STORAGE_BINDING = 1 << 3;
        /// Allows a texture to be an output attachment of a renderpass.
        const RENDER_ATTACHMENT = 1 << 4;
    }
}

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(TextureUsages);

/// Configures a [`Surface`] for presentation.
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct SurfaceConfiguration {
    /// The usage of the swap chain. The only supported usage is `RENDER_ATTACHMENT`.
    pub usage: TextureUsages,
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

/// Status of the recieved surface image.
#[repr(C)]
#[derive(Debug)]
pub enum SurfaceStatus {
    /// No issues.
    Good,
    /// The swap chain is operational, but it does no longer perfectly
    /// match the surface. A re-configuration is needed.
    Suboptimal,
    /// Unable to get the next frame, timed out.
    Timeout,
    /// The surface under the swap chain has changed.
    Outdated,
    /// The surface under the swap chain is lost.
    Lost,
}

/// RGBA double precision color.
///
/// This is not to be used as a generic color type, only for specific wgpu interfaces.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Color {
    ///
    pub r: f64,
    ///
    pub g: f64,
    ///
    pub b: f64,
    ///
    pub a: f64,
}

#[allow(missing_docs)]
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
    #[cfg_attr(feature = "serde", serde(rename = "1d"))]
    D1,
    /// 2D texture
    #[cfg_attr(feature = "serde", serde(rename = "2d"))]
    D2,
    /// 3D texture
    #[cfg_attr(feature = "serde", serde(rename = "3d"))]
    D3,
}

/// Origin of a copy to/from a texture.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Origin3d {
    ///
    pub x: u32,
    ///
    pub y: u32,
    ///
    pub z: u32,
}

impl Origin3d {
    /// Zero origin.
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
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Extent3d {
    ///
    pub width: u32,
    ///
    pub height: u32,
    ///
    #[cfg_attr(feature = "serde", serde(default = "default_depth"))]
    pub depth_or_array_layers: u32,
}

#[cfg(feature = "serde")]
fn default_depth() -> u32 {
    1
}

impl Default for Extent3d {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        }
    }
}

impl Extent3d {
    /// Calculates the [physical size] is backing an texture of the given format and extent.
    /// This includes padding to the block width and height of the format.
    ///
    /// This is the texture extent that you must upload at when uploading to _mipmaps_ of compressed textures.
    ///
    /// ```rust
    /// # use wgpu_types as wgpu;
    /// let format = wgpu::TextureFormat::Bc1RgbaUnormSrgb; // 4x4 blocks
    /// assert_eq!(
    ///     wgpu::Extent3d { width: 7, height: 7, depth_or_array_layers: 1 }.physical_size(format),
    ///     wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 }
    /// );
    /// // Doesn't change, already aligned
    /// assert_eq!(
    ///     wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 }.physical_size(format),
    ///     wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 }
    /// );
    /// let format = wgpu::TextureFormat::Astc8x5RgbaUnorm; // 8x5 blocks
    /// assert_eq!(
    ///     wgpu::Extent3d { width: 7, height: 7, depth_or_array_layers: 1 }.physical_size(format),
    ///     wgpu::Extent3d { width: 8, height: 10, depth_or_array_layers: 1 }
    /// );
    /// ```
    ///
    /// [physical size]: https://gpuweb.github.io/gpuweb/#physical-size
    pub fn physical_size(&self, format: TextureFormat) -> Self {
        let (block_width, block_height) = format.describe().block_dimensions;
        let block_width = block_width as u32;
        let block_height = block_height as u32;

        let width = ((self.width + block_width - 1) / block_width) * block_width;
        let height = ((self.height + block_height - 1) / block_height) * block_height;

        Self {
            width,
            height,
            depth_or_array_layers: self.depth_or_array_layers,
        }
    }

    /// Calculates the maximum possible count of mipmaps.
    ///
    /// Treats the depth as part of the mipmaps. If calculating
    /// for a 2DArray texture, which does not mipmap depth, set depth to 1.
    ///
    /// ```rust
    /// # use wgpu_types as wgpu;
    /// assert_eq!(wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }.max_mips(), 1);
    /// assert_eq!(wgpu::Extent3d { width: 60, height: 60, depth_or_array_layers: 1 }.max_mips(), 6);
    /// assert_eq!(wgpu::Extent3d { width: 240, height: 1, depth_or_array_layers: 1 }.max_mips(), 8);
    /// ```
    pub fn max_mips(&self) -> u32 {
        let max_dim = self.width.max(self.height.max(self.depth_or_array_layers));
        32 - max_dim.leading_zeros()
    }

    /// Calculates the extent at a given mip level.
    /// Does *not* account for memory size being a multiple of block size.
    pub fn mip_level_size(&self, level: u32, is_3d_texture: bool) -> Extent3d {
        Extent3d {
            width: u32::max(1, self.width >> level),
            height: u32::max(1, self.height >> level),
            depth_or_array_layers: match is_3d_texture {
                false => self.depth_or_array_layers,
                true => u32::max(1, self.depth_or_array_layers >> level),
            },
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
    /// Size of the texture. All components must be greater than zero. For a
    /// regular 1D/2D texture, the unused sizes will be 1. For 2DArray textures,
    /// Z is the number of 2D textures in that array.
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
    pub usage: TextureUsages,
}

impl<L> TextureDescriptor<L> {
    ///
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

    /// Calculates the extent at a given mip level.
    ///
    /// If the given mip level is larger than possible, returns None.
    ///
    /// Treats the depth as part of the mipmaps. If calculating
    /// for a 2DArray texture, which does not mipmap depth, set depth to 1.
    ///
    /// ```rust
    /// # use wgpu_types as wgpu;
    /// let desc = wgpu::TextureDescriptor {
    ///   label: (),
    ///   size: wgpu::Extent3d { width: 100, height: 60, depth_or_array_layers: 1 },
    ///   mip_level_count: 7,
    ///   sample_count: 1,
    ///   dimension: wgpu::TextureDimension::D3,
    ///   format: wgpu::TextureFormat::Rgba8Sint,
    ///   usage: wgpu::TextureUsages::empty(),
    /// };
    ///
    /// assert_eq!(desc.mip_level_size(0), Some(wgpu::Extent3d { width: 100, height: 60, depth_or_array_layers: 1 }));
    /// assert_eq!(desc.mip_level_size(1), Some(wgpu::Extent3d { width: 50, height: 30, depth_or_array_layers: 1 }));
    /// assert_eq!(desc.mip_level_size(2), Some(wgpu::Extent3d { width: 25, height: 15, depth_or_array_layers: 1 }));
    /// assert_eq!(desc.mip_level_size(3), Some(wgpu::Extent3d { width: 12, height: 7, depth_or_array_layers: 1 }));
    /// assert_eq!(desc.mip_level_size(4), Some(wgpu::Extent3d { width: 6, height: 3, depth_or_array_layers: 1 }));
    /// assert_eq!(desc.mip_level_size(5), Some(wgpu::Extent3d { width: 3, height: 1, depth_or_array_layers: 1 }));
    /// assert_eq!(desc.mip_level_size(6), Some(wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }));
    /// assert_eq!(desc.mip_level_size(7), None);
    /// ```
    pub fn mip_level_size(&self, level: u32) -> Option<Extent3d> {
        if level >= self.mip_level_count {
            return None;
        }

        Some(
            self.size
                .mip_level_size(level, self.dimension == TextureDimension::D3),
        )
    }

    /// Returns the number of array layers.
    pub fn array_layer_count(&self) -> u32 {
        match self.dimension {
            TextureDimension::D1 | TextureDimension::D2 => self.size.depth_or_array_layers,
            TextureDimension::D3 => 1,
        }
    }
}

/// Kind of data the texture holds.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
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
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
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
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
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
    pub stages: ShaderStages,
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
    /// Debug label of this command buffer.
    pub label: L,
}

impl<L> CommandBufferDescriptor<L> {
    ///
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CommandBufferDescriptor<K> {
        CommandBufferDescriptor {
            label: fun(&self.label),
        }
    }
}

/// Describes the depth/stencil attachment for render bundles.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct RenderBundleDepthStencil {
    /// Format of the attachment.
    pub format: TextureFormat,
    /// True if the depth aspect is used but not modified.
    pub depth_read_only: bool,
    /// True if the stencil aspect is used but not modified.
    pub stencil_read_only: bool,
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
    ///
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
///
/// The bytes per row and rows per image can be hard to figure out so here are some examples:
///
/// | Resolution | Format | Bytes per block | Pixels per block | Bytes per row                          | Rows per image               |
/// |------------|--------|-----------------|------------------|----------------------------------------|------------------------------|
/// | 256x256    | RGBA8  | 4               | 1 * 1 * 1        | 256 * 4 = Some(1024)                   | None                         |
/// | 32x16x8    | RGBA8  | 4               | 1 * 1 * 1        | 32 * 4 = 128 padded to 256 = Some(256) | None                         |
/// | 256x256    | BC3    | 16              | 4 * 4 * 1        | 16 * (256 / 4) = 1024 = Some(1024)     | None                         |
/// | 64x64x8    | BC3    | 16              | 4 * 4 * 1        | 16 * (64 / 4) = 256 = Some(256)        | 64 / 4 = 16 = Some(16)       |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ImageDataLayout {
    /// Offset into the buffer that is the start of the texture. Must be a multiple of texture block size.
    /// For non-compressed textures, this is 1.
    pub offset: BufferAddress,
    /// Bytes per "row" in an image.
    ///
    /// A row is one row of pixels or of compressed blocks in the x direction.
    ///
    /// This value is required if there are multiple rows (i.e. height or depth is more than one pixel or pixel block for compressed textures)
    ///
    /// Must be a multiple of 256 for [`CommandEncoder::copy_buffer_to_texture`] and [`CommandEncoder::copy_texture_to_buffer`]. You must manually pad
    /// the image such that this is a multiple of 256. It will not affect the image data.
    ///
    /// [`Queue::write_texture`] does not have this requirement.
    ///
    /// Must be a multiple of the texture block size. For non-compressed textures, this is 1.
    pub bytes_per_row: Option<NonZeroU32>,
    /// "Rows" that make up a single "image".
    ///
    /// A row is one row of pixels or of compressed blocks in the x direction.
    ///
    /// An image is one layer in the z direction of a 3D image or 2DArray texture.
    ///
    /// The amount of rows per image may be larger than the actual amount of rows of data.
    ///
    /// Required if there are multiple images (i.e. the depth is more than one).
    pub rows_per_image: Option<NonZeroU32>,
}

/// Specific type of a buffer binding.
///
/// WebGPU spec: <https://gpuweb.github.io/gpuweb/#enumdef-gpubufferbindingtype>
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
/// WebGPU spec: <https://gpuweb.github.io/gpuweb/#enumdef-gputexturesampletype>
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureSampleType {
    /// Sampling returns floats.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    Float {
        /// If `filterable` is false, the texture can't be sampled with
        /// a filtering sampler.
        filterable: bool,
    },
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

/// Specific type of a sample in a texture binding.
///
/// WebGPU spec: <https://gpuweb.github.io/gpuweb/#enumdef-gpustoragetextureaccess>
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum StorageTextureAccess {
    /// The texture can only be written in the shader and it must be annotated with `writeonly`.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) writeonly uniform image2D myStorageImage;
    /// ```
    WriteOnly,
    /// The texture can only be read in the shader and it must be annotated with `readonly`.
    /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] must be enabled to use this access mode,
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) readonly uniform image2D myStorageImage;
    /// ```
    ReadOnly,
    /// The texture can be both read and written in the shader.
    /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] must be enabled to use this access mode.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) uniform image2D myStorageImage;
    /// ```
    ReadWrite,
}

/// Specific type of a sampler binding.
///
/// WebGPU spec: <https://gpuweb.github.io/gpuweb/#enumdef-gpusamplerbindingtype>
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum SamplerBindingType {
    /// The sampling result is produced based on more than a single color sample from a texture,
    /// e.g. when bilinear interpolation is enabled.
    Filtering,
    /// The sampling result is produced based on a single color sample from a texture.
    NonFiltering,
    /// Use as a comparison sampler instead of a normal sampler.
    /// For more info take a look at the analogous functionality in OpenGL: <https://www.khronos.org/opengl/wiki/Sampler_Object#Comparison_mode>.
    Comparison,
}

/// Specific type of a binding.
///
/// WebGPU spec: the enum of
/// - <https://gpuweb.github.io/gpuweb/#dictdef-gpubufferbindinglayout>
/// - <https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerbindinglayout>
/// - <https://gpuweb.github.io/gpuweb/#dictdef-gputexturebindinglayout>
/// - <https://gpuweb.github.io/gpuweb/#dictdef-gpustoragetexturebindinglayout>
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BindingType {
    /// A buffer binding.
    Buffer {
        /// Sub-type of the buffer binding.
        ty: BufferBindingType,
        /// Indicates that the binding has a dynamic offset.
        /// One offset must be passed to [`RenderPass::set_bind_group`] for each dynamic binding in increasing order of binding number.
        #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
        has_dynamic_offset: bool,
        /// Minimum size of the corresponding `BufferBinding` required to match this entry.
        /// When pipeline is created, the size has to cover at least the corresponding structure in the shader
        /// plus one element of the unbound array, which can only be last in the structure.
        /// If `None`, the check is performed at draw call time instead of pipeline and bind group creation.
        #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
        min_binding_size: Option<BufferSize>,
    },
    /// A sampler that can be used to sample a texture.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform sampler s;
    /// ```
    Sampler(SamplerBindingType),
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
    /// A list of valid formats can be found in the specification here: <https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.html#layout-qualifiers>
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
    /// Returns true for buffer bindings with dynamic offset enabled.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BindGroupLayoutEntry {
    /// Binding index. Must match shader index and be unique inside a BindGroupLayout. A binding
    /// of index 1, would be described as `layout(set = 0, binding = 1) uniform` in shaders.
    pub binding: u32,
    /// Which shader stages can see this binding.
    pub visibility: ShaderStages,
    /// The type of the binding
    pub ty: BindingType,
    /// If this value is Some, indicates this entry is an array. Array size must be 1 or greater.
    ///
    /// If this value is Some and `ty` is `BindingType::Texture`, [`Features::TEXTURE_BINDING_ARRAY`] must be supported.
    ///
    /// If this value is Some and `ty` is any other variant, bind group creation will fail.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
    pub count: Option<NonZeroU32>,
}

/// View of a buffer which can be used to copy to/from a texture.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ImageCopyBuffer<B> {
    /// The buffer to be copied to/from.
    pub buffer: B,
    /// The layout of the texture data in this buffer.
    pub layout: ImageDataLayout,
}

/// View of a texture which can be used to copy to/from a buffer/texture.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ImageCopyTexture<T> {
    /// The texture to be copied to/from.
    pub texture: T,
    /// The target mip level of the texture.
    pub mip_level: u32,
    /// The base texel of the texture in the selected `mip_level`.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
    pub origin: Origin3d,
    /// The copy aspect.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
    pub aspect: TextureAspect,
}

/// Subresource range within an image
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ImageSubresourceRange {
    /// Aspect of the texture. Color textures must be [`TextureAspect::All`](wgt::TextureAspect::All).
    pub aspect: TextureAspect,
    /// Base mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    /// If `Some(count)`, `base_mip_level + count` must be less or equal to underlying texture mip count.
    /// If `None`, considered to include the rest of the mipmap levels, but at least 1 in total.
    pub mip_level_count: Option<NonZeroU32>,
    /// Base array layer.
    pub base_array_layer: u32,
    /// Layer count.
    /// If `Some(count)`, `base_array_layer + count` must be less or equal to the underlying array count.
    /// If `None`, considered to include the rest of the array layers, but at least 1 in total.
    pub array_layer_count: Option<NonZeroU32>,
}

impl ImageSubresourceRange {
    /// Returns the mip level range of a subresource range describes for a specific texture.
    pub fn mip_range<L>(&self, texture_desc: &TextureDescriptor<L>) -> Range<u32> {
        self.base_mip_level..match self.mip_level_count {
            Some(mip_level_count) => self.base_mip_level + mip_level_count.get(),
            None => texture_desc.mip_level_count,
        }
    }

    /// Returns the layer range of a subresource range describes for a specific texture.
    pub fn layer_range<L>(&self, texture_desc: &TextureDescriptor<L>) -> Range<u32> {
        self.base_array_layer..match self.array_layer_count {
            Some(array_layer_count) => self.base_array_layer + array_layer_count.get(),
            None => {
                if texture_desc.dimension == TextureDimension::D3 {
                    self.base_array_layer + 1
                } else {
                    texture_desc.size.depth_or_array_layers
                }
            }
        }
    }
}

/// Color variation to use when sampler addressing mode is [`AddressMode::ClampToBorder`]
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum SamplerBorderColor {
    /// [0, 0, 0, 0]
    TransparentBlack,
    /// [0, 0, 0, 1]
    OpaqueBlack,
    /// [1, 1, 1, 1]
    OpaqueWhite,
}

/// Describes how to create a QuerySet.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct QuerySetDescriptor<L> {
    /// Debug label for the query set.
    pub label: L,
    /// Kind of query that this query set should contain.
    pub ty: QueryType,
    /// Total count of queries the set contains. Must not be zero.
    /// Must not be greater than [`QUERY_SET_MAX_QUERIES`].
    pub count: u32,
}

impl<L> QuerySetDescriptor<L> {
    ///
    pub fn map_label<'a, K>(&'a self, fun: impl FnOnce(&'a L) -> K) -> QuerySetDescriptor<K> {
        QuerySetDescriptor {
            label: fun(&self.label),
            ty: self.ty,
            count: self.count,
        }
    }
}

/// Type of query contained in a QuerySet.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum QueryType {
    /// Query returns a single 64-bit number, serving as an occlusion boolean.
    Occlusion,
    /// Query returns up to 5 64-bit numbers based on the given flags.
    ///
    /// See [`PipelineStatisticsTypes`]'s documentation for more information
    /// on how they get resolved.
    ///
    /// [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled to use this query type.
    PipelineStatistics(PipelineStatisticsTypes),
    /// Query returns a 64-bit number indicating the GPU-timestamp
    /// where all previous commands have finished executing.
    ///
    /// Must be multiplied by [`Device::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    ///
    /// [`Features::TIMESTAMP_QUERY`] must be enabled to use this query type.
    Timestamp,
}

bitflags::bitflags! {
    /// Flags for which pipeline data should be recorded.
    ///
    /// The amount of values written when resolved depends
    /// on the amount of flags. If 3 flags are enabled, 3
    /// 64-bit values will be written per-query.
    ///
    /// The order they are written is the order they are declared
    /// in this bitflags. If you enabled `CLIPPER_PRIMITIVES_OUT`
    /// and `COMPUTE_SHADER_INVOCATIONS`, it would write 16 bytes,
    /// the first 8 bytes being the primitive out value, the last 8
    /// bytes being the compute shader invocation count.
    #[repr(transparent)]
    pub struct PipelineStatisticsTypes : u8 {
        /// Amount of times the vertex shader is ran. Accounts for
        /// the vertex cache when doing indexed rendering.
        const VERTEX_SHADER_INVOCATIONS = 1 << 0;
        /// Amount of times the clipper is invoked. This
        /// is also the amount of triangles output by the vertex shader.
        const CLIPPER_INVOCATIONS = 1 << 1;
        /// Amount of primitives that are not culled by the clipper.
        /// This is the amount of triangles that are actually on screen
        /// and will be rasterized and rendered.
        const CLIPPER_PRIMITIVES_OUT = 1 << 2;
        /// Amount of times the fragment shader is ran. Accounts for
        /// fragment shaders running in 2x2 blocks in order to get
        /// derivatives.
        const FRAGMENT_SHADER_INVOCATIONS = 1 << 3;
        /// Amount of times a compute shader is invoked. This will
        /// be equivalent to the dispatch count times the workgroup size.
        const COMPUTE_SHADER_INVOCATIONS = 1 << 4;
    }
}

#[cfg(feature = "bitflags_serde_shim")]
bitflags_serde_shim::impl_serde_for_bitflags!(PipelineStatisticsTypes);

/// Argument buffer layout for draw_indirect commands.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawIndirectArgs {
    /// The number of vertices to draw.
    pub vertex_count: u32,
    /// The number of instances to draw.
    pub instance_count: u32,
    /// Offset into the vertex buffers, in vertices, to begin drawing from.
    pub first_vertex: u32,
    /// First instance to draw.
    pub first_instance: u32,
}

/// Argument buffer layout for draw_indexed_indirect commands.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawIndexedIndirectArgs {
    /// The number of indices to draw.
    pub index_count: u32,
    /// The number of instances to draw.
    pub instance_count: u32,
    /// Offset into the index buffer, in indices, begin drawing from.
    pub first_index: u32,
    /// Added to each index value before indexing into the vertex buffers.
    pub base_vertex: i32,
    /// First instance to draw.
    pub first_instance: u32,
}

/// Argument buffer layout for dispatch_indirect commands.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DispatchIndirectArgs {
    /// X dimension of the grid of workgroups to dispatch.
    pub group_size_x: u32,
    /// Y dimension of the grid of workgroups to dispatch.
    pub group_size_y: u32,
    /// Z dimension of the grid of workgroups to dispatch.
    pub group_size_z: u32,
}

/// Describes how shader bound checks should be performed.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ShaderBoundChecks {
    runtime_checks: bool,
}

impl ShaderBoundChecks {
    /// Creates a new configuration where the shader is bound checked.
    pub fn new() -> Self {
        ShaderBoundChecks {
            runtime_checks: true,
        }
    }

    /// Creates a new configuration where the shader isn't bound checked.
    ///
    /// # Safety
    /// The caller MUST ensure that all shaders built with this configuration don't perform any
    /// out of bounds reads or writes.
    pub unsafe fn unchecked() -> Self {
        ShaderBoundChecks {
            runtime_checks: false,
        }
    }

    /// Query whether runtime bound checks are enabled in this configuration
    pub fn runtime_checks(&self) -> bool {
        self.runtime_checks
    }
}

impl Default for ShaderBoundChecks {
    fn default() -> Self {
        Self::new()
    }
}
