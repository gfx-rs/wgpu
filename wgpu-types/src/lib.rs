//! This library describes the API surface of WebGPU that is agnostic of the backend.
//! This API is used for targeting both Web and Native.

#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![allow(
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
)]
#![warn(clippy::ptr_as_ptr, missing_docs, unsafe_op_in_unsafe_fn)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use alloc::borrow::Cow;
use alloc::{string::String, vec, vec::Vec};
use core::cmp::Ordering;
use core::{
    fmt,
    hash::{Hash, Hasher},
    mem,
    num::NonZeroU32,
    ops::Range,
};

use bytemuck::{Pod, Zeroable};

#[cfg(any(feature = "serde", test))]
use {
    alloc::format,
    serde::{Deserialize, Serialize},
};

pub mod assertions;
mod cast_utils;
mod counters;
mod env;
pub mod error;
mod features;
pub mod instance;
pub mod math;
mod tokens;
mod transfers;

pub use counters::*;
pub use features::*;
pub use instance::*;
pub use tokens::*;
pub use transfers::*;

/// Integral type used for [`Buffer`] offsets and sizes.
///
/// [`Buffer`]: ../wgpu/struct.Buffer.html
pub type BufferAddress = u64;

/// Integral type used for [`BufferSlice`] sizes.
///
/// Note that while this type is non-zero, a [`Buffer`] *per se* can have a size of zero,
/// but no slice or mapping can be created from it.
///
/// [`Buffer`]: ../wgpu/struct.Buffer.html
/// [`BufferSlice`]: ../wgpu/struct.BufferSlice.html
pub type BufferSize = core::num::NonZeroU64;

/// Integral type used for binding locations in shaders.
///
/// Used in [`VertexAttribute`]s and errors.
///
/// [`VertexAttribute`]: ../wgpu/struct.VertexAttribute.html
pub type ShaderLocation = u32;

/// Integral type used for
/// [dynamic bind group offsets](../wgpu/struct.RenderPass.html#method.set_bind_group).
pub type DynamicOffset = u32;

/// Buffer-texture copies must have [`bytes_per_row`] aligned to this number.
///
/// This doesn't apply to [`Queue::write_texture`][Qwt], only to [`copy_buffer_to_texture()`]
/// and [`copy_texture_to_buffer()`].
///
/// [`bytes_per_row`]: TexelCopyBufferLayout::bytes_per_row
/// [`copy_buffer_to_texture()`]: ../wgpu/struct.Queue.html#method.copy_buffer_to_texture
/// [`copy_texture_to_buffer()`]: ../wgpu/struct.Queue.html#method.copy_texture_to_buffer
/// [Qwt]: ../wgpu/struct.Queue.html#method.write_texture
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// An [offset into the query resolve buffer] has to be aligned to this.
///
/// [offset into the query resolve buffer]: ../wgpu/struct.CommandEncoder.html#method.resolve_query_set
pub const QUERY_RESOLVE_BUFFER_ALIGNMENT: BufferAddress = 256;

/// Buffer to buffer copy as well as buffer clear offsets and sizes must be aligned to this number.
pub const COPY_BUFFER_ALIGNMENT: BufferAddress = 4;

/// Minimum alignment of buffer mappings.
///
/// The range passed to [`map_async()`] or [`get_mapped_range()`] must be at least this aligned.
///
/// [`map_async()`]: ../wgpu/struct.Buffer.html#method.map_async
/// [`get_mapped_range()`]: ../wgpu/struct.Buffer.html#method.get_mapped_range
pub const MAP_ALIGNMENT: BufferAddress = 8;

/// [Vertex buffer offsets] and [strides] have to be a multiple of this number.
///
/// [Vertex buffer offsets]: ../wgpu/util/trait.RenderEncoder.html#tymethod.set_vertex_buffer
/// [strides]: ../wgpu/struct.VertexBufferLayout.html#structfield.array_stride
pub const VERTEX_ALIGNMENT: BufferAddress = 4;

/// [Vertex buffer strides] have to be a multiple of this number.
///
/// [Vertex buffer strides]: ../wgpu/struct.VertexBufferLayout.html#structfield.array_stride
#[deprecated(note = "Use `VERTEX_ALIGNMENT` instead", since = "27.0.0")]
pub const VERTEX_STRIDE_ALIGNMENT: BufferAddress = 4;

/// Ranges of [writes to push constant storage] must be at least this aligned.
///
/// [writes to push constant storage]: ../wgpu/struct.RenderPass.html#method.set_push_constants
pub const PUSH_CONSTANT_ALIGNMENT: u32 = 4;

/// Maximum queries in a [`QuerySetDescriptor`].
pub const QUERY_SET_MAX_QUERIES: u32 = 4096;

/// Size in bytes of a single piece of [query] data.
///
/// [query]: ../wgpu/struct.QuerySet.html
pub const QUERY_SIZE: u32 = 8;

/// Backends supported by wgpu.
///
/// See also [`Backends`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Backend {
    /// Dummy backend, which may be used for testing.
    ///
    /// It performs no rendering or computation, but allows creation of stub GPU resource types,
    /// so that code which manages GPU resources can be tested without an available GPU.
    /// Specifically, the following operations are implemented:
    ///
    /// * Enumerating adapters will always return one noop adapter, which can be used to create
    ///   devices.
    /// * Buffers may be created, written, mapped, and copied to other buffers.
    /// * Command encoders may be created, but only buffer operations are useful.
    ///
    /// Other resources can be created but are nonfunctional; notably,
    ///
    /// * Render passes and compute passes are not executed.
    /// * Textures may be created, but do not store any texels.
    /// * There are no compatible surfaces.
    ///
    /// An adapter using the noop backend can only be obtained if [`NoopBackendOptions`]
    /// enables it, in addition to the ordinary requirement of [`Backends::NOOP`] being set.
    /// This ensures that applications not desiring a non-functional backend will not receive it.
    Noop = 0,
    /// Vulkan API (Windows, Linux, Android, MacOS via `vulkan-portability`/MoltenVK)
    Vulkan = 1,
    /// Metal API (Apple platforms)
    Metal = 2,
    /// Direct3D-12 (Windows)
    Dx12 = 3,
    /// OpenGL 3.3+ (Windows), OpenGL ES 3.0+ (Linux, Android, MacOS via Angle), and WebGL2
    Gl = 4,
    /// WebGPU in the browser
    BrowserWebGpu = 5,
}

impl Backend {
    /// Array of all [`Backend`] values, corresponding to [`Backends::all()`].
    pub const ALL: [Backend; Backends::all().bits().count_ones() as usize] = [
        Self::Noop,
        Self::Vulkan,
        Self::Metal,
        Self::Dx12,
        Self::Gl,
        Self::BrowserWebGpu,
    ];

    /// Returns the string name of the backend.
    #[must_use]
    pub const fn to_str(self) -> &'static str {
        match self {
            Backend::Noop => "noop",
            Backend::Vulkan => "vulkan",
            Backend::Metal => "metal",
            Backend::Dx12 => "dx12",
            Backend::Gl => "gl",
            Backend::BrowserWebGpu => "webgpu",
        }
    }
}

impl core::fmt::Display for Backend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.to_str())
    }
}

/// Power Preference when choosing a physical adapter.
///
/// Corresponds to [WebGPU `GPUPowerPreference`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpupowerpreference).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum PowerPreference {
    #[default]
    /// Power usage is not considered when choosing an adapter.
    None = 0,
    /// Adapter that uses the least possible power. This is often an integrated GPU.
    LowPower = 1,
    /// Adapter that has the highest performance. This is often a discrete GPU.
    HighPerformance = 2,
}

impl PowerPreference {
    /// Get a power preference from the environment variable `WGPU_POWER_PREF`.
    pub fn from_env() -> Option<Self> {
        let env = crate::env::var("WGPU_POWER_PREF")?;
        match env.to_lowercase().as_str() {
            "low" => Some(Self::LowPower),
            "high" => Some(Self::HighPerformance),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

bitflags::bitflags! {
    /// Represents the backends that wgpu will use.
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct Backends: u32 {
        /// [`Backend::Noop`].
        const NOOP = 1 << Backend::Noop as u32;

        /// [`Backend::Vulkan`].
        /// Supported on Windows, Linux/Android, and macOS/iOS via Vulkan Portability (with the Vulkan feature enabled)
        const VULKAN = 1 << Backend::Vulkan as u32;

        /// [`Backend::Gl`].
        /// Supported on Linux/Android, the web through webassembly via WebGL, and Windows and
        /// macOS/iOS via ANGLE
        const GL = 1 << Backend::Gl as u32;

        /// [`Backend::Metal`].
        /// Supported on macOS and iOS.
        const METAL = 1 << Backend::Metal as u32;

        /// [`Backend::Dx12`].
        /// Supported on Windows 10 and later
        const DX12 = 1 << Backend::Dx12 as u32;

        /// [`Backend::BrowserWebGpu`].
        /// Supported when targeting the web through WebAssembly with the `webgpu` feature enabled.
        ///
        /// The WebGPU backend is special in several ways:
        /// It is not not implemented by `wgpu_core` and instead by the higher level `wgpu` crate.
        /// Whether WebGPU is targeted is decided upon the creation of the `wgpu::Instance`,
        /// *not* upon adapter creation. See `wgpu::Instance::new`.
        const BROWSER_WEBGPU = 1 << Backend::BrowserWebGpu as u32;

        /// All the apis that wgpu offers first tier of support for.
        ///
        /// * [`Backends::VULKAN`]
        /// * [`Backends::METAL`]
        /// * [`Backends::DX12`]
        /// * [`Backends::BROWSER_WEBGPU`]
        const PRIMARY = Self::VULKAN.bits()
            | Self::METAL.bits()
            | Self::DX12.bits()
            | Self::BROWSER_WEBGPU.bits();

        /// All the apis that wgpu offers second tier of support for. These may
        /// be unsupported/still experimental.
        ///
        /// * [`Backends::GL`]
        const SECONDARY = Self::GL.bits();
    }
}

impl Default for Backends {
    fn default() -> Self {
        Self::all()
    }
}

impl From<Backend> for Backends {
    fn from(backend: Backend) -> Self {
        Self::from_bits(1 << backend as u32).unwrap()
    }
}

impl Backends {
    /// Gets a set of backends from the environment variable `WGPU_BACKEND`.
    ///
    /// See [`Self::from_comma_list()`] for the format of the string.
    pub fn from_env() -> Option<Self> {
        let env = crate::env::var("WGPU_BACKEND")?;
        Some(Self::from_comma_list(&env))
    }

    /// Takes the given options, modifies them based on the `WGPU_BACKEND` environment variable, and returns the result.
    pub fn with_env(&self) -> Self {
        if let Some(env) = Self::from_env() {
            env
        } else {
            *self
        }
    }

    /// Generates a set of backends from a comma separated list of case-insensitive backend names.
    ///
    /// Whitespace is stripped, so both 'gl, dx12' and 'gl,dx12' are valid.
    ///
    /// Always returns WEBGPU on wasm over webgpu.
    ///
    /// Names:
    /// - vulkan = "vulkan" or "vk"
    /// - dx12   = "dx12" or "d3d12"
    /// - metal  = "metal" or "mtl"
    /// - gles   = "opengl" or "gles" or "gl"
    /// - webgpu = "webgpu"
    pub fn from_comma_list(string: &str) -> Self {
        let mut backends = Self::empty();
        for backend in string.to_lowercase().split(',') {
            backends |= match backend.trim() {
                "vulkan" | "vk" => Self::VULKAN,
                "dx12" | "d3d12" => Self::DX12,
                "metal" | "mtl" => Self::METAL,
                "opengl" | "gles" | "gl" => Self::GL,
                "webgpu" => Self::BROWSER_WEBGPU,
                "noop" => Self::NOOP,
                b => {
                    log::warn!("unknown backend string '{b}'");
                    continue;
                }
            }
        }

        if backends.is_empty() {
            log::warn!("no valid backend strings found!");
        }

        backends
    }
}

/// Options for requesting adapter.
///
/// Corresponds to [WebGPU `GPURequestAdapterOptions`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions).
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RequestAdapterOptions<S> {
    /// Power preference for the adapter.
    pub power_preference: PowerPreference,
    /// Indicates that only a fallback adapter can be returned. This is generally a "software"
    /// implementation on the system.
    pub force_fallback_adapter: bool,
    /// Surface that is required to be presentable with the requested adapter. This does not
    /// create the surface, only guarantees that the adapter can present to said surface.
    /// For WebGL, this is strictly required, as an adapter can not be created without a surface.
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

/// Error when [`Instance::request_adapter()`] fails.
///
/// This type is not part of the WebGPU standard, where `requestAdapter()` would simply return null.
///
/// [`Instance::request_adapter()`]: ../wgpu/struct.Instance.html#method.request_adapter
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum RequestAdapterError {
    /// No adapter available via the instance’s backends matched the request’s adapter criteria.
    NotFound {
        // These fields must be set by wgpu-core and wgpu, but are not intended to be stable API,
        // only data for the production of the error message.
        #[doc(hidden)]
        active_backends: Backends,
        #[doc(hidden)]
        requested_backends: Backends,
        #[doc(hidden)]
        supported_backends: Backends,
        #[doc(hidden)]
        no_fallback_backends: Backends,
        #[doc(hidden)]
        no_adapter_backends: Backends,
        #[doc(hidden)]
        incompatible_surface_backends: Backends,
    },

    /// Attempted to obtain adapter specified by environment variable, but the environment variable
    /// was not set.
    EnvNotSet,
}

impl core::error::Error for RequestAdapterError {}
impl fmt::Display for RequestAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestAdapterError::NotFound {
                active_backends,
                requested_backends,
                supported_backends,
                no_fallback_backends,
                no_adapter_backends,
                incompatible_surface_backends,
            } => {
                write!(f, "No suitable graphics adapter found; ")?;
                let mut first = true;
                for backend in Backend::ALL {
                    let bit = Backends::from(backend);
                    let comma = if mem::take(&mut first) { "" } else { ", " };
                    let explanation = if !requested_backends.contains(bit) {
                        // We prefer reporting this, because it makes the error most stable with
                        // respect to what is directly controllable by the caller, as opposed to
                        // compilation options or the run-time environment.
                        "not requested"
                    } else if !supported_backends.contains(bit) {
                        "support not compiled in"
                    } else if no_adapter_backends.contains(bit) {
                        "found no adapters"
                    } else if incompatible_surface_backends.contains(bit) {
                        "not compatible with provided surface"
                    } else if no_fallback_backends.contains(bit) {
                        "had no fallback adapters"
                    } else if !active_backends.contains(bit) {
                        // Backend requested but not active in this instance
                        if backend == Backend::Noop {
                            "not explicitly enabled"
                        } else {
                            "drivers/libraries could not be loaded"
                        }
                    } else {
                        // This path should be unreachable, but don't crash.
                        "[unknown reason]"
                    };
                    write!(f, "{comma}{backend} {explanation}")?;
                }
            }
            RequestAdapterError::EnvNotSet => f.write_str("WGPU_ADAPTER_NAME not set")?,
        }
        Ok(())
    }
}

/// Invoke a macro for each of the limits.
///
/// The supplied macro should take two arguments. The first is a limit name, as
/// an identifier, typically used to access a member of `struct Limits`. The
/// second is `Ordering::Less` if valid values are less than the limit (the
/// common case), or `Ordering::Greater` if valid values are more than the limit
/// (for limits like alignments, which are minima instead of maxima).
macro_rules! with_limits {
    ($macro_name:ident) => {
        $macro_name!(max_texture_dimension_1d, Ordering::Less);
        $macro_name!(max_texture_dimension_1d, Ordering::Less);
        $macro_name!(max_texture_dimension_2d, Ordering::Less);
        $macro_name!(max_texture_dimension_3d, Ordering::Less);
        $macro_name!(max_texture_array_layers, Ordering::Less);
        $macro_name!(max_bind_groups, Ordering::Less);
        $macro_name!(max_bindings_per_bind_group, Ordering::Less);
        $macro_name!(
            max_dynamic_uniform_buffers_per_pipeline_layout,
            Ordering::Less
        );
        $macro_name!(
            max_dynamic_storage_buffers_per_pipeline_layout,
            Ordering::Less
        );
        $macro_name!(max_sampled_textures_per_shader_stage, Ordering::Less);
        $macro_name!(max_samplers_per_shader_stage, Ordering::Less);
        $macro_name!(max_storage_buffers_per_shader_stage, Ordering::Less);
        $macro_name!(max_storage_textures_per_shader_stage, Ordering::Less);
        $macro_name!(max_uniform_buffers_per_shader_stage, Ordering::Less);
        $macro_name!(max_binding_array_elements_per_shader_stage, Ordering::Less);
        $macro_name!(max_uniform_buffer_binding_size, Ordering::Less);
        $macro_name!(max_storage_buffer_binding_size, Ordering::Less);
        $macro_name!(max_vertex_buffers, Ordering::Less);
        $macro_name!(max_buffer_size, Ordering::Less);
        $macro_name!(max_vertex_attributes, Ordering::Less);
        $macro_name!(max_vertex_buffer_array_stride, Ordering::Less);
        $macro_name!(min_uniform_buffer_offset_alignment, Ordering::Greater);
        $macro_name!(min_storage_buffer_offset_alignment, Ordering::Greater);
        $macro_name!(max_inter_stage_shader_components, Ordering::Less);
        $macro_name!(max_color_attachments, Ordering::Less);
        $macro_name!(max_color_attachment_bytes_per_sample, Ordering::Less);
        $macro_name!(max_compute_workgroup_storage_size, Ordering::Less);
        $macro_name!(max_compute_invocations_per_workgroup, Ordering::Less);
        $macro_name!(max_compute_workgroup_size_x, Ordering::Less);
        $macro_name!(max_compute_workgroup_size_y, Ordering::Less);
        $macro_name!(max_compute_workgroup_size_z, Ordering::Less);
        $macro_name!(max_compute_workgroups_per_dimension, Ordering::Less);

        $macro_name!(min_subgroup_size, Ordering::Greater);
        $macro_name!(max_subgroup_size, Ordering::Less);

        $macro_name!(max_push_constant_size, Ordering::Less);
        $macro_name!(max_non_sampler_bindings, Ordering::Less);

        $macro_name!(max_task_workgroup_total_count, Ordering::Less);
        $macro_name!(max_task_workgroups_per_dimension, Ordering::Less);
        $macro_name!(max_mesh_multiview_count, Ordering::Less);
        $macro_name!(max_mesh_output_layers, Ordering::Less);

        $macro_name!(max_blas_primitive_count, Ordering::Less);
        $macro_name!(max_blas_geometry_count, Ordering::Less);
        $macro_name!(max_tlas_instance_count, Ordering::Less);
    };
}

/// Represents the sets of limits an adapter/device supports.
///
/// We provide three different defaults.
/// - [`Limits::downlevel_defaults()`]. This is a set of limits that is guaranteed to work on almost
///   all backends, including "downlevel" backends such as OpenGL and D3D11, other than WebGL. For
///   most applications we recommend using these limits, assuming they are high enough for your
///   application, and you do not intent to support WebGL.
/// - [`Limits::downlevel_webgl2_defaults()`] This is a set of limits that is lower even than the
///   [`downlevel_defaults()`], configured to be low enough to support running in the browser using
///   WebGL2.
/// - [`Limits::default()`]. This is the set of limits that is guaranteed to work on all modern
///   backends and is guaranteed to be supported by WebGPU. Applications needing more modern
///   features can use this as a reasonable set of limits if they are targeting only desktop and
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
/// Corresponds to [WebGPU `GPUSupportedLimits`](
/// https://gpuweb.github.io/gpuweb/#gpusupportedlimits).
///
/// [`downlevel_defaults()`]: Limits::downlevel_defaults
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase", default))]
pub struct Limits {
    /// Maximum allowed value for the `size.width` of a texture created with `TextureDimension::D1`.
    /// Defaults to 8192. Higher is "better".
    #[cfg_attr(feature = "serde", serde(rename = "maxTextureDimension1D"))]
    pub max_texture_dimension_1d: u32,
    /// Maximum allowed value for the `size.width` and `size.height` of a texture created with `TextureDimension::D2`.
    /// Defaults to 8192. Higher is "better".
    #[cfg_attr(feature = "serde", serde(rename = "maxTextureDimension2D"))]
    pub max_texture_dimension_2d: u32,
    /// Maximum allowed value for the `size.width`, `size.height`, and `size.depth_or_array_layers`
    /// of a texture created with `TextureDimension::D3`.
    /// Defaults to 2048. Higher is "better".
    #[cfg_attr(feature = "serde", serde(rename = "maxTextureDimension3D"))]
    pub max_texture_dimension_3d: u32,
    /// Maximum allowed value for the `size.depth_or_array_layers` of a texture created with `TextureDimension::D2`.
    /// Defaults to 256. Higher is "better".
    pub max_texture_array_layers: u32,
    /// Amount of bind groups that can be attached to a pipeline at the same time. Defaults to 4. Higher is "better".
    pub max_bind_groups: u32,
    /// Maximum binding index allowed in `create_bind_group_layout`. Defaults to 1000. Higher is "better".
    pub max_bindings_per_bind_group: u32,
    /// Amount of uniform buffer bindings that can be dynamic in a single pipeline. Defaults to 8. Higher is "better".
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    /// Amount of storage buffer bindings that can be dynamic in a single pipeline. Defaults to 4. Higher is "better".
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    /// Amount of sampled textures visible in a single shader stage. Defaults to 16. Higher is "better".
    pub max_sampled_textures_per_shader_stage: u32,
    /// Amount of samplers visible in a single shader stage. Defaults to 16. Higher is "better".
    pub max_samplers_per_shader_stage: u32,
    /// Amount of storage buffers visible in a single shader stage. Defaults to 8. Higher is "better".
    pub max_storage_buffers_per_shader_stage: u32,
    /// Amount of storage textures visible in a single shader stage. Defaults to 4. Higher is "better".
    pub max_storage_textures_per_shader_stage: u32,
    /// Amount of uniform buffers visible in a single shader stage. Defaults to 12. Higher is "better".
    pub max_uniform_buffers_per_shader_stage: u32,
    /// Amount of individual resources within binding arrays that can be accessed in a single shader stage. Applies
    /// to all types of bindings except samplers.
    ///
    /// This "defaults" to 0. However if binding arrays are supported, all devices can support 500,000. Higher is "better".
    pub max_binding_array_elements_per_shader_stage: u32,
    /// Amount of individual samplers within binding arrays that can be accessed in a single shader stage.
    ///
    /// This "defaults" to 0. However if binding arrays are supported, all devices can support 1,000. Higher is "better".
    pub max_binding_array_sampler_elements_per_shader_stage: u32,
    /// Maximum size in bytes of a binding to a uniform buffer. Defaults to 64 KiB. Higher is "better".
    pub max_uniform_buffer_binding_size: u32,
    /// Maximum size in bytes of a binding to a storage buffer. Defaults to 128 MiB. Higher is "better".
    pub max_storage_buffer_binding_size: u32,
    /// Maximum length of `VertexState::buffers` when creating a `RenderPipeline`.
    /// Defaults to 8. Higher is "better".
    pub max_vertex_buffers: u32,
    /// A limit above which buffer allocations are guaranteed to fail.
    /// Defaults to 256 MiB. Higher is "better".
    ///
    /// Buffer allocations below the maximum buffer size may not succeed depending on available memory,
    /// fragmentation and other factors.
    pub max_buffer_size: u64,
    /// Maximum length of `VertexBufferLayout::attributes`, summed over all `VertexState::buffers`,
    /// when creating a `RenderPipeline`.
    /// Defaults to 16. Higher is "better".
    pub max_vertex_attributes: u32,
    /// Maximum value for `VertexBufferLayout::array_stride` when creating a `RenderPipeline`.
    /// Defaults to 2048. Higher is "better".
    pub max_vertex_buffer_array_stride: u32,
    /// Required `BufferBindingType::Uniform` alignment for `BufferBinding::offset`
    /// when creating a `BindGroup`, or for `set_bind_group` `dynamicOffsets`.
    /// Defaults to 256. Lower is "better".
    pub min_uniform_buffer_offset_alignment: u32,
    /// Required `BufferBindingType::Storage` alignment for `BufferBinding::offset`
    /// when creating a `BindGroup`, or for `set_bind_group` `dynamicOffsets`.
    /// Defaults to 256. Lower is "better".
    pub min_storage_buffer_offset_alignment: u32,
    /// Maximum allowed number of components (scalars) of input or output locations for
    /// inter-stage communication (vertex outputs to fragment inputs). Defaults to 60.
    /// Higher is "better".
    pub max_inter_stage_shader_components: u32,
    /// The maximum allowed number of color attachments.
    pub max_color_attachments: u32,
    /// The maximum number of bytes necessary to hold one sample (pixel or subpixel) of render
    /// pipeline output data, across all color attachments as described by [`TextureFormat::target_pixel_byte_cost`]
    /// and [`TextureFormat::target_component_alignment`]. Defaults to 32. Higher is "better".
    ///
    /// ⚠️ `Rgba8Unorm`/`Rgba8Snorm`/`Bgra8Unorm`/`Bgra8Snorm` are deceptively 8 bytes per sample. ⚠️
    pub max_color_attachment_bytes_per_sample: u32,
    /// Maximum number of bytes used for workgroup memory in a compute entry point. Defaults to
    /// 16384. Higher is "better".
    pub max_compute_workgroup_storage_size: u32,
    /// Maximum value of the product of the `workgroup_size` dimensions for a compute entry-point.
    /// Defaults to 256. Higher is "better".
    pub max_compute_invocations_per_workgroup: u32,
    /// The maximum value of the `workgroup_size` X dimension for a compute stage `ShaderModule` entry-point.
    /// Defaults to 256. Higher is "better".
    pub max_compute_workgroup_size_x: u32,
    /// The maximum value of the `workgroup_size` Y dimension for a compute stage `ShaderModule` entry-point.
    /// Defaults to 256. Higher is "better".
    pub max_compute_workgroup_size_y: u32,
    /// The maximum value of the `workgroup_size` Z dimension for a compute stage `ShaderModule` entry-point.
    /// Defaults to 64. Higher is "better".
    pub max_compute_workgroup_size_z: u32,
    /// The maximum value for each dimension of a `ComputePass::dispatch(x, y, z)` operation.
    /// Defaults to 65535. Higher is "better".
    pub max_compute_workgroups_per_dimension: u32,

    /// Minimal number of invocations in a subgroup. Higher is "better".
    pub min_subgroup_size: u32,
    /// Maximal number of invocations in a subgroup. Lower is "better".
    pub max_subgroup_size: u32,
    /// Amount of storage available for push constants in bytes. Defaults to 0. Higher is "better".
    /// Requesting more than 0 during device creation requires [`Features::PUSH_CONSTANTS`] to be enabled.
    ///
    /// Expect the size to be:
    /// - Vulkan: 128-256 bytes
    /// - DX12: 256 bytes
    /// - Metal: 4096 bytes
    /// - OpenGL doesn't natively support push constants, and are emulated with uniforms,
    ///   so this number is less useful but likely 256.
    pub max_push_constant_size: u32,
    /// Maximum number of live non-sampler bindings.
    ///
    /// This limit only affects the d3d12 backend. Using a large number will allow the device
    /// to create many bind groups at the cost of a large up-front allocation at device creation.
    pub max_non_sampler_bindings: u32,

    /// The maximum total value of x*y*z for a given `draw_mesh_tasks` command
    pub max_task_workgroup_total_count: u32,
    /// The maximum value for each dimension of a `RenderPass::draw_mesh_tasks(x, y, z)` operation.
    /// Defaults to 65535. Higher is "better".
    pub max_task_workgroups_per_dimension: u32,
    /// The maximum number of layers that can be output from a mesh shader
    pub max_mesh_output_layers: u32,
    /// The maximum number of views that can be used by a mesh shader
    pub max_mesh_multiview_count: u32,

    /// The maximum number of primitive (ex: triangles, aabbs) a BLAS is allowed to have. Requesting
    /// more than 0 during device creation only makes sense if [`Features::EXPERIMENTAL_RAY_QUERY`]
    /// is enabled.
    pub max_blas_primitive_count: u32,
    /// The maximum number of geometry descriptors a BLAS is allowed to have. Requesting
    /// more than 0 during device creation only makes sense if [`Features::EXPERIMENTAL_RAY_QUERY`]
    /// is enabled.
    pub max_blas_geometry_count: u32,
    /// The maximum number of instances a TLAS is allowed to have. Requesting more than 0 during
    /// device creation only makes sense if [`Features::EXPERIMENTAL_RAY_QUERY`]
    /// is enabled.
    pub max_tlas_instance_count: u32,
    /// The maximum number of acceleration structures allowed to be used in a shader stage.
    /// Requesting more than 0 during device creation only makes sense if [`Features::EXPERIMENTAL_RAY_QUERY`]
    /// is enabled.
    pub max_acceleration_structures_per_shader_stage: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self::defaults()
    }
}

impl Limits {
    /// These default limits are guaranteed to to work on all modern
    /// backends and guaranteed to be supported by WebGPU
    ///
    /// Those limits are as follows:
    /// ```rust
    /// # use wgpu_types::Limits;
    /// assert_eq!(Limits::defaults(), Limits {
    ///     max_texture_dimension_1d: 8192,
    ///     max_texture_dimension_2d: 8192,
    ///     max_texture_dimension_3d: 2048,
    ///     max_texture_array_layers: 256,
    ///     max_bind_groups: 4,
    ///     max_bindings_per_bind_group: 1000,
    ///     max_dynamic_uniform_buffers_per_pipeline_layout: 8,
    ///     max_dynamic_storage_buffers_per_pipeline_layout: 4,
    ///     max_sampled_textures_per_shader_stage: 16,
    ///     max_samplers_per_shader_stage: 16,
    ///     max_storage_buffers_per_shader_stage: 8,
    ///     max_storage_textures_per_shader_stage: 4,
    ///     max_uniform_buffers_per_shader_stage: 12,
    ///     max_binding_array_elements_per_shader_stage: 0,
    ///     max_binding_array_sampler_elements_per_shader_stage: 0,
    ///     max_uniform_buffer_binding_size: 64 << 10, // (64 KiB)
    ///     max_storage_buffer_binding_size: 128 << 20, // (128 MiB)
    ///     max_vertex_buffers: 8,
    ///     max_buffer_size: 256 << 20, // (256 MiB)
    ///     max_vertex_attributes: 16,
    ///     max_vertex_buffer_array_stride: 2048,
    ///     min_uniform_buffer_offset_alignment: 256,
    ///     min_storage_buffer_offset_alignment: 256,
    ///     max_inter_stage_shader_components: 60,
    ///     max_color_attachments: 8,
    ///     max_color_attachment_bytes_per_sample: 32,
    ///     max_compute_workgroup_storage_size: 16384,
    ///     max_compute_invocations_per_workgroup: 256,
    ///     max_compute_workgroup_size_x: 256,
    ///     max_compute_workgroup_size_y: 256,
    ///     max_compute_workgroup_size_z: 64,
    ///     max_compute_workgroups_per_dimension: 65535,
    ///     min_subgroup_size: 0,
    ///     max_subgroup_size: 0,
    ///     max_push_constant_size: 0,
    ///     max_non_sampler_bindings: 1_000_000,
    ///     max_task_workgroup_total_count: 0,
    ///     max_task_workgroups_per_dimension: 0,
    ///     max_mesh_multiview_count: 0,
    ///     max_mesh_output_layers: 0,
    ///     max_blas_primitive_count: 0,
    ///     max_blas_geometry_count: 0,
    ///     max_tlas_instance_count: 0,
    ///     max_acceleration_structures_per_shader_stage: 0,
    /// });
    /// ```
    ///
    /// Rust doesn't allow const in trait implementations, so we break this out
    /// to allow reusing these defaults in const contexts
    #[must_use]
    pub const fn defaults() -> Self {
        Self {
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 1000,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 4,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_textures_per_shader_stage: 4,
            max_uniform_buffers_per_shader_stage: 12,
            max_binding_array_elements_per_shader_stage: 0,
            max_binding_array_sampler_elements_per_shader_stage: 0,
            max_uniform_buffer_binding_size: 64 << 10, // (64 KiB)
            max_storage_buffer_binding_size: 128 << 20, // (128 MiB)
            max_vertex_buffers: 8,
            max_buffer_size: 256 << 20, // (256 MiB)
            max_vertex_attributes: 16,
            max_vertex_buffer_array_stride: 2048,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_inter_stage_shader_components: 60,
            max_color_attachments: 8,
            max_color_attachment_bytes_per_sample: 32,
            max_compute_workgroup_storage_size: 16384,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65535,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            max_push_constant_size: 0,
            max_non_sampler_bindings: 1_000_000,

            max_task_workgroup_total_count: 0,
            max_task_workgroups_per_dimension: 0,
            max_mesh_multiview_count: 0,
            max_mesh_output_layers: 0,

            max_blas_primitive_count: 0,
            max_blas_geometry_count: 0,
            max_tlas_instance_count: 0,
            max_acceleration_structures_per_shader_stage: 0,
        }
    }

    /// These default limits are guaranteed to be compatible with GLES-3.1, and D3D11
    ///
    /// Those limits are as follows (different from default are marked with *):
    /// ```rust
    /// # use wgpu_types::Limits;
    /// assert_eq!(Limits::downlevel_defaults(), Limits {
    ///     max_texture_dimension_1d: 2048, // *
    ///     max_texture_dimension_2d: 2048, // *
    ///     max_texture_dimension_3d: 256, // *
    ///     max_texture_array_layers: 256,
    ///     max_bind_groups: 4,
    ///     max_bindings_per_bind_group: 1000,
    ///     max_dynamic_uniform_buffers_per_pipeline_layout: 8,
    ///     max_dynamic_storage_buffers_per_pipeline_layout: 4,
    ///     max_sampled_textures_per_shader_stage: 16,
    ///     max_samplers_per_shader_stage: 16,
    ///     max_storage_buffers_per_shader_stage: 4, // *
    ///     max_storage_textures_per_shader_stage: 4,
    ///     max_uniform_buffers_per_shader_stage: 12,
    ///     max_binding_array_elements_per_shader_stage: 0,
    ///     max_binding_array_sampler_elements_per_shader_stage: 0,
    ///     max_uniform_buffer_binding_size: 16 << 10, // * (16 KiB)
    ///     max_storage_buffer_binding_size: 128 << 20, // (128 MiB)
    ///     max_vertex_buffers: 8,
    ///     max_vertex_attributes: 16,
    ///     max_vertex_buffer_array_stride: 2048,
    ///     min_subgroup_size: 0,
    ///     max_subgroup_size: 0,
    ///     max_push_constant_size: 0,
    ///     min_uniform_buffer_offset_alignment: 256,
    ///     min_storage_buffer_offset_alignment: 256,
    ///     max_inter_stage_shader_components: 60,
    ///     max_color_attachments: 4,
    ///     max_color_attachment_bytes_per_sample: 32,
    ///     max_compute_workgroup_storage_size: 16352, // *
    ///     max_compute_invocations_per_workgroup: 256,
    ///     max_compute_workgroup_size_x: 256,
    ///     max_compute_workgroup_size_y: 256,
    ///     max_compute_workgroup_size_z: 64,
    ///     max_compute_workgroups_per_dimension: 65535,
    ///     max_buffer_size: 256 << 20, // (256 MiB)
    ///     max_non_sampler_bindings: 1_000_000,
    ///
    ///     max_task_workgroup_total_count: 0,
    ///     max_task_workgroups_per_dimension: 0,
    ///     max_mesh_multiview_count: 0,
    ///     max_mesh_output_layers: 0,
    ///
    ///     max_blas_primitive_count: 0,
    ///     max_blas_geometry_count: 0,
    ///     max_tlas_instance_count: 0,
    ///     max_acceleration_structures_per_shader_stage: 0,
    /// });
    /// ```
    #[must_use]
    pub const fn downlevel_defaults() -> Self {
        Self {
            max_texture_dimension_1d: 2048,
            max_texture_dimension_2d: 2048,
            max_texture_dimension_3d: 256,
            max_storage_buffers_per_shader_stage: 4,
            max_uniform_buffer_binding_size: 16 << 10, // (16 KiB)
            max_color_attachments: 4,
            // see: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf#page=7
            max_compute_workgroup_storage_size: 16352,

            max_task_workgroups_per_dimension: 0,
            max_task_workgroup_total_count: 0,
            max_mesh_multiview_count: 0,
            max_mesh_output_layers: 0,
            ..Self::defaults()
        }
    }

    /// These default limits are guaranteed to be compatible with GLES-3.0, and D3D11, and WebGL2
    ///
    /// Those limits are as follows (different from `downlevel_defaults` are marked with +,
    /// *'s from `downlevel_defaults` shown as well.):
    /// ```rust
    /// # use wgpu_types::Limits;
    /// assert_eq!(Limits::downlevel_webgl2_defaults(), Limits {
    ///     max_texture_dimension_1d: 2048, // *
    ///     max_texture_dimension_2d: 2048, // *
    ///     max_texture_dimension_3d: 256, // *
    ///     max_texture_array_layers: 256,
    ///     max_bind_groups: 4,
    ///     max_bindings_per_bind_group: 1000,
    ///     max_dynamic_uniform_buffers_per_pipeline_layout: 8,
    ///     max_dynamic_storage_buffers_per_pipeline_layout: 0, // +
    ///     max_sampled_textures_per_shader_stage: 16,
    ///     max_samplers_per_shader_stage: 16,
    ///     max_storage_buffers_per_shader_stage: 0, // * +
    ///     max_storage_textures_per_shader_stage: 0, // +
    ///     max_uniform_buffers_per_shader_stage: 11, // +
    ///     max_binding_array_elements_per_shader_stage: 0,
    ///     max_binding_array_sampler_elements_per_shader_stage: 0,
    ///     max_uniform_buffer_binding_size: 16 << 10, // * (16 KiB)
    ///     max_storage_buffer_binding_size: 0, // * +
    ///     max_vertex_buffers: 8,
    ///     max_vertex_attributes: 16,
    ///     max_vertex_buffer_array_stride: 255, // +
    ///     min_subgroup_size: 0,
    ///     max_subgroup_size: 0,
    ///     max_push_constant_size: 0,
    ///     min_uniform_buffer_offset_alignment: 256,
    ///     min_storage_buffer_offset_alignment: 256,
    ///     max_inter_stage_shader_components: 31,
    ///     max_color_attachments: 4,
    ///     max_color_attachment_bytes_per_sample: 32,
    ///     max_compute_workgroup_storage_size: 0, // +
    ///     max_compute_invocations_per_workgroup: 0, // +
    ///     max_compute_workgroup_size_x: 0, // +
    ///     max_compute_workgroup_size_y: 0, // +
    ///     max_compute_workgroup_size_z: 0, // +
    ///     max_compute_workgroups_per_dimension: 0, // +
    ///     max_buffer_size: 256 << 20, // (256 MiB),
    ///     max_non_sampler_bindings: 1_000_000,
    ///
    ///     max_task_workgroup_total_count: 0,
    ///     max_task_workgroups_per_dimension: 0,
    ///     max_mesh_multiview_count: 0,
    ///     max_mesh_output_layers: 0,
    ///
    ///     max_blas_primitive_count: 0,
    ///     max_blas_geometry_count: 0,
    ///     max_tlas_instance_count: 0,
    ///     max_acceleration_structures_per_shader_stage: 0,
    /// });
    /// ```
    #[must_use]
    pub const fn downlevel_webgl2_defaults() -> Self {
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
            min_subgroup_size: 0,
            max_subgroup_size: 0,

            // Value supported by Intel Celeron B830 on Windows (OpenGL 3.1)
            max_inter_stage_shader_components: 31,

            // Most of the values should be the same as the downlevel defaults
            ..Self::downlevel_defaults()
        }
    }

    /// Modify the current limits to use the resolution limits of the other.
    ///
    /// This is useful because the swapchain might need to be larger than any other image in the application.
    ///
    /// If your application only needs 512x512, you might be running on a 4k display and need extremely high resolution limits.
    #[must_use]
    pub const fn using_resolution(self, other: Self) -> Self {
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
    #[must_use]
    pub const fn using_alignment(self, other: Self) -> Self {
        Self {
            min_uniform_buffer_offset_alignment: other.min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment: other.min_storage_buffer_offset_alignment,
            ..self
        }
    }

    /// The minimum guaranteed limits for acceleration structures if you enable [`Features::EXPERIMENTAL_RAY_QUERY`]
    #[must_use]
    pub const fn using_minimum_supported_acceleration_structure_values(self) -> Self {
        Self {
            max_blas_geometry_count: (1 << 24) - 1, // 2^24 - 1: Vulkan's minimum
            max_tlas_instance_count: (1 << 24) - 1, // 2^24 - 1: Vulkan's minimum
            max_blas_primitive_count: 1 << 28,      // 2^28: Metal's minimum
            max_acceleration_structures_per_shader_stage: 16, // Vulkan's minimum
            ..self
        }
    }

    /// Modify the current limits to use the acceleration structure limits of `other` (`other` could
    /// be the limits of the adapter).
    #[must_use]
    pub const fn using_acceleration_structure_values(self, other: Self) -> Self {
        Self {
            max_blas_geometry_count: other.max_blas_geometry_count,
            max_tlas_instance_count: other.max_tlas_instance_count,
            max_blas_primitive_count: other.max_blas_primitive_count,
            max_acceleration_structures_per_shader_stage: other
                .max_acceleration_structures_per_shader_stage,
            ..self
        }
    }

    /// The recommended minimum limits for mesh shaders if you enable [`Features::EXPERIMENTAL_MESH_SHADER`]
    ///
    /// These are chosen somewhat arbitrarily. They are small enough that they should cover all physical devices,
    /// but not necessarily all use cases.
    #[must_use]
    pub const fn using_recommended_minimum_mesh_shader_values(self) -> Self {
        Self {
            // Literally just made this up as 256^2 or 2^16.
            // My GPU supports 2^22, and compute shaders don't have this kind of limit.
            // This very likely is never a real limiter
            max_task_workgroup_total_count: 65536,
            max_task_workgroups_per_dimension: 256,
            // llvmpipe reports 0 multiview count, which just means no multiview is allowed
            max_mesh_multiview_count: 0,
            // llvmpipe once again requires this to be 8. An RTX 3060 supports well over 1024.
            max_mesh_output_layers: 8,
            ..self
        }
    }

    /// Compares every limits within self is within the limits given in `allowed`.
    ///
    /// If you need detailed information on failures, look at [`Limits::check_limits_with_fail_fn`].
    #[must_use]
    pub fn check_limits(&self, allowed: &Self) -> bool {
        let mut within = true;
        self.check_limits_with_fail_fn(allowed, true, |_, _, _| within = false);
        within
    }

    /// Compares every limits within self is within the limits given in `allowed`.
    /// For an easy to use binary choice, use [`Limits::check_limits`].
    ///
    /// If a value is not within the allowed limit, this function calls the `fail_fn`
    /// with the:
    ///  - limit name
    ///  - self's limit
    ///  - allowed's limit.
    ///
    /// If fatal is true, a single failure bails out the comparison after a single failure.
    pub fn check_limits_with_fail_fn(
        &self,
        allowed: &Self,
        fatal: bool,
        mut fail_fn: impl FnMut(&'static str, u64, u64),
    ) {
        macro_rules! check_with_fail_fn {
            ($name:ident, $ordering:expr) => {
                let invalid_ord = $ordering.reverse();
                // In the case of `min_subgroup_size`, requesting a value of
                // zero means "I'm not going to use subgroups", so we have to
                // special case that. If any of our minimum limits could
                // meaningfully go all the way to zero, that would conflict with
                // this.
                if self.$name != 0 && self.$name.cmp(&allowed.$name) == invalid_ord {
                    fail_fn(stringify!($name), self.$name as u64, allowed.$name as u64);
                    if fatal {
                        return;
                    }
                }
            };
        }

        if self.min_subgroup_size > self.max_subgroup_size {
            fail_fn(
                "max_subgroup_size",
                self.min_subgroup_size as u64,
                allowed.min_subgroup_size as u64,
            );
        }
        with_limits!(check_with_fail_fn);
    }

    /// For each limit in `other` that is better than the value in `self`,
    /// replace the value in `self` with the value from `other`.
    ///
    /// A request for a limit value less than the WebGPU-specified default must
    /// be ignored. This function is used to clamp such requests to the default
    /// value.
    ///
    /// This function is not for clamping requests for values beyond the
    /// supported limits. For that purpose the desired function would be
    /// `or_worse_values_from` (which doesn't exist, but could be added if
    /// needed).
    #[must_use]
    pub fn or_better_values_from(mut self, other: &Self) -> Self {
        macro_rules! or_better_value_from {
            ($name:ident, $ordering:expr) => {
                match $ordering {
                    // Limits that are maximum values (most of them)
                    Ordering::Less => self.$name = self.$name.max(other.$name),
                    // Limits that are minimum values
                    Ordering::Greater => self.$name = self.$name.min(other.$name),
                    Ordering::Equal => unreachable!(),
                }
            };
        }

        with_limits!(or_better_value_from);

        self
    }
}

/// Represents the sets of additional limits on an adapter,
/// which take place when running on downlevel backends.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DownlevelLimits {}

#[allow(clippy::derivable_impls)]
impl Default for DownlevelLimits {
    fn default() -> Self {
        DownlevelLimits {}
    }
}

/// Lists various ways the underlying platform does not conform to the WebGPU standard.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
            flags: DownlevelFlags::all(),
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
    #[must_use]
    pub fn is_webgpu_compliant(&self) -> bool {
        self.flags.contains(DownlevelFlags::compliant())
            && self.limits == DownlevelLimits::default()
            && self.shader_model >= ShaderModel::Sm5
    }
}

bitflags::bitflags! {
    /// Binary flags listing features that may or may not be present on downlevel adapters.
    ///
    /// A downlevel adapter is a GPU adapter that wgpu supports, but with potentially limited
    /// features, due to the lack of hardware feature support.
    ///
    /// Flags that are **not** present for a downlevel adapter or device usually indicates
    /// non-compliance with the WebGPU specification, but not always.
    ///
    /// You can check whether a set of flags is compliant through the
    /// [`DownlevelCapabilities::is_webgpu_compliant()`] function.
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct DownlevelFlags: u32 {
        /// The device supports compiling and using compute shaders.
        ///
        /// WebGL2, and GLES3.0 devices do not support compute.
        const COMPUTE_SHADERS = 1 << 0;
        /// Supports binding storage buffers and textures to fragment shaders.
        const FRAGMENT_WRITABLE_STORAGE = 1 << 1;
        /// Supports indirect drawing and dispatching.
        ///
        /// [`Self::COMPUTE_SHADERS`] must be present for this flag.
        ///
        /// WebGL2, GLES 3.0, and Metal on Apple1/Apple2 GPUs do not support indirect.
        const INDIRECT_EXECUTION = 1 << 2;
        /// Supports non-zero `base_vertex` parameter to direct indexed draw calls.
        ///
        /// Indirect calls, if supported, always support non-zero `base_vertex`.
        ///
        /// Supported by:
        /// - Vulkan
        /// - DX12
        /// - Metal on Apple3+ or Mac1+
        /// - OpenGL 3.2+
        /// - OpenGL ES 3.2
        const BASE_VERTEX = 1 << 3;
        /// Supports reading from a depth/stencil texture while using it as a read-only
        /// depth/stencil attachment.
        ///
        /// The WebGL2 and GLES backends do not support RODS.
        const READ_ONLY_DEPTH_STENCIL = 1 << 4;
        /// Supports textures with mipmaps which have a non power of two size.
        const NON_POWER_OF_TWO_MIPMAPPED_TEXTURES = 1 << 5;
        /// Supports textures that are cube arrays.
        const CUBE_ARRAY_TEXTURES = 1 << 6;
        /// Supports comparison samplers.
        const COMPARISON_SAMPLERS = 1 << 7;
        /// Supports different blend operations per color attachment.
        const INDEPENDENT_BLEND = 1 << 8;
        /// Supports storage buffers in vertex shaders.
        const VERTEX_STORAGE = 1 << 9;

        /// Supports samplers with anisotropic filtering. Note this isn't actually required by
        /// WebGPU, the implementation is allowed to completely ignore aniso clamp. This flag is
        /// here for native backends so they can communicate to the user of aniso is enabled.
        ///
        /// All backends and all devices support anisotropic filtering.
        const ANISOTROPIC_FILTERING = 1 << 10;

        /// Supports storage buffers in fragment shaders.
        const FRAGMENT_STORAGE = 1 << 11;

        /// Supports sample-rate shading.
        const MULTISAMPLED_SHADING = 1 << 12;

        /// Supports copies between depth textures and buffers.
        ///
        /// GLES/WebGL don't support this.
        const DEPTH_TEXTURE_AND_BUFFER_COPIES = 1 << 13;

        /// Supports all the texture usages described in WebGPU. If this isn't supported, you
        /// should call `get_texture_format_features` to get how you can use textures of a given format
        const WEBGPU_TEXTURE_FORMAT_SUPPORT = 1 << 14;

        /// Supports buffer bindings with sizes that aren't a multiple of 16.
        ///
        /// WebGL doesn't support this.
        const BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED = 1 << 15;

        /// Supports buffers to combine [`BufferUsages::INDEX`] with usages other than [`BufferUsages::COPY_DST`] and [`BufferUsages::COPY_SRC`].
        /// Furthermore, in absence of this feature it is not allowed to copy index buffers from/to buffers with a set of usage flags containing
        /// [`BufferUsages::VERTEX`]/[`BufferUsages::UNIFORM`]/[`BufferUsages::STORAGE`] or [`BufferUsages::INDIRECT`].
        ///
        /// WebGL doesn't support this.
        const UNRESTRICTED_INDEX_BUFFER = 1 << 16;

        /// Supports full 32-bit range indices (2^32-1 as opposed to 2^24-1 without this flag)
        ///
        /// Corresponds to Vulkan's `VkPhysicalDeviceFeatures.fullDrawIndexUint32`
        const FULL_DRAW_INDEX_UINT32 = 1 << 17;

        /// Supports depth bias clamping
        ///
        /// Corresponds to Vulkan's `VkPhysicalDeviceFeatures.depthBiasClamp`
        const DEPTH_BIAS_CLAMP = 1 << 18;

        /// Supports specifying which view format values are allowed when create_view() is called on a texture.
        ///
        /// The WebGL and GLES backends doesn't support this.
        const VIEW_FORMATS = 1 << 19;

        /// With this feature not present, there are the following restrictions on `Queue::copy_external_image_to_texture`:
        /// - The source must not be [`web_sys::OffscreenCanvas`]
        /// - [`CopyExternalImageSourceInfo::origin`] must be zero.
        /// - [`CopyExternalImageDestInfo::color_space`] must be srgb.
        /// - If the source is an [`web_sys::ImageBitmap`]:
        ///   - [`CopyExternalImageSourceInfo::flip_y`] must be false.
        ///   - [`CopyExternalImageDestInfo::premultiplied_alpha`] must be false.
        ///
        /// WebGL doesn't support this. WebGPU does.
        const UNRESTRICTED_EXTERNAL_TEXTURE_COPIES = 1 << 20;

        /// Supports specifying which view formats are allowed when calling create_view on the texture returned by
        /// `Surface::get_current_texture`.
        ///
        /// The GLES/WebGL and Vulkan on Android doesn't support this.
        const SURFACE_VIEW_FORMATS = 1 << 21;

        /// If this is true, calls to `CommandEncoder::resolve_query_set` will be performed on the queue timeline.
        ///
        /// If this is false, calls to `CommandEncoder::resolve_query_set` will be performed on the device (i.e. cpu) timeline
        /// and will block that timeline until the query has data. You may work around this limitation by waiting until the submit
        /// whose queries you are resolving is fully finished (through use of `queue.on_submitted_work_done`) and only
        /// then submitting the resolve_query_set command. The queries will be guaranteed finished, so will not block.
        ///
        /// Supported by:
        /// - Vulkan,
        /// - DX12
        /// - Metal
        /// - OpenGL 4.4+
        ///
        /// Not Supported by:
        /// - GL ES / WebGL
        const NONBLOCKING_QUERY_RESOLVE = 1 << 22;

        /// Allows shaders to use `quantizeToF16`, `pack2x16float`, and `unpack2x16float`, which
        /// operate on `f16`-precision values stored in `f32`s.
        ///
        /// Not supported by Vulkan on Mesa when [`Features::SHADER_F16`] is absent.
        const SHADER_F16_IN_F32 = 1 << 23;
    }
}

impl DownlevelFlags {
    /// All flags that indicate if the backend is WebGPU compliant
    #[must_use]
    pub const fn compliant() -> Self {
        // We use manual bit twiddling to make this a const fn as `Sub` and `.remove` aren't const

        // WebGPU doesn't actually require aniso
        Self::from_bits_truncate(Self::all().bits() & !Self::ANISOTROPIC_FILTERING.bits())
    }
}

/// Collections of shader features a device supports if they support less than WebGPU normally allows.
// TODO: Fill out the differences between shader models more completely
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceType {
    /// Other or Unknown.
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
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdapterInfo {
    /// Adapter name
    pub name: String,
    /// [`Backend`]-specific vendor ID of the adapter
    ///
    /// This generally is a 16-bit PCI vendor ID in the least significant bytes of this field.
    /// However, more significant bytes may be non-zero if the backend uses a different
    /// representation.
    ///
    /// * For [`Backend::Vulkan`], the [`VkPhysicalDeviceProperties::vendorID`] is used, which is
    ///   a superset of PCI IDs.
    ///
    /// [`VkPhysicalDeviceProperties::vendorID`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceProperties.html
    pub vendor: u32,
    /// [`Backend`]-specific device ID of the adapter
    ///
    ///
    /// This generally is a 16-bit PCI device ID in the least significant bytes of this field.
    /// However, more significant bytes may be non-zero if the backend uses a different
    /// representation.
    ///
    /// * For [`Backend::Vulkan`], the [`VkPhysicalDeviceProperties::deviceID`] is used, which is
    ///   a superset of PCI IDs.
    ///
    /// [`VkPhysicalDeviceProperties::deviceID`]: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceProperties.html
    pub device: u32,
    /// Type of device
    pub device_type: DeviceType,
    /// Driver name
    pub driver: String,
    /// Driver info
    pub driver_info: String,
    /// Backend used for device
    pub backend: Backend,
}

/// Hints to the device about the memory allocation strategy.
///
/// Some backends may ignore these hints.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MemoryHints {
    /// Favor performance over memory usage (the default value).
    #[default]
    Performance,
    /// Favor memory usage over performance.
    MemoryUsage,
    /// Applications that have control over the content that is rendered
    /// (typically games) may find an optimal compromise between memory
    /// usage and performance by specifying the allocation configuration.
    Manual {
        /// Defines the range of allowed memory block sizes for sub-allocated
        /// resources.
        ///
        /// The backend may attempt to group multiple resources into fewer
        /// device memory blocks (sub-allocation) for performance reasons.
        /// The start of the provided range specifies the initial memory
        /// block size for sub-allocated resources. After running out of
        /// space in existing memory blocks, the backend may chose to
        /// progressively increase the block size of subsequent allocations
        /// up to a limit specified by the end of the range.
        ///
        /// This does not limit resource sizes. If a resource does not fit
        /// in the specified range, it will typically be placed in a dedicated
        /// memory block.
        suballocated_device_memory_block_size: Range<u64>,
    },
}

/// Describes a [`Device`](../wgpu/struct.Device.html).
///
/// Corresponds to [WebGPU `GPUDeviceDescriptor`](
/// https://gpuweb.github.io/gpuweb/#gpudevicedescriptor).
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeviceDescriptor<L> {
    /// Debug label for the device.
    pub label: L,
    /// Specifies the features that are required by the device request.
    /// The request will fail if the adapter cannot provide these features.
    ///
    /// Exactly the specified set of features, and no more or less,
    /// will be allowed in validation of API calls on the resulting device.
    pub required_features: Features,
    /// Specifies the limits that are required by the device request.
    /// The request will fail if the adapter cannot provide these limits.
    ///
    /// Exactly the specified limits, and no better or worse,
    /// will be allowed in validation of API calls on the resulting device.
    pub required_limits: Limits,
    /// Specifies whether `self.required_features` is allowed to contain experimental features.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub experimental_features: ExperimentalFeatures,
    /// Hints for memory allocation strategies.
    pub memory_hints: MemoryHints,
    /// Whether API tracing for debugging is enabled,
    /// and where the trace is written if so.
    pub trace: Trace,
}

impl<L> DeviceDescriptor<L> {
    /// Takes a closure and maps the label of the device descriptor into another.
    #[must_use]
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> DeviceDescriptor<K> {
        DeviceDescriptor {
            label: fun(&self.label),
            required_features: self.required_features,
            required_limits: self.required_limits.clone(),
            experimental_features: self.experimental_features,
            memory_hints: self.memory_hints.clone(),
            trace: self.trace.clone(),
        }
    }
}

/// Controls API call tracing and specifies where the trace is written.
///
/// **Note:** Tracing is currently unavailable.
/// See [issue 5974](https://github.com/gfx-rs/wgpu/issues/5974) for updates.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// This enum must be non-exhaustive so that enabling the "trace" feature is not a semver break.
#[non_exhaustive]
pub enum Trace {
    /// Tracing disabled.
    #[default]
    Off,

    /// Tracing enabled.
    #[cfg(feature = "trace")]
    // This must be owned rather than `&'a Path`, because if it were that, then the lifetime
    // parameter would be unused when the "trace" feature is disabled, which is prohibited.
    Directory(std::path::PathBuf),
}

bitflags::bitflags! {
    /// Describes the shader stages that a binding will be visible from.
    ///
    /// These can be combined so something that is visible from both vertex and fragment shaders can be defined as:
    ///
    /// `ShaderStages::VERTEX | ShaderStages::FRAGMENT`
    ///
    /// Corresponds to [WebGPU `GPUShaderStageFlags`](
    /// https://gpuweb.github.io/gpuweb/#typedefdef-gpushaderstageflags).
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
        const VERTEX_FRAGMENT = Self::VERTEX.bits() | Self::FRAGMENT.bits();
        /// Binding is visible from the task shader of a mesh pipeline.
        const TASK = 1 << 3;
        /// Binding is visible from the mesh shader of a mesh pipeline.
        const MESH = 1 << 4;
    }
}

/// Order in which texture data is laid out in memory.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub enum TextureDataOrder {
    /// The texture is laid out densely in memory as:
    ///
    /// ```text
    /// Layer0Mip0 Layer0Mip1 Layer0Mip2
    /// Layer1Mip0 Layer1Mip1 Layer1Mip2
    /// Layer2Mip0 Layer2Mip1 Layer2Mip2
    /// ````
    ///
    /// This is the layout used by dds files.
    #[default]
    LayerMajor,
    /// The texture is laid out densely in memory as:
    ///
    /// ```text
    /// Layer0Mip0 Layer1Mip0 Layer2Mip0
    /// Layer0Mip1 Layer1Mip1 Layer2Mip1
    /// Layer0Mip2 Layer1Mip2 Layer2Mip2
    /// ```
    ///
    /// This is the layout used by ktx and ktx2 files.
    MipMajor,
}

/// Dimensions of a particular texture view.
///
/// Corresponds to [WebGPU `GPUTextureViewDimension`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputextureviewdimension).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TextureViewDimension {
    /// A one dimensional texture. `texture_1d` in WGSL and `texture1D` in GLSL.
    #[cfg_attr(feature = "serde", serde(rename = "1d"))]
    D1,
    /// A two dimensional texture. `texture_2d` in WGSL and `texture2D` in GLSL.
    #[cfg_attr(feature = "serde", serde(rename = "2d"))]
    #[default]
    D2,
    /// A two dimensional array texture. `texture_2d_array` in WGSL and `texture2DArray` in GLSL.
    #[cfg_attr(feature = "serde", serde(rename = "2d-array"))]
    D2Array,
    /// A cubemap texture. `texture_cube` in WGSL and `textureCube` in GLSL.
    #[cfg_attr(feature = "serde", serde(rename = "cube"))]
    Cube,
    /// A cubemap array texture. `texture_cube_array` in WGSL and `textureCubeArray` in GLSL.
    #[cfg_attr(feature = "serde", serde(rename = "cube-array"))]
    CubeArray,
    /// A three dimensional texture. `texture_3d` in WGSL and `texture3D` in GLSL.
    #[cfg_attr(feature = "serde", serde(rename = "3d"))]
    D3,
}

impl TextureViewDimension {
    /// Get the texture dimension required of this texture view dimension.
    #[must_use]
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
/// Corresponds to [WebGPU `GPUBlendFactor`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpublendfactor). Values using `Src1`
/// require [`Features::DUAL_SOURCE_BLENDING`] and can only be used with the first
/// render target.
///
/// For further details on how the blend factors are applied, see the analogous
/// functionality in OpenGL: <https://www.khronos.org/opengl/wiki/Blending#Blending_Parameters>.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    /// S1.component
    Src1 = 13,
    /// 1.0 - S1.component
    OneMinusSrc1 = 14,
    /// S1.alpha
    Src1Alpha = 15,
    /// 1.0 - S1.alpha
    OneMinusSrc1Alpha = 16,
}

impl BlendFactor {
    /// Returns `true` if the blend factor references the second blend source.
    ///
    /// Note that the usage of those blend factors require [`Features::DUAL_SOURCE_BLENDING`].
    #[must_use]
    pub fn ref_second_blend_source(&self) -> bool {
        match self {
            BlendFactor::Src1
            | BlendFactor::OneMinusSrc1
            | BlendFactor::Src1Alpha
            | BlendFactor::OneMinusSrc1Alpha => true,
            _ => false,
        }
    }
}

/// Alpha blend operation.
///
/// Corresponds to [WebGPU `GPUBlendOperation`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpublendoperation).
///
/// For further details on how the blend operations are applied, see
/// the analogous functionality in OpenGL: <https://www.khronos.org/opengl/wiki/Blending#Blend_Equations>.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum BlendOperation {
    /// Src + Dst
    #[default]
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

/// Describes a blend component of a [`BlendState`].
///
/// Corresponds to [WebGPU `GPUBlendComponent`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpublendcomponent).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    /// Blend state of `(1 * src) + ((1 - src_alpha) * dst)`.
    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };

    /// Returns true if the state relies on the constant color, which is
    /// set independently on a render command encoder.
    #[must_use]
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

/// Describe the blend state of a render pipeline,
/// within [`ColorTargetState`].
///
/// Corresponds to [WebGPU `GPUBlendState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpublendstate).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
///
/// Corresponds to [WebGPU `GPUColorTargetState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucolortargetstate).
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ColorTargetState {
    /// The [`TextureFormat`] of the image that this pipeline will render to. Must match the format
    /// of the corresponding color attachment in [`CommandEncoder::begin_render_pass`][CEbrp]
    ///
    /// [CEbrp]: ../wgpu/struct.CommandEncoder.html#method.begin_render_pass
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
///
/// Corresponds to [WebGPU `GPUPrimitiveTopology`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpuprimitivetopology).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    #[default]
    TriangleList = 3,
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create four triangles `0 1 2`, `2 1 3`, `2 3 4`, and `4 3 5`
    TriangleStrip = 4,
}

impl PrimitiveTopology {
    /// Returns true for strip topologies.
    #[must_use]
    pub fn is_strip(&self) -> bool {
        match *self {
            Self::PointList | Self::LineList | Self::TriangleList => false,
            Self::LineStrip | Self::TriangleStrip => true,
        }
    }
}

/// Vertex winding order which classifies the "front" face of a triangle.
///
/// Corresponds to [WebGPU `GPUFrontFace`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpufrontface).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum FrontFace {
    /// Triangles with vertices in counter clockwise order are considered the front face.
    ///
    /// This is the default with right handed coordinate spaces.
    #[default]
    Ccw = 0,
    /// Triangles with vertices in clockwise order are considered the front face.
    ///
    /// This is the default with left handed coordinate spaces.
    Cw = 1,
}

/// Face of a vertex.
///
/// Corresponds to [WebGPU `GPUCullMode`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpucullmode),
/// except that the `"none"` value is represented using `Option<Face>` instead.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum Face {
    /// Front face
    Front = 0,
    /// Back face
    Back = 1,
}

/// Type of drawing mode for polygons
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum PolygonMode {
    /// Polygons are filled
    #[default]
    Fill = 0,
    /// Polygons are drawn as line segments
    Line = 1,
    /// Polygons are drawn as points
    Point = 2,
}

/// Describes the state of primitive assembly and rasterization in a render pipeline.
///
/// Corresponds to [WebGPU `GPUPrimitiveState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuprimitivestate).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct PrimitiveState {
    /// The primitive topology used to interpret vertices.
    pub topology: PrimitiveTopology,
    /// When drawing strip topologies with indices, this is the required format for the index buffer.
    /// This has no effect on non-indexed or non-strip draws.
    ///
    /// Specifying this value enables primitive restart, allowing individual strips to be separated
    /// with the index value `0xFFFF` when using `Uint16`, or `0xFFFFFFFF` when using `Uint32`.
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
    /// Enabling this requires [`Features::DEPTH_CLIP_CONTROL`] to be enabled.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unclipped_depth: bool,
    /// Controls the way each polygon is rasterized. Can be either `Fill` (default), `Line` or `Point`
    ///
    /// Setting this to `Line` requires [`Features::POLYGON_MODE_LINE`] to be enabled.
    ///
    /// Setting this to `Point` requires [`Features::POLYGON_MODE_POINT`] to be enabled.
    #[cfg_attr(feature = "serde", serde(default))]
    pub polygon_mode: PolygonMode,
    /// If set to true, the primitives are rendered with conservative overestimation. I.e. any rastered pixel touched by it is filled.
    /// Only valid for `[PolygonMode::Fill`]!
    ///
    /// Enabling this requires [`Features::CONSERVATIVE_RASTERIZATION`] to be enabled.
    pub conservative: bool,
}

/// Describes the multi-sampling state of a render pipeline.
///
/// Corresponds to [WebGPU `GPUMultisampleState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpumultisamplestate).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct MultisampleState {
    /// The number of samples calculated per pixel (for MSAA). For non-multisampled textures,
    /// this should be `1`
    pub count: u32,
    /// Bitmask that restricts the samples of a pixel modified by this pipeline. All samples
    /// can be enabled using the value `!0`
    pub mask: u64,
    /// When enabled, produces another sample mask per pixel based on the alpha output value, that
    /// is ANDed with the sample mask and the primitive coverage to restrict the set of samples
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
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct TextureFormatFeatureFlags: u32 {
        /// If not present, the texture can't be sampled with a filtering sampler.
        /// This may overwrite TextureSampleType::Float.filterable
        const FILTERABLE = 1 << 0;
        /// Allows [`TextureDescriptor::sample_count`] to be `2`.
        const MULTISAMPLE_X2 = 1 << 1;
        /// Allows [`TextureDescriptor::sample_count`] to be `4`.
        const MULTISAMPLE_X4 = 1 << 2 ;
        /// Allows [`TextureDescriptor::sample_count`] to be `8`.
        const MULTISAMPLE_X8 = 1 << 3 ;
        /// Allows [`TextureDescriptor::sample_count`] to be `16`.
        const MULTISAMPLE_X16 = 1 << 4;
        /// Allows a texture of this format to back a view passed as `resolve_target`
        /// to a render pass for an automatic driver-implemented resolve.
        const MULTISAMPLE_RESOLVE = 1 << 5;
        /// When used as a STORAGE texture, then a texture with this format can be bound with
        /// [`StorageTextureAccess::ReadOnly`].
        const STORAGE_READ_ONLY = 1 << 6;
        /// When used as a STORAGE texture, then a texture with this format can be bound with
        /// [`StorageTextureAccess::WriteOnly`].
        const STORAGE_WRITE_ONLY = 1 << 7;
        /// When used as a STORAGE texture, then a texture with this format can be bound with
        /// [`StorageTextureAccess::ReadWrite`].
        const STORAGE_READ_WRITE = 1 << 8;
        /// When used as a STORAGE texture, then a texture with this format can be bound with
        /// [`StorageTextureAccess::Atomic`].
        const STORAGE_ATOMIC = 1 << 9;
        /// If not present, the texture can't be blended into the render target.
        const BLENDABLE = 1 << 10;
    }
}

impl TextureFormatFeatureFlags {
    /// Sample count supported by a given texture format.
    ///
    /// returns `true` if `count` is a supported sample count.
    #[must_use]
    pub fn sample_count_supported(&self, count: u32) -> bool {
        use TextureFormatFeatureFlags as tfsc;

        match count {
            1 => true,
            2 => self.contains(tfsc::MULTISAMPLE_X2),
            4 => self.contains(tfsc::MULTISAMPLE_X4),
            8 => self.contains(tfsc::MULTISAMPLE_X8),
            16 => self.contains(tfsc::MULTISAMPLE_X16),
            _ => false,
        }
    }

    /// A `Vec` of supported sample counts.
    #[must_use]
    pub fn supported_sample_counts(&self) -> Vec<u32> {
        let all_possible_sample_counts: [u32; 5] = [1, 2, 4, 8, 16];
        all_possible_sample_counts
            .into_iter()
            .filter(|&sc| self.sample_count_supported(sc))
            .collect()
    }
}

/// Features supported by a given texture format
///
/// Features are defined by WebGPU specification unless [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] is enabled.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureFormatFeatures {
    /// Valid bits for `TextureDescriptor::Usage` provided for format creation.
    pub allowed_usages: TextureUsages,
    /// Additional property flags for the format.
    pub flags: TextureFormatFeatureFlags,
}

/// ASTC block dimensions
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AstcBlock {
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px).
    B4x4,
    /// 5x4 block compressed texture. 16 bytes per block (6.4 bit/px).
    B5x4,
    /// 5x5 block compressed texture. 16 bytes per block (5.12 bit/px).
    B5x5,
    /// 6x5 block compressed texture. 16 bytes per block (4.27 bit/px).
    B6x5,
    /// 6x6 block compressed texture. 16 bytes per block (3.56 bit/px).
    B6x6,
    /// 8x5 block compressed texture. 16 bytes per block (3.2 bit/px).
    B8x5,
    /// 8x6 block compressed texture. 16 bytes per block (2.67 bit/px).
    B8x6,
    /// 8x8 block compressed texture. 16 bytes per block (2 bit/px).
    B8x8,
    /// 10x5 block compressed texture. 16 bytes per block (2.56 bit/px).
    B10x5,
    /// 10x6 block compressed texture. 16 bytes per block (2.13 bit/px).
    B10x6,
    /// 10x8 block compressed texture. 16 bytes per block (1.6 bit/px).
    B10x8,
    /// 10x10 block compressed texture. 16 bytes per block (1.28 bit/px).
    B10x10,
    /// 12x10 block compressed texture. 16 bytes per block (1.07 bit/px).
    B12x10,
    /// 12x12 block compressed texture. 16 bytes per block (0.89 bit/px).
    B12x12,
}

/// ASTC RGBA channel
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AstcChannel {
    /// 8 bit integer RGBA, [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC`] must be enabled to use this channel.
    Unorm,
    /// 8 bit integer RGBA, Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC`] must be enabled to use this channel.
    UnormSrgb,
    /// floating-point RGBA, linear-color float can be outside of the [0, 1] range.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ASTC_HDR`] must be enabled to use this channel.
    Hdr,
}

/// Format in which a texture’s texels are stored in GPU memory.
///
/// Certain formats additionally specify a conversion.
/// When these formats are used in a shader, the conversion automatically takes place when loading
/// from or storing to the texture.
///
/// * `Unorm` formats linearly scale the integer range of the storage format to a floating-point
///   range of 0 to 1, inclusive.
/// * `Snorm` formats linearly scale the integer range of the storage format to a floating-point
///   range of &minus;1 to 1, inclusive, except that the most negative value
///   (&minus;128 for 8-bit, &minus;32768 for 16-bit) is excluded; on conversion,
///   it is treated as identical to the second most negative
///   (&minus;127 for 8-bit, &minus;32767 for 16-bit),
///   so that the positive and negative ranges are symmetric.
/// * `UnormSrgb` formats apply the [sRGB transfer function] so that the storage is sRGB encoded
///   while the shader works with linear intensity values.
/// * `Uint`, `Sint`, and `Float` formats perform no conversion.
///
/// Corresponds to [WebGPU `GPUTextureFormat`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputextureformat).
///
/// [sRGB transfer function]: https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22)
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum TextureFormat {
    // Normal 8 bit formats
    /// Red channel only. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    R8Unorm,
    /// Red channel only. 8 bit integer per channel. [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    R8Snorm,
    /// Red channel only. 8 bit integer per channel. Unsigned in shader.
    R8Uint,
    /// Red channel only. 8 bit integer per channel. Signed in shader.
    R8Sint,

    // Normal 16 bit formats
    /// Red channel only. 16 bit integer per channel. Unsigned in shader.
    R16Uint,
    /// Red channel only. 16 bit integer per channel. Signed in shader.
    R16Sint,
    /// Red channel only. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    R16Unorm,
    /// Red channel only. 16 bit integer per channel. [&minus;32767, 32767] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    R16Snorm,
    /// Red channel only. 16 bit float per channel. Float in shader.
    R16Float,
    /// Red and green channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    Rg8Unorm,
    /// Red and green channels. 8 bit integer per channel. [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    Rg8Snorm,
    /// Red and green channels. 8 bit integer per channel. Unsigned in shader.
    Rg8Uint,
    /// Red and green channels. 8 bit integer per channel. Signed in shader.
    Rg8Sint,

    // Normal 32 bit formats
    /// Red channel only. 32 bit integer per channel. Unsigned in shader.
    R32Uint,
    /// Red channel only. 32 bit integer per channel. Signed in shader.
    R32Sint,
    /// Red channel only. 32 bit float per channel. Float in shader.
    R32Float,
    /// Red and green channels. 16 bit integer per channel. Unsigned in shader.
    Rg16Uint,
    /// Red and green channels. 16 bit integer per channel. Signed in shader.
    Rg16Sint,
    /// Red and green channels. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    Rg16Unorm,
    /// Red and green channels. 16 bit integer per channel. [&minus;32767, 32767] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    Rg16Snorm,
    /// Red and green channels. 16 bit float per channel. Float in shader.
    Rg16Float,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    Rgba8Unorm,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    Rgba8UnormSrgb,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    Rgba8Snorm,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Unsigned in shader.
    Rgba8Uint,
    /// Red, green, blue, and alpha channels. 8 bit integer per channel. Signed in shader.
    Rgba8Sint,
    /// Blue, green, red, and alpha channels. 8 bit integer per channel. [0, 255] converted to/from float [0, 1] in shader.
    Bgra8Unorm,
    /// Blue, green, red, and alpha channels. 8 bit integer per channel. Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    Bgra8UnormSrgb,

    // Packed 32 bit formats
    /// Packed unsigned float with 9 bits mantisa for each RGB component, then a common 5 bits exponent
    Rgb9e5Ufloat,
    /// Red, green, blue, and alpha channels. 10 bit integer for RGB channels, 2 bit integer for alpha channel. Unsigned in shader.
    Rgb10a2Uint,
    /// Red, green, blue, and alpha channels. 10 bit integer for RGB channels, 2 bit integer for alpha channel. [0, 1023] ([0, 3] for alpha) converted to/from float [0, 1] in shader.
    Rgb10a2Unorm,
    /// Red, green, and blue channels. 11 bit float with no sign bit for RG channels. 10 bit float with no sign bit for blue channel. Float in shader.
    Rg11b10Ufloat,

    // Normal 64 bit formats
    /// Red channel only. 64 bit integer per channel. Unsigned in shader.
    ///
    /// [`Features::TEXTURE_INT64_ATOMIC`] must be enabled to use this texture format.
    R64Uint,
    /// Red and green channels. 32 bit integer per channel. Unsigned in shader.
    Rg32Uint,
    /// Red and green channels. 32 bit integer per channel. Signed in shader.
    Rg32Sint,
    /// Red and green channels. 32 bit float per channel. Float in shader.
    Rg32Float,
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. Unsigned in shader.
    Rgba16Uint,
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. Signed in shader.
    Rgba16Sint,
    /// Red, green, blue, and alpha channels. 16 bit integer per channel. [0, 65535] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    Rgba16Unorm,
    /// Red, green, blue, and alpha. 16 bit integer per channel. [&minus;32767, 32767] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_FORMAT_16BIT_NORM`] must be enabled to use this texture format.
    Rgba16Snorm,
    /// Red, green, blue, and alpha channels. 16 bit float per channel. Float in shader.
    Rgba16Float,

    // Normal 128 bit formats
    /// Red, green, blue, and alpha channels. 32 bit integer per channel. Unsigned in shader.
    Rgba32Uint,
    /// Red, green, blue, and alpha channels. 32 bit integer per channel. Signed in shader.
    Rgba32Sint,
    /// Red, green, blue, and alpha channels. 32 bit float per channel. Float in shader.
    Rgba32Float,

    // Depth and stencil formats
    /// Stencil format with 8 bit integer stencil.
    Stencil8,
    /// Special depth format with 16 bit integer depth.
    Depth16Unorm,
    /// Special depth format with at least 24 bit integer depth.
    Depth24Plus,
    /// Special depth/stencil format with at least 24 bit integer depth and 8 bits integer stencil.
    Depth24PlusStencil8,
    /// Special depth format with 32 bit floating point depth.
    Depth32Float,
    /// Special depth/stencil format with 32 bit floating point depth and 8 bits integer stencil.
    ///
    /// [`Features::DEPTH32FLOAT_STENCIL8`] must be enabled to use this texture format.
    Depth32FloatStencil8,

    /// YUV 4:2:0 chroma subsampled format.
    ///
    /// Contains two planes:
    /// - 0: Single 8 bit channel luminance.
    /// - 1: Dual 8 bit channel chrominance at half width and half height.
    ///
    /// Valid view formats for luminance are [`TextureFormat::R8Unorm`].
    ///
    /// Valid view formats for chrominance are [`TextureFormat::Rg8Unorm`].
    ///
    /// Width and height must be even.
    ///
    /// [`Features::TEXTURE_FORMAT_NV12`] must be enabled to use this texture format.
    NV12,

    /// YUV 4:2:0 chroma subsampled format.
    ///
    /// Contains two planes:
    /// - 0: Single 16 bit channel luminance, of which only the high 10 bits
    ///   are used.
    /// - 1: Dual 16 bit channel chrominance at half width and half height, of
    ///   which only the high 10 bits are used.
    ///
    /// Valid view formats for luminance are [`TextureFormat::R16Unorm`].
    ///
    /// Valid view formats for chrominance are [`TextureFormat::Rg16Unorm`].
    ///
    /// Width and height must be even.
    ///
    /// [`Features::TEXTURE_FORMAT_P010`] must be enabled to use this texture format.
    P010,

    // Compressed textures usable with `TEXTURE_COMPRESSION_BC` feature. `TEXTURE_COMPRESSION_SLICED_3D` is required to use with 3D textures.
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 4 color + alpha pallet. 5 bit R + 6 bit G + 5 bit B + 1 bit alpha.
    /// [0, 63] ([0, 1] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc1RgbaUnorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 4 color + alpha pallet. 5 bit R + 6 bit G + 5 bit B + 1 bit alpha.
    /// Srgb-color [0, 63] ([0, 1] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc1RgbaUnormSrgb,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet. 5 bit R + 6 bit G + 5 bit B + 4 bit alpha.
    /// [0, 63] ([0, 15] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT3.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc2RgbaUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet. 5 bit R + 6 bit G + 5 bit B + 4 bit alpha.
    /// Srgb-color [0, 63] ([0, 255] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT3.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc2RgbaUnormSrgb,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet + 8 alpha pallet. 5 bit R + 6 bit G + 5 bit B + 8 bit alpha.
    /// [0, 63] ([0, 255] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// Also known as DXT5.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc3RgbaUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 4 color pallet + 8 alpha pallet. 5 bit R + 6 bit G + 5 bit B + 8 bit alpha.
    /// Srgb-color [0, 63] ([0, 255] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as DXT5.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc3RgbaUnormSrgb,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 8 color pallet. 8 bit R.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as RGTC1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc4RUnorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). 8 color pallet. 8 bit R.
    /// [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    ///
    /// Also known as RGTC1.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc4RSnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 8 color red pallet + 8 color green pallet. 8 bit RG.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as RGTC2.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc5RgUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). 8 color red pallet + 8 color green pallet. 8 bit RG.
    /// [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    ///
    /// Also known as RGTC2.
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc5RgSnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 16 bit unsigned float RGB. Float in shader.
    ///
    /// Also known as BPTC (float).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc6hRgbUfloat,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 16 bit signed float RGB. Float in shader.
    ///
    /// Also known as BPTC (float).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc6hRgbFloat,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 8 bit integer RGBA.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// Also known as BPTC (unorm).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc7RgbaUnorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Variable sized pallet. 8 bit integer RGBA.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// Also known as BPTC (unorm).
    ///
    /// [`Features::TEXTURE_COMPRESSION_BC`] must be enabled to use this texture format.
    /// [`Features::TEXTURE_COMPRESSION_BC_SLICED_3D`] must be enabled to use this texture format with 3D dimension.
    Bc7RgbaUnormSrgb,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    Etc2Rgb8Unorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    Etc2Rgb8UnormSrgb,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB + 1 bit alpha.
    /// [0, 255] ([0, 1] for alpha) converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    Etc2Rgb8A1Unorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 8 bit integer RGB + 1 bit alpha.
    /// Srgb-color [0, 255] ([0, 1] for alpha) converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    Etc2Rgb8A1UnormSrgb,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 8 bit integer RGB + 8 bit alpha.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    Etc2Rgba8Unorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 8 bit integer RGB + 8 bit alpha.
    /// Srgb-color [0, 255] converted to/from linear-color float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    Etc2Rgba8UnormSrgb,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 11 bit integer R.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    EacR11Unorm,
    /// 4x4 block compressed texture. 8 bytes per block (4 bit/px). Complex pallet. 11 bit integer R.
    /// [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    EacR11Snorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 11 bit integer R + 11 bit integer G.
    /// [0, 255] converted to/from float [0, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    EacRg11Unorm,
    /// 4x4 block compressed texture. 16 bytes per block (8 bit/px). Complex pallet. 11 bit integer R + 11 bit integer G.
    /// [&minus;127, 127] converted to/from float [&minus;1, 1] in shader.
    ///
    /// [`Features::TEXTURE_COMPRESSION_ETC2`] must be enabled to use this texture format.
    EacRg11Snorm,
    /// block compressed texture. 16 bytes per block.
    ///
    /// Features [`TEXTURE_COMPRESSION_ASTC`] or [`TEXTURE_COMPRESSION_ASTC_HDR`]
    /// must be enabled to use this texture format.
    ///
    /// [`TEXTURE_COMPRESSION_ASTC`]: Features::TEXTURE_COMPRESSION_ASTC
    /// [`TEXTURE_COMPRESSION_ASTC_HDR`]: Features::TEXTURE_COMPRESSION_ASTC_HDR
    Astc {
        /// compressed block dimensions
        block: AstcBlock,
        /// ASTC RGBA channel
        channel: AstcChannel,
    },
}

#[cfg(any(feature = "serde", test))]
impl<'de> Deserialize<'de> for TextureFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Error, Unexpected};

        struct TextureFormatVisitor;

        impl de::Visitor<'_> for TextureFormatVisitor {
            type Value = TextureFormat;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("a valid texture format")
            }

            fn visit_str<E: Error>(self, s: &str) -> Result<Self::Value, E> {
                let format = match s {
                    "r8unorm" => TextureFormat::R8Unorm,
                    "r8snorm" => TextureFormat::R8Snorm,
                    "r8uint" => TextureFormat::R8Uint,
                    "r8sint" => TextureFormat::R8Sint,
                    "r16uint" => TextureFormat::R16Uint,
                    "r16sint" => TextureFormat::R16Sint,
                    "r16unorm" => TextureFormat::R16Unorm,
                    "r16snorm" => TextureFormat::R16Snorm,
                    "r16float" => TextureFormat::R16Float,
                    "rg8unorm" => TextureFormat::Rg8Unorm,
                    "rg8snorm" => TextureFormat::Rg8Snorm,
                    "rg8uint" => TextureFormat::Rg8Uint,
                    "rg8sint" => TextureFormat::Rg8Sint,
                    "r32uint" => TextureFormat::R32Uint,
                    "r32sint" => TextureFormat::R32Sint,
                    "r32float" => TextureFormat::R32Float,
                    "rg16uint" => TextureFormat::Rg16Uint,
                    "rg16sint" => TextureFormat::Rg16Sint,
                    "rg16unorm" => TextureFormat::Rg16Unorm,
                    "rg16snorm" => TextureFormat::Rg16Snorm,
                    "rg16float" => TextureFormat::Rg16Float,
                    "rgba8unorm" => TextureFormat::Rgba8Unorm,
                    "rgba8unorm-srgb" => TextureFormat::Rgba8UnormSrgb,
                    "rgba8snorm" => TextureFormat::Rgba8Snorm,
                    "rgba8uint" => TextureFormat::Rgba8Uint,
                    "rgba8sint" => TextureFormat::Rgba8Sint,
                    "bgra8unorm" => TextureFormat::Bgra8Unorm,
                    "bgra8unorm-srgb" => TextureFormat::Bgra8UnormSrgb,
                    "rgb10a2uint" => TextureFormat::Rgb10a2Uint,
                    "rgb10a2unorm" => TextureFormat::Rgb10a2Unorm,
                    "rg11b10ufloat" => TextureFormat::Rg11b10Ufloat,
                    "r64uint" => TextureFormat::R64Uint,
                    "rg32uint" => TextureFormat::Rg32Uint,
                    "rg32sint" => TextureFormat::Rg32Sint,
                    "rg32float" => TextureFormat::Rg32Float,
                    "rgba16uint" => TextureFormat::Rgba16Uint,
                    "rgba16sint" => TextureFormat::Rgba16Sint,
                    "rgba16unorm" => TextureFormat::Rgba16Unorm,
                    "rgba16snorm" => TextureFormat::Rgba16Snorm,
                    "rgba16float" => TextureFormat::Rgba16Float,
                    "rgba32uint" => TextureFormat::Rgba32Uint,
                    "rgba32sint" => TextureFormat::Rgba32Sint,
                    "rgba32float" => TextureFormat::Rgba32Float,
                    "stencil8" => TextureFormat::Stencil8,
                    "depth32float" => TextureFormat::Depth32Float,
                    "depth32float-stencil8" => TextureFormat::Depth32FloatStencil8,
                    "depth16unorm" => TextureFormat::Depth16Unorm,
                    "depth24plus" => TextureFormat::Depth24Plus,
                    "depth24plus-stencil8" => TextureFormat::Depth24PlusStencil8,
                    "nv12" => TextureFormat::NV12,
                    "p010" => TextureFormat::P010,
                    "rgb9e5ufloat" => TextureFormat::Rgb9e5Ufloat,
                    "bc1-rgba-unorm" => TextureFormat::Bc1RgbaUnorm,
                    "bc1-rgba-unorm-srgb" => TextureFormat::Bc1RgbaUnormSrgb,
                    "bc2-rgba-unorm" => TextureFormat::Bc2RgbaUnorm,
                    "bc2-rgba-unorm-srgb" => TextureFormat::Bc2RgbaUnormSrgb,
                    "bc3-rgba-unorm" => TextureFormat::Bc3RgbaUnorm,
                    "bc3-rgba-unorm-srgb" => TextureFormat::Bc3RgbaUnormSrgb,
                    "bc4-r-unorm" => TextureFormat::Bc4RUnorm,
                    "bc4-r-snorm" => TextureFormat::Bc4RSnorm,
                    "bc5-rg-unorm" => TextureFormat::Bc5RgUnorm,
                    "bc5-rg-snorm" => TextureFormat::Bc5RgSnorm,
                    "bc6h-rgb-ufloat" => TextureFormat::Bc6hRgbUfloat,
                    "bc6h-rgb-float" => TextureFormat::Bc6hRgbFloat,
                    "bc7-rgba-unorm" => TextureFormat::Bc7RgbaUnorm,
                    "bc7-rgba-unorm-srgb" => TextureFormat::Bc7RgbaUnormSrgb,
                    "etc2-rgb8unorm" => TextureFormat::Etc2Rgb8Unorm,
                    "etc2-rgb8unorm-srgb" => TextureFormat::Etc2Rgb8UnormSrgb,
                    "etc2-rgb8a1unorm" => TextureFormat::Etc2Rgb8A1Unorm,
                    "etc2-rgb8a1unorm-srgb" => TextureFormat::Etc2Rgb8A1UnormSrgb,
                    "etc2-rgba8unorm" => TextureFormat::Etc2Rgba8Unorm,
                    "etc2-rgba8unorm-srgb" => TextureFormat::Etc2Rgba8UnormSrgb,
                    "eac-r11unorm" => TextureFormat::EacR11Unorm,
                    "eac-r11snorm" => TextureFormat::EacR11Snorm,
                    "eac-rg11unorm" => TextureFormat::EacRg11Unorm,
                    "eac-rg11snorm" => TextureFormat::EacRg11Snorm,
                    other => {
                        if let Some(parts) = other.strip_prefix("astc-") {
                            let (block, channel) = parts
                                .split_once('-')
                                .ok_or_else(|| E::invalid_value(Unexpected::Str(s), &self))?;

                            let block = match block {
                                "4x4" => AstcBlock::B4x4,
                                "5x4" => AstcBlock::B5x4,
                                "5x5" => AstcBlock::B5x5,
                                "6x5" => AstcBlock::B6x5,
                                "6x6" => AstcBlock::B6x6,
                                "8x5" => AstcBlock::B8x5,
                                "8x6" => AstcBlock::B8x6,
                                "8x8" => AstcBlock::B8x8,
                                "10x5" => AstcBlock::B10x5,
                                "10x6" => AstcBlock::B10x6,
                                "10x8" => AstcBlock::B10x8,
                                "10x10" => AstcBlock::B10x10,
                                "12x10" => AstcBlock::B12x10,
                                "12x12" => AstcBlock::B12x12,
                                _ => return Err(E::invalid_value(Unexpected::Str(s), &self)),
                            };

                            let channel = match channel {
                                "unorm" => AstcChannel::Unorm,
                                "unorm-srgb" => AstcChannel::UnormSrgb,
                                "hdr" => AstcChannel::Hdr,
                                _ => return Err(E::invalid_value(Unexpected::Str(s), &self)),
                            };

                            TextureFormat::Astc { block, channel }
                        } else {
                            return Err(E::invalid_value(Unexpected::Str(s), &self));
                        }
                    }
                };

                Ok(format)
            }
        }

        deserializer.deserialize_str(TextureFormatVisitor)
    }
}

#[cfg(any(feature = "serde", test))]
impl Serialize for TextureFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s: String;
        let name = match *self {
            TextureFormat::R8Unorm => "r8unorm",
            TextureFormat::R8Snorm => "r8snorm",
            TextureFormat::R8Uint => "r8uint",
            TextureFormat::R8Sint => "r8sint",
            TextureFormat::R16Uint => "r16uint",
            TextureFormat::R16Sint => "r16sint",
            TextureFormat::R16Unorm => "r16unorm",
            TextureFormat::R16Snorm => "r16snorm",
            TextureFormat::R16Float => "r16float",
            TextureFormat::Rg8Unorm => "rg8unorm",
            TextureFormat::Rg8Snorm => "rg8snorm",
            TextureFormat::Rg8Uint => "rg8uint",
            TextureFormat::Rg8Sint => "rg8sint",
            TextureFormat::R32Uint => "r32uint",
            TextureFormat::R32Sint => "r32sint",
            TextureFormat::R32Float => "r32float",
            TextureFormat::Rg16Uint => "rg16uint",
            TextureFormat::Rg16Sint => "rg16sint",
            TextureFormat::Rg16Unorm => "rg16unorm",
            TextureFormat::Rg16Snorm => "rg16snorm",
            TextureFormat::Rg16Float => "rg16float",
            TextureFormat::Rgba8Unorm => "rgba8unorm",
            TextureFormat::Rgba8UnormSrgb => "rgba8unorm-srgb",
            TextureFormat::Rgba8Snorm => "rgba8snorm",
            TextureFormat::Rgba8Uint => "rgba8uint",
            TextureFormat::Rgba8Sint => "rgba8sint",
            TextureFormat::Bgra8Unorm => "bgra8unorm",
            TextureFormat::Bgra8UnormSrgb => "bgra8unorm-srgb",
            TextureFormat::Rgb10a2Uint => "rgb10a2uint",
            TextureFormat::Rgb10a2Unorm => "rgb10a2unorm",
            TextureFormat::Rg11b10Ufloat => "rg11b10ufloat",
            TextureFormat::R64Uint => "r64uint",
            TextureFormat::Rg32Uint => "rg32uint",
            TextureFormat::Rg32Sint => "rg32sint",
            TextureFormat::Rg32Float => "rg32float",
            TextureFormat::Rgba16Uint => "rgba16uint",
            TextureFormat::Rgba16Sint => "rgba16sint",
            TextureFormat::Rgba16Unorm => "rgba16unorm",
            TextureFormat::Rgba16Snorm => "rgba16snorm",
            TextureFormat::Rgba16Float => "rgba16float",
            TextureFormat::Rgba32Uint => "rgba32uint",
            TextureFormat::Rgba32Sint => "rgba32sint",
            TextureFormat::Rgba32Float => "rgba32float",
            TextureFormat::Stencil8 => "stencil8",
            TextureFormat::Depth32Float => "depth32float",
            TextureFormat::Depth16Unorm => "depth16unorm",
            TextureFormat::Depth32FloatStencil8 => "depth32float-stencil8",
            TextureFormat::Depth24Plus => "depth24plus",
            TextureFormat::Depth24PlusStencil8 => "depth24plus-stencil8",
            TextureFormat::NV12 => "nv12",
            TextureFormat::P010 => "p010",
            TextureFormat::Rgb9e5Ufloat => "rgb9e5ufloat",
            TextureFormat::Bc1RgbaUnorm => "bc1-rgba-unorm",
            TextureFormat::Bc1RgbaUnormSrgb => "bc1-rgba-unorm-srgb",
            TextureFormat::Bc2RgbaUnorm => "bc2-rgba-unorm",
            TextureFormat::Bc2RgbaUnormSrgb => "bc2-rgba-unorm-srgb",
            TextureFormat::Bc3RgbaUnorm => "bc3-rgba-unorm",
            TextureFormat::Bc3RgbaUnormSrgb => "bc3-rgba-unorm-srgb",
            TextureFormat::Bc4RUnorm => "bc4-r-unorm",
            TextureFormat::Bc4RSnorm => "bc4-r-snorm",
            TextureFormat::Bc5RgUnorm => "bc5-rg-unorm",
            TextureFormat::Bc5RgSnorm => "bc5-rg-snorm",
            TextureFormat::Bc6hRgbUfloat => "bc6h-rgb-ufloat",
            TextureFormat::Bc6hRgbFloat => "bc6h-rgb-float",
            TextureFormat::Bc7RgbaUnorm => "bc7-rgba-unorm",
            TextureFormat::Bc7RgbaUnormSrgb => "bc7-rgba-unorm-srgb",
            TextureFormat::Etc2Rgb8Unorm => "etc2-rgb8unorm",
            TextureFormat::Etc2Rgb8UnormSrgb => "etc2-rgb8unorm-srgb",
            TextureFormat::Etc2Rgb8A1Unorm => "etc2-rgb8a1unorm",
            TextureFormat::Etc2Rgb8A1UnormSrgb => "etc2-rgb8a1unorm-srgb",
            TextureFormat::Etc2Rgba8Unorm => "etc2-rgba8unorm",
            TextureFormat::Etc2Rgba8UnormSrgb => "etc2-rgba8unorm-srgb",
            TextureFormat::EacR11Unorm => "eac-r11unorm",
            TextureFormat::EacR11Snorm => "eac-r11snorm",
            TextureFormat::EacRg11Unorm => "eac-rg11unorm",
            TextureFormat::EacRg11Snorm => "eac-rg11snorm",
            TextureFormat::Astc { block, channel } => {
                let block = match block {
                    AstcBlock::B4x4 => "4x4",
                    AstcBlock::B5x4 => "5x4",
                    AstcBlock::B5x5 => "5x5",
                    AstcBlock::B6x5 => "6x5",
                    AstcBlock::B6x6 => "6x6",
                    AstcBlock::B8x5 => "8x5",
                    AstcBlock::B8x6 => "8x6",
                    AstcBlock::B8x8 => "8x8",
                    AstcBlock::B10x5 => "10x5",
                    AstcBlock::B10x6 => "10x6",
                    AstcBlock::B10x8 => "10x8",
                    AstcBlock::B10x10 => "10x10",
                    AstcBlock::B12x10 => "12x10",
                    AstcBlock::B12x12 => "12x12",
                };

                let channel = match channel {
                    AstcChannel::Unorm => "unorm",
                    AstcChannel::UnormSrgb => "unorm-srgb",
                    AstcChannel::Hdr => "hdr",
                };

                s = format!("astc-{block}-{channel}");
                &s
            }
        };
        serializer.serialize_str(name)
    }
}

impl TextureAspect {
    /// Returns the texture aspect for a given plane.
    #[must_use]
    pub fn from_plane(plane: u32) -> Option<Self> {
        Some(match plane {
            0 => Self::Plane0,
            1 => Self::Plane1,
            2 => Self::Plane2,
            _ => return None,
        })
    }
}

// There are some additional texture format helpers in `wgpu-core/src/conv.rs`,
// that may need to be modified along with the ones here.
impl TextureFormat {
    /// Returns the aspect-specific format of the original format
    ///
    /// see <https://gpuweb.github.io/gpuweb/#abstract-opdef-resolving-gputextureaspect>
    #[must_use]
    pub fn aspect_specific_format(&self, aspect: TextureAspect) -> Option<Self> {
        match (*self, aspect) {
            (Self::Stencil8, TextureAspect::StencilOnly) => Some(*self),
            (
                Self::Depth16Unorm | Self::Depth24Plus | Self::Depth32Float,
                TextureAspect::DepthOnly,
            ) => Some(*self),
            (
                Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8,
                TextureAspect::StencilOnly,
            ) => Some(Self::Stencil8),
            (Self::Depth24PlusStencil8, TextureAspect::DepthOnly) => Some(Self::Depth24Plus),
            (Self::Depth32FloatStencil8, TextureAspect::DepthOnly) => Some(Self::Depth32Float),
            (Self::NV12, TextureAspect::Plane0) => Some(Self::R8Unorm),
            (Self::NV12, TextureAspect::Plane1) => Some(Self::Rg8Unorm),
            (Self::P010, TextureAspect::Plane0) => Some(Self::R16Unorm),
            (Self::P010, TextureAspect::Plane1) => Some(Self::Rg16Unorm),
            // views to multi-planar formats must specify the plane
            (format, TextureAspect::All) if !format.is_multi_planar_format() => Some(format),
            _ => None,
        }
    }

    /// Returns `true` if `self` is a depth or stencil component of the given
    /// combined depth-stencil format
    #[must_use]
    pub fn is_depth_stencil_component(&self, combined_format: Self) -> bool {
        match (combined_format, *self) {
            (Self::Depth24PlusStencil8, Self::Depth24Plus | Self::Stencil8)
            | (Self::Depth32FloatStencil8, Self::Depth32Float | Self::Stencil8) => true,
            _ => false,
        }
    }

    /// Returns `true` if the format is a depth and/or stencil format
    ///
    /// see <https://gpuweb.github.io/gpuweb/#depth-formats>
    #[must_use]
    pub fn is_depth_stencil_format(&self) -> bool {
        match *self {
            Self::Stencil8
            | Self::Depth16Unorm
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8
            | Self::Depth32Float
            | Self::Depth32FloatStencil8 => true,
            _ => false,
        }
    }

    /// Returns `true` if the format is a combined depth-stencil format
    ///
    /// see <https://gpuweb.github.io/gpuweb/#combined-depth-stencil-format>
    #[must_use]
    pub fn is_combined_depth_stencil_format(&self) -> bool {
        match *self {
            Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8 => true,
            _ => false,
        }
    }

    /// Returns `true` if the format is a multi-planar format
    #[must_use]
    pub fn is_multi_planar_format(&self) -> bool {
        self.planes().is_some()
    }

    /// Returns the number of planes a multi-planar format has.
    #[must_use]
    pub fn planes(&self) -> Option<u32> {
        match *self {
            Self::NV12 => Some(2),
            Self::P010 => Some(2),
            _ => None,
        }
    }

    /// Returns `true` if the format has a color aspect
    #[must_use]
    pub fn has_color_aspect(&self) -> bool {
        !self.is_depth_stencil_format()
    }

    /// Returns `true` if the format has a depth aspect
    #[must_use]
    pub fn has_depth_aspect(&self) -> bool {
        match *self {
            Self::Depth16Unorm
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8
            | Self::Depth32Float
            | Self::Depth32FloatStencil8 => true,
            _ => false,
        }
    }

    /// Returns `true` if the format has a stencil aspect
    #[must_use]
    pub fn has_stencil_aspect(&self) -> bool {
        match *self {
            Self::Stencil8 | Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8 => true,
            _ => false,
        }
    }

    /// Returns the size multiple requirement for a texture using this format.
    #[must_use]
    pub fn size_multiple_requirement(&self) -> (u32, u32) {
        match *self {
            Self::NV12 => (2, 2),
            Self::P010 => (2, 2),
            _ => self.block_dimensions(),
        }
    }

    /// Returns the dimension of a [block](https://gpuweb.github.io/gpuweb/#texel-block) of texels.
    ///
    /// Uncompressed formats have a block dimension of `(1, 1)`.
    #[must_use]
    pub fn block_dimensions(&self) -> (u32, u32) {
        match *self {
            Self::R8Unorm
            | Self::R8Snorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Unorm
            | Self::R16Snorm
            | Self::R16Float
            | Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rg16Float
            | Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Rgb9e5Ufloat
            | Self::Rgb10a2Uint
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Ufloat
            | Self::R64Uint
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba16Float
            | Self::Rgba32Uint
            | Self::Rgba32Sint
            | Self::Rgba32Float
            | Self::Stencil8
            | Self::Depth16Unorm
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8
            | Self::Depth32Float
            | Self::Depth32FloatStencil8
            | Self::NV12
            | Self::P010 => (1, 1),

            Self::Bc1RgbaUnorm
            | Self::Bc1RgbaUnormSrgb
            | Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb => (4, 4),

            Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm => (4, 4),

            Self::Astc { block, .. } => match block {
                AstcBlock::B4x4 => (4, 4),
                AstcBlock::B5x4 => (5, 4),
                AstcBlock::B5x5 => (5, 5),
                AstcBlock::B6x5 => (6, 5),
                AstcBlock::B6x6 => (6, 6),
                AstcBlock::B8x5 => (8, 5),
                AstcBlock::B8x6 => (8, 6),
                AstcBlock::B8x8 => (8, 8),
                AstcBlock::B10x5 => (10, 5),
                AstcBlock::B10x6 => (10, 6),
                AstcBlock::B10x8 => (10, 8),
                AstcBlock::B10x10 => (10, 10),
                AstcBlock::B12x10 => (12, 10),
                AstcBlock::B12x12 => (12, 12),
            },
        }
    }

    /// Returns `true` for compressed formats.
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.block_dimensions() != (1, 1)
    }

    /// Returns `true` for BCn compressed formats.
    #[must_use]
    pub fn is_bcn(&self) -> bool {
        self.required_features() == Features::TEXTURE_COMPRESSION_BC
    }

    /// Returns `true` for ASTC compressed formats.
    #[must_use]
    pub fn is_astc(&self) -> bool {
        self.required_features() == Features::TEXTURE_COMPRESSION_ASTC
            || self.required_features() == Features::TEXTURE_COMPRESSION_ASTC_HDR
    }

    /// Returns the required features (if any) in order to use the texture.
    #[must_use]
    pub fn required_features(&self) -> Features {
        match *self {
            Self::R8Unorm
            | Self::R8Snorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Float
            | Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float
            | Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Rgb9e5Ufloat
            | Self::Rgb10a2Uint
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Ufloat
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float
            | Self::Rgba32Uint
            | Self::Rgba32Sint
            | Self::Rgba32Float
            | Self::Stencil8
            | Self::Depth16Unorm
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8
            | Self::Depth32Float => Features::empty(),

            Self::R64Uint => Features::TEXTURE_INT64_ATOMIC,

            Self::Depth32FloatStencil8 => Features::DEPTH32FLOAT_STENCIL8,

            Self::NV12 => Features::TEXTURE_FORMAT_NV12,
            Self::P010 => Features::TEXTURE_FORMAT_P010,

            Self::R16Unorm
            | Self::R16Snorm
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm => Features::TEXTURE_FORMAT_16BIT_NORM,

            Self::Bc1RgbaUnorm
            | Self::Bc1RgbaUnormSrgb
            | Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb => Features::TEXTURE_COMPRESSION_BC,

            Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm => Features::TEXTURE_COMPRESSION_ETC2,

            Self::Astc { channel, .. } => match channel {
                AstcChannel::Hdr => Features::TEXTURE_COMPRESSION_ASTC_HDR,
                AstcChannel::Unorm | AstcChannel::UnormSrgb => Features::TEXTURE_COMPRESSION_ASTC,
            },
        }
    }

    /// Returns the format features guaranteed by the WebGPU spec.
    ///
    /// Additional features are available if `Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` is enabled.
    #[must_use]
    pub fn guaranteed_format_features(&self, device_features: Features) -> TextureFormatFeatures {
        // Multisampling
        let none = TextureFormatFeatureFlags::empty();
        let msaa = TextureFormatFeatureFlags::MULTISAMPLE_X4;
        let msaa_resolve = msaa | TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE;

        let s_ro_wo = TextureFormatFeatureFlags::STORAGE_READ_ONLY
            | TextureFormatFeatureFlags::STORAGE_WRITE_ONLY;
        let s_all = s_ro_wo | TextureFormatFeatureFlags::STORAGE_READ_WRITE;

        // Flags
        let basic =
            TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
        let attachment = basic | TextureUsages::RENDER_ATTACHMENT;
        let storage = basic | TextureUsages::STORAGE_BINDING;
        let binding = TextureUsages::TEXTURE_BINDING;
        let all_flags = attachment | storage | binding;
        let atomic_64 = if device_features.contains(Features::TEXTURE_ATOMIC) {
            storage | binding | TextureUsages::STORAGE_ATOMIC
        } else {
            storage | binding
        };
        let atomic = attachment | atomic_64;
        let (rg11b10f_f, rg11b10f_u) =
            if device_features.contains(Features::RG11B10UFLOAT_RENDERABLE) {
                (msaa_resolve, attachment)
            } else {
                (msaa, basic)
            };
        let (bgra8unorm_f, bgra8unorm) = if device_features.contains(Features::BGRA8UNORM_STORAGE) {
            (
                msaa_resolve | TextureFormatFeatureFlags::STORAGE_WRITE_ONLY,
                attachment | TextureUsages::STORAGE_BINDING,
            )
        } else {
            (msaa_resolve, attachment)
        };

        #[rustfmt::skip] // lets make a nice table
        let (
            mut flags,
            allowed_usages,
        ) = match *self {
            Self::R8Unorm =>              (msaa_resolve, attachment),
            Self::R8Snorm =>              (        none,      basic),
            Self::R8Uint =>               (        msaa, attachment),
            Self::R8Sint =>               (        msaa, attachment),
            Self::R16Uint =>              (        msaa, attachment),
            Self::R16Sint =>              (        msaa, attachment),
            Self::R16Float =>             (msaa_resolve, attachment),
            Self::Rg8Unorm =>             (msaa_resolve, attachment),
            Self::Rg8Snorm =>             (        none,      basic),
            Self::Rg8Uint =>              (        msaa, attachment),
            Self::Rg8Sint =>              (        msaa, attachment),
            Self::R32Uint =>              (       s_all,     atomic),
            Self::R32Sint =>              (       s_all,     atomic),
            Self::R32Float =>             (msaa | s_all,  all_flags),
            Self::Rg16Uint =>             (        msaa, attachment),
            Self::Rg16Sint =>             (        msaa, attachment),
            Self::Rg16Float =>            (msaa_resolve, attachment),
            Self::Rgba8Unorm =>           (msaa_resolve | s_ro_wo,  all_flags),
            Self::Rgba8UnormSrgb =>       (msaa_resolve, attachment),
            Self::Rgba8Snorm =>           (     s_ro_wo,    storage),
            Self::Rgba8Uint =>            (        msaa | s_ro_wo,  all_flags),
            Self::Rgba8Sint =>            (        msaa | s_ro_wo,  all_flags),
            Self::Bgra8Unorm =>           (bgra8unorm_f, bgra8unorm),
            Self::Bgra8UnormSrgb =>       (msaa_resolve, attachment),
            Self::Rgb10a2Uint =>          (        msaa, attachment),
            Self::Rgb10a2Unorm =>         (msaa_resolve, attachment),
            Self::Rg11b10Ufloat =>        (  rg11b10f_f, rg11b10f_u),
            Self::R64Uint =>              (     s_ro_wo,  atomic_64),
            Self::Rg32Uint =>             (     s_ro_wo,  all_flags),
            Self::Rg32Sint =>             (     s_ro_wo,  all_flags),
            Self::Rg32Float =>            (     s_ro_wo,  all_flags),
            Self::Rgba16Uint =>           (        msaa | s_ro_wo,  all_flags),
            Self::Rgba16Sint =>           (        msaa | s_ro_wo,  all_flags),
            Self::Rgba16Float =>          (msaa_resolve | s_ro_wo,  all_flags),
            Self::Rgba32Uint =>           (     s_ro_wo,  all_flags),
            Self::Rgba32Sint =>           (     s_ro_wo,  all_flags),
            Self::Rgba32Float =>          (     s_ro_wo,  all_flags),

            Self::Stencil8 =>             (        msaa, attachment),
            Self::Depth16Unorm =>         (        msaa, attachment),
            Self::Depth24Plus =>          (        msaa, attachment),
            Self::Depth24PlusStencil8 =>  (        msaa, attachment),
            Self::Depth32Float =>         (        msaa, attachment),
            Self::Depth32FloatStencil8 => (        msaa, attachment),

            // We only support sampling nv12 and p010 textures until we
            // implement transfer plane data.
            Self::NV12 =>                 (        none,    binding),
            Self::P010 =>                 (        none,    binding),

            Self::R16Unorm =>             (        msaa | s_ro_wo,    storage),
            Self::R16Snorm =>             (        msaa | s_ro_wo,    storage),
            Self::Rg16Unorm =>            (        msaa | s_ro_wo,    storage),
            Self::Rg16Snorm =>            (        msaa | s_ro_wo,    storage),
            Self::Rgba16Unorm =>          (        msaa | s_ro_wo,    storage),
            Self::Rgba16Snorm =>          (        msaa | s_ro_wo,    storage),

            Self::Rgb9e5Ufloat =>         (        none,      basic),

            Self::Bc1RgbaUnorm =>         (        none,      basic),
            Self::Bc1RgbaUnormSrgb =>     (        none,      basic),
            Self::Bc2RgbaUnorm =>         (        none,      basic),
            Self::Bc2RgbaUnormSrgb =>     (        none,      basic),
            Self::Bc3RgbaUnorm =>         (        none,      basic),
            Self::Bc3RgbaUnormSrgb =>     (        none,      basic),
            Self::Bc4RUnorm =>            (        none,      basic),
            Self::Bc4RSnorm =>            (        none,      basic),
            Self::Bc5RgUnorm =>           (        none,      basic),
            Self::Bc5RgSnorm =>           (        none,      basic),
            Self::Bc6hRgbUfloat =>        (        none,      basic),
            Self::Bc6hRgbFloat =>         (        none,      basic),
            Self::Bc7RgbaUnorm =>         (        none,      basic),
            Self::Bc7RgbaUnormSrgb =>     (        none,      basic),

            Self::Etc2Rgb8Unorm =>        (        none,      basic),
            Self::Etc2Rgb8UnormSrgb =>    (        none,      basic),
            Self::Etc2Rgb8A1Unorm =>      (        none,      basic),
            Self::Etc2Rgb8A1UnormSrgb =>  (        none,      basic),
            Self::Etc2Rgba8Unorm =>       (        none,      basic),
            Self::Etc2Rgba8UnormSrgb =>   (        none,      basic),
            Self::EacR11Unorm =>          (        none,      basic),
            Self::EacR11Snorm =>          (        none,      basic),
            Self::EacRg11Unorm =>         (        none,      basic),
            Self::EacRg11Snorm =>         (        none,      basic),

            Self::Astc { .. } =>          (        none,      basic),
        };

        // Get whether the format is filterable, taking features into account
        let sample_type1 = self.sample_type(None, Some(device_features));
        let is_filterable = sample_type1 == Some(TextureSampleType::Float { filterable: true });

        // Features that enable filtering don't affect blendability
        let sample_type2 = self.sample_type(None, None);
        let is_blendable = sample_type2 == Some(TextureSampleType::Float { filterable: true });

        flags.set(TextureFormatFeatureFlags::FILTERABLE, is_filterable);
        flags.set(TextureFormatFeatureFlags::BLENDABLE, is_blendable);
        flags.set(
            TextureFormatFeatureFlags::STORAGE_ATOMIC,
            allowed_usages.contains(TextureUsages::STORAGE_ATOMIC),
        );

        TextureFormatFeatures {
            allowed_usages,
            flags,
        }
    }

    /// Returns the sample type compatible with this format and aspect.
    ///
    /// Returns `None` only if this is a combined depth-stencil format or a multi-planar format
    /// and `TextureAspect::All` or no `aspect` was provided.
    #[must_use]
    pub fn sample_type(
        &self,
        aspect: Option<TextureAspect>,
        device_features: Option<Features>,
    ) -> Option<TextureSampleType> {
        let float = TextureSampleType::Float { filterable: true };
        let unfilterable_float = TextureSampleType::Float { filterable: false };
        let float32_sample_type = TextureSampleType::Float {
            filterable: device_features
                .unwrap_or(Features::empty())
                .contains(Features::FLOAT32_FILTERABLE),
        };
        let depth = TextureSampleType::Depth;
        let uint = TextureSampleType::Uint;
        let sint = TextureSampleType::Sint;

        match *self {
            Self::R8Unorm
            | Self::R8Snorm
            | Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::R16Float
            | Self::Rg16Float
            | Self::Rgba16Float
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Ufloat => Some(float),

            Self::R32Float | Self::Rg32Float | Self::Rgba32Float => Some(float32_sample_type),

            Self::R8Uint
            | Self::Rg8Uint
            | Self::Rgba8Uint
            | Self::R16Uint
            | Self::Rg16Uint
            | Self::Rgba16Uint
            | Self::R32Uint
            | Self::R64Uint
            | Self::Rg32Uint
            | Self::Rgba32Uint
            | Self::Rgb10a2Uint => Some(uint),

            Self::R8Sint
            | Self::Rg8Sint
            | Self::Rgba8Sint
            | Self::R16Sint
            | Self::Rg16Sint
            | Self::Rgba16Sint
            | Self::R32Sint
            | Self::Rg32Sint
            | Self::Rgba32Sint => Some(sint),

            Self::Stencil8 => Some(uint),
            Self::Depth16Unorm | Self::Depth24Plus | Self::Depth32Float => Some(depth),
            Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8 => match aspect {
                Some(TextureAspect::DepthOnly) => Some(depth),
                Some(TextureAspect::StencilOnly) => Some(uint),
                _ => None,
            },

            Self::NV12 | Self::P010 => match aspect {
                Some(TextureAspect::Plane0) | Some(TextureAspect::Plane1) => {
                    Some(unfilterable_float)
                }
                _ => None,
            },

            Self::R16Unorm
            | Self::R16Snorm
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm => Some(float),

            Self::Rgb9e5Ufloat => Some(float),

            Self::Bc1RgbaUnorm
            | Self::Bc1RgbaUnormSrgb
            | Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb => Some(float),

            Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm => Some(float),

            Self::Astc { .. } => Some(float),
        }
    }

    /// The number of bytes one [texel block](https://gpuweb.github.io/gpuweb/#texel-block) occupies during an image copy, if applicable.
    ///
    /// Known as the [texel block copy footprint](https://gpuweb.github.io/gpuweb/#texel-block-copy-footprint).
    ///
    /// Note that for uncompressed formats this is the same as the size of a single texel,
    /// since uncompressed formats have a block size of 1x1.
    ///
    /// Returns `None` if any of the following are true:
    ///  - the format is a combined depth-stencil and no `aspect` was provided
    ///  - the format is a multi-planar format and no `aspect` was provided
    ///  - the format is `Depth24Plus`
    ///  - the format is `Depth24PlusStencil8` and `aspect` is depth.
    #[deprecated(since = "0.19.0", note = "Use `block_copy_size` instead.")]
    #[must_use]
    pub fn block_size(&self, aspect: Option<TextureAspect>) -> Option<u32> {
        self.block_copy_size(aspect)
    }

    /// The number of bytes one [texel block](https://gpuweb.github.io/gpuweb/#texel-block) occupies during an image copy, if applicable.
    ///
    /// Known as the [texel block copy footprint](https://gpuweb.github.io/gpuweb/#texel-block-copy-footprint).
    ///
    /// Note that for uncompressed formats this is the same as the size of a single texel,
    /// since uncompressed formats have a block size of 1x1.
    ///
    /// Returns `None` if any of the following are true:
    ///  - the format is a combined depth-stencil and no `aspect` was provided
    ///  - the format is a multi-planar format and no `aspect` was provided
    ///  - the format is `Depth24Plus`
    ///  - the format is `Depth24PlusStencil8` and `aspect` is depth.
    #[must_use]
    pub fn block_copy_size(&self, aspect: Option<TextureAspect>) -> Option<u32> {
        match *self {
            Self::R8Unorm | Self::R8Snorm | Self::R8Uint | Self::R8Sint => Some(1),

            Self::Rg8Unorm | Self::Rg8Snorm | Self::Rg8Uint | Self::Rg8Sint => Some(2),
            Self::R16Unorm | Self::R16Snorm | Self::R16Uint | Self::R16Sint | Self::R16Float => {
                Some(2)
            }

            Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb => Some(4),
            Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float => Some(4),
            Self::R32Uint | Self::R32Sint | Self::R32Float => Some(4),
            Self::Rgb9e5Ufloat | Self::Rgb10a2Uint | Self::Rgb10a2Unorm | Self::Rg11b10Ufloat => {
                Some(4)
            }

            Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float => Some(8),
            Self::R64Uint | Self::Rg32Uint | Self::Rg32Sint | Self::Rg32Float => Some(8),

            Self::Rgba32Uint | Self::Rgba32Sint | Self::Rgba32Float => Some(16),

            Self::Stencil8 => Some(1),
            Self::Depth16Unorm => Some(2),
            Self::Depth32Float => Some(4),
            Self::Depth24Plus => None,
            Self::Depth24PlusStencil8 => match aspect {
                Some(TextureAspect::DepthOnly) => None,
                Some(TextureAspect::StencilOnly) => Some(1),
                _ => None,
            },
            Self::Depth32FloatStencil8 => match aspect {
                Some(TextureAspect::DepthOnly) => Some(4),
                Some(TextureAspect::StencilOnly) => Some(1),
                _ => None,
            },

            Self::NV12 => match aspect {
                Some(TextureAspect::Plane0) => Some(1),
                Some(TextureAspect::Plane1) => Some(2),
                _ => None,
            },

            Self::P010 => match aspect {
                Some(TextureAspect::Plane0) => Some(2),
                Some(TextureAspect::Plane1) => Some(4),
                _ => None,
            },

            Self::Bc1RgbaUnorm | Self::Bc1RgbaUnormSrgb | Self::Bc4RUnorm | Self::Bc4RSnorm => {
                Some(8)
            }
            Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb => Some(16),

            Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm => Some(8),
            Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm => Some(16),

            Self::Astc { .. } => Some(16),
        }
    }

    /// The largest number that can be returned by [`Self::target_pixel_byte_cost`].
    pub const MAX_TARGET_PIXEL_BYTE_COST: u32 = 16;

    /// The number of bytes occupied per pixel in a color attachment
    /// <https://gpuweb.github.io/gpuweb/#render-target-pixel-byte-cost>
    #[must_use]
    pub fn target_pixel_byte_cost(&self) -> Option<u32> {
        match *self {
            Self::R8Unorm | Self::R8Snorm | Self::R8Uint | Self::R8Sint => Some(1),
            Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Unorm
            | Self::R16Snorm
            | Self::R16Float => Some(2),
            Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rg16Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float => Some(4),
            // Despite being 4 bytes per pixel, these are 8 bytes per pixel in the table
            Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            // ---
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba16Float
            | Self::R64Uint
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Rgb10a2Uint
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Ufloat => Some(8),
            Self::Rgba32Uint | Self::Rgba32Sint | Self::Rgba32Float => Some(16),
            // ⚠️ If you add formats with larger sizes, make sure you change `MAX_TARGET_PIXEL_BYTE_COST`` ⚠️
            Self::Stencil8
            | Self::Depth16Unorm
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8
            | Self::Depth32Float
            | Self::Depth32FloatStencil8
            | Self::NV12
            | Self::P010
            | Self::Rgb9e5Ufloat
            | Self::Bc1RgbaUnorm
            | Self::Bc1RgbaUnormSrgb
            | Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb
            | Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm
            | Self::Astc { .. } => None,
        }
    }

    /// See <https://gpuweb.github.io/gpuweb/#render-target-component-alignment>
    #[must_use]
    pub fn target_component_alignment(&self) -> Option<u32> {
        match *self {
            Self::R8Unorm
            | Self::R8Snorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb => Some(1),
            Self::R16Uint
            | Self::R16Sint
            | Self::R16Unorm
            | Self::R16Snorm
            | Self::R16Float
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rg16Float
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba16Float => Some(2),
            Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::R64Uint
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float
            | Self::Rgba32Uint
            | Self::Rgba32Sint
            | Self::Rgba32Float
            | Self::Rgb10a2Uint
            | Self::Rgb10a2Unorm
            | Self::Rg11b10Ufloat => Some(4),
            Self::Stencil8
            | Self::Depth16Unorm
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8
            | Self::Depth32Float
            | Self::Depth32FloatStencil8
            | Self::NV12
            | Self::P010
            | Self::Rgb9e5Ufloat
            | Self::Bc1RgbaUnorm
            | Self::Bc1RgbaUnormSrgb
            | Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb
            | Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm
            | Self::Astc { .. } => None,
        }
    }

    /// Returns the number of components this format has.
    #[must_use]
    pub fn components(&self) -> u8 {
        self.components_with_aspect(TextureAspect::All)
    }

    /// Returns the number of components this format has taking into account the `aspect`.
    ///
    /// The `aspect` is only relevant for combined depth-stencil formats and multi-planar formats.
    #[must_use]
    pub fn components_with_aspect(&self, aspect: TextureAspect) -> u8 {
        match *self {
            Self::R8Unorm
            | Self::R8Snorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::R16Unorm
            | Self::R16Snorm
            | Self::R16Uint
            | Self::R16Sint
            | Self::R16Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::R64Uint => 1,

            Self::Rg8Unorm
            | Self::Rg8Snorm
            | Self::Rg8Uint
            | Self::Rg8Sint
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rg16Uint
            | Self::Rg16Sint
            | Self::Rg16Float
            | Self::Rg32Uint
            | Self::Rg32Sint
            | Self::Rg32Float => 2,

            Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Rgba8Snorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Rgba16Unorm
            | Self::Rgba16Snorm
            | Self::Rgba16Uint
            | Self::Rgba16Sint
            | Self::Rgba16Float
            | Self::Rgba32Uint
            | Self::Rgba32Sint
            | Self::Rgba32Float => 4,

            Self::Rgb9e5Ufloat | Self::Rg11b10Ufloat => 3,
            Self::Rgb10a2Uint | Self::Rgb10a2Unorm => 4,

            Self::Stencil8 | Self::Depth16Unorm | Self::Depth24Plus | Self::Depth32Float => 1,

            Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8 => match aspect {
                TextureAspect::DepthOnly | TextureAspect::StencilOnly => 1,
                _ => 2,
            },

            Self::NV12 | Self::P010 => match aspect {
                TextureAspect::Plane0 => 1,
                TextureAspect::Plane1 => 2,
                _ => 3,
            },

            Self::Bc4RUnorm | Self::Bc4RSnorm => 1,
            Self::Bc5RgUnorm | Self::Bc5RgSnorm => 2,
            Self::Bc6hRgbUfloat | Self::Bc6hRgbFloat => 3,
            Self::Bc1RgbaUnorm
            | Self::Bc1RgbaUnormSrgb
            | Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb => 4,

            Self::EacR11Unorm | Self::EacR11Snorm => 1,
            Self::EacRg11Unorm | Self::EacRg11Snorm => 2,
            Self::Etc2Rgb8Unorm | Self::Etc2Rgb8UnormSrgb => 3,
            Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb => 4,

            Self::Astc { .. } => 4,
        }
    }

    /// Strips the `Srgb` suffix from the given texture format.
    #[must_use]
    pub fn remove_srgb_suffix(&self) -> TextureFormat {
        match *self {
            Self::Rgba8UnormSrgb => Self::Rgba8Unorm,
            Self::Bgra8UnormSrgb => Self::Bgra8Unorm,
            Self::Bc1RgbaUnormSrgb => Self::Bc1RgbaUnorm,
            Self::Bc2RgbaUnormSrgb => Self::Bc2RgbaUnorm,
            Self::Bc3RgbaUnormSrgb => Self::Bc3RgbaUnorm,
            Self::Bc7RgbaUnormSrgb => Self::Bc7RgbaUnorm,
            Self::Etc2Rgb8UnormSrgb => Self::Etc2Rgb8Unorm,
            Self::Etc2Rgb8A1UnormSrgb => Self::Etc2Rgb8A1Unorm,
            Self::Etc2Rgba8UnormSrgb => Self::Etc2Rgba8Unorm,
            Self::Astc {
                block,
                channel: AstcChannel::UnormSrgb,
            } => Self::Astc {
                block,
                channel: AstcChannel::Unorm,
            },
            _ => *self,
        }
    }

    /// Adds an `Srgb` suffix to the given texture format, if the format supports it.
    #[must_use]
    pub fn add_srgb_suffix(&self) -> TextureFormat {
        match *self {
            Self::Rgba8Unorm => Self::Rgba8UnormSrgb,
            Self::Bgra8Unorm => Self::Bgra8UnormSrgb,
            Self::Bc1RgbaUnorm => Self::Bc1RgbaUnormSrgb,
            Self::Bc2RgbaUnorm => Self::Bc2RgbaUnormSrgb,
            Self::Bc3RgbaUnorm => Self::Bc3RgbaUnormSrgb,
            Self::Bc7RgbaUnorm => Self::Bc7RgbaUnormSrgb,
            Self::Etc2Rgb8Unorm => Self::Etc2Rgb8UnormSrgb,
            Self::Etc2Rgb8A1Unorm => Self::Etc2Rgb8A1UnormSrgb,
            Self::Etc2Rgba8Unorm => Self::Etc2Rgba8UnormSrgb,
            Self::Astc {
                block,
                channel: AstcChannel::Unorm,
            } => Self::Astc {
                block,
                channel: AstcChannel::UnormSrgb,
            },
            _ => *self,
        }
    }

    /// Returns `true` for srgb formats.
    #[must_use]
    pub fn is_srgb(&self) -> bool {
        *self != self.remove_srgb_suffix()
    }

    /// Returns the theoretical memory footprint of a texture with the given format and dimensions.
    ///
    /// Actual memory usage may greatly exceed this value due to alignment and padding.
    #[must_use]
    pub fn theoretical_memory_footprint(&self, size: Extent3d) -> u64 {
        let (block_width, block_height) = self.block_dimensions();

        let block_size = self.block_copy_size(None);

        let approximate_block_size = match block_size {
            Some(size) => size,
            None => match self {
                // One f16 per pixel
                Self::Depth16Unorm => 2,
                // One u24 per pixel, padded to 4 bytes
                Self::Depth24Plus => 4,
                // One u24 per pixel, plus one u8 per pixel
                Self::Depth24PlusStencil8 => 4,
                // One f32 per pixel
                Self::Depth32Float => 4,
                // One f32 per pixel, plus one u8 per pixel, with 3 bytes intermediary padding
                Self::Depth32FloatStencil8 => 8,
                // One u8 per pixel
                Self::Stencil8 => 1,
                // Two chroma bytes per block, one luma byte per block
                Self::NV12 => 3,
                // Two chroma u16s and one luma u16 per block
                Self::P010 => 6,
                f => {
                    log::warn!("Memory footprint for format {f:?} is not implemented");
                    0
                }
            },
        };

        let width_blocks = size.width.div_ceil(block_width) as u64;
        let height_blocks = size.height.div_ceil(block_height) as u64;

        let total_blocks = width_blocks * height_blocks * size.depth_or_array_layers as u64;

        total_blocks * approximate_block_size as u64
    }
}

#[test]
fn texture_format_serialize() {
    use alloc::string::ToString;

    assert_eq!(
        serde_json::to_string(&TextureFormat::R8Unorm).unwrap(),
        "\"r8unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R8Snorm).unwrap(),
        "\"r8snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R8Uint).unwrap(),
        "\"r8uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R8Sint).unwrap(),
        "\"r8sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R16Uint).unwrap(),
        "\"r16uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R16Sint).unwrap(),
        "\"r16sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R16Unorm).unwrap(),
        "\"r16unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R16Snorm).unwrap(),
        "\"r16snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R16Float).unwrap(),
        "\"r16float\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg8Unorm).unwrap(),
        "\"rg8unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg8Snorm).unwrap(),
        "\"rg8snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg8Uint).unwrap(),
        "\"rg8uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg8Sint).unwrap(),
        "\"rg8sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R32Uint).unwrap(),
        "\"r32uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R32Sint).unwrap(),
        "\"r32sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R32Float).unwrap(),
        "\"r32float\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg16Uint).unwrap(),
        "\"rg16uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg16Sint).unwrap(),
        "\"rg16sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg16Unorm).unwrap(),
        "\"rg16unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg16Snorm).unwrap(),
        "\"rg16snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg16Float).unwrap(),
        "\"rg16float\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba8Unorm).unwrap(),
        "\"rgba8unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba8UnormSrgb).unwrap(),
        "\"rgba8unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba8Snorm).unwrap(),
        "\"rgba8snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba8Uint).unwrap(),
        "\"rgba8uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba8Sint).unwrap(),
        "\"rgba8sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bgra8Unorm).unwrap(),
        "\"bgra8unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bgra8UnormSrgb).unwrap(),
        "\"bgra8unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgb10a2Uint).unwrap(),
        "\"rgb10a2uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgb10a2Unorm).unwrap(),
        "\"rgb10a2unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg11b10Ufloat).unwrap(),
        "\"rg11b10ufloat\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::R64Uint).unwrap(),
        "\"r64uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg32Uint).unwrap(),
        "\"rg32uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg32Sint).unwrap(),
        "\"rg32sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rg32Float).unwrap(),
        "\"rg32float\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba16Uint).unwrap(),
        "\"rgba16uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba16Sint).unwrap(),
        "\"rgba16sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba16Unorm).unwrap(),
        "\"rgba16unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba16Snorm).unwrap(),
        "\"rgba16snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba16Float).unwrap(),
        "\"rgba16float\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba32Uint).unwrap(),
        "\"rgba32uint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba32Sint).unwrap(),
        "\"rgba32sint\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgba32Float).unwrap(),
        "\"rgba32float\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Stencil8).unwrap(),
        "\"stencil8\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Depth32Float).unwrap(),
        "\"depth32float\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Depth16Unorm).unwrap(),
        "\"depth16unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Depth32FloatStencil8).unwrap(),
        "\"depth32float-stencil8\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Depth24Plus).unwrap(),
        "\"depth24plus\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Depth24PlusStencil8).unwrap(),
        "\"depth24plus-stencil8\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Rgb9e5Ufloat).unwrap(),
        "\"rgb9e5ufloat\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc1RgbaUnorm).unwrap(),
        "\"bc1-rgba-unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc1RgbaUnormSrgb).unwrap(),
        "\"bc1-rgba-unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc2RgbaUnorm).unwrap(),
        "\"bc2-rgba-unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc2RgbaUnormSrgb).unwrap(),
        "\"bc2-rgba-unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc3RgbaUnorm).unwrap(),
        "\"bc3-rgba-unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc3RgbaUnormSrgb).unwrap(),
        "\"bc3-rgba-unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc4RUnorm).unwrap(),
        "\"bc4-r-unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc4RSnorm).unwrap(),
        "\"bc4-r-snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc5RgUnorm).unwrap(),
        "\"bc5-rg-unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc5RgSnorm).unwrap(),
        "\"bc5-rg-snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc6hRgbUfloat).unwrap(),
        "\"bc6h-rgb-ufloat\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc6hRgbFloat).unwrap(),
        "\"bc6h-rgb-float\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc7RgbaUnorm).unwrap(),
        "\"bc7-rgba-unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Bc7RgbaUnormSrgb).unwrap(),
        "\"bc7-rgba-unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Etc2Rgb8Unorm).unwrap(),
        "\"etc2-rgb8unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Etc2Rgb8UnormSrgb).unwrap(),
        "\"etc2-rgb8unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Etc2Rgb8A1Unorm).unwrap(),
        "\"etc2-rgb8a1unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Etc2Rgb8A1UnormSrgb).unwrap(),
        "\"etc2-rgb8a1unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Etc2Rgba8Unorm).unwrap(),
        "\"etc2-rgba8unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::Etc2Rgba8UnormSrgb).unwrap(),
        "\"etc2-rgba8unorm-srgb\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::EacR11Unorm).unwrap(),
        "\"eac-r11unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::EacR11Snorm).unwrap(),
        "\"eac-r11snorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::EacRg11Unorm).unwrap(),
        "\"eac-rg11unorm\"".to_string()
    );
    assert_eq!(
        serde_json::to_string(&TextureFormat::EacRg11Snorm).unwrap(),
        "\"eac-rg11snorm\"".to_string()
    );
}

#[test]
fn texture_format_deserialize() {
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r8unorm\"").unwrap(),
        TextureFormat::R8Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r8snorm\"").unwrap(),
        TextureFormat::R8Snorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r8uint\"").unwrap(),
        TextureFormat::R8Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r8sint\"").unwrap(),
        TextureFormat::R8Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r16uint\"").unwrap(),
        TextureFormat::R16Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r16sint\"").unwrap(),
        TextureFormat::R16Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r16unorm\"").unwrap(),
        TextureFormat::R16Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r16snorm\"").unwrap(),
        TextureFormat::R16Snorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r16float\"").unwrap(),
        TextureFormat::R16Float
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg8unorm\"").unwrap(),
        TextureFormat::Rg8Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg8snorm\"").unwrap(),
        TextureFormat::Rg8Snorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg8uint\"").unwrap(),
        TextureFormat::Rg8Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg8sint\"").unwrap(),
        TextureFormat::Rg8Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r32uint\"").unwrap(),
        TextureFormat::R32Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r32sint\"").unwrap(),
        TextureFormat::R32Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r32float\"").unwrap(),
        TextureFormat::R32Float
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg16uint\"").unwrap(),
        TextureFormat::Rg16Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg16sint\"").unwrap(),
        TextureFormat::Rg16Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg16unorm\"").unwrap(),
        TextureFormat::Rg16Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg16snorm\"").unwrap(),
        TextureFormat::Rg16Snorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg16float\"").unwrap(),
        TextureFormat::Rg16Float
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba8unorm\"").unwrap(),
        TextureFormat::Rgba8Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba8unorm-srgb\"").unwrap(),
        TextureFormat::Rgba8UnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba8snorm\"").unwrap(),
        TextureFormat::Rgba8Snorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba8uint\"").unwrap(),
        TextureFormat::Rgba8Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba8sint\"").unwrap(),
        TextureFormat::Rgba8Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bgra8unorm\"").unwrap(),
        TextureFormat::Bgra8Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bgra8unorm-srgb\"").unwrap(),
        TextureFormat::Bgra8UnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgb10a2uint\"").unwrap(),
        TextureFormat::Rgb10a2Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgb10a2unorm\"").unwrap(),
        TextureFormat::Rgb10a2Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg11b10ufloat\"").unwrap(),
        TextureFormat::Rg11b10Ufloat
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"r64uint\"").unwrap(),
        TextureFormat::R64Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg32uint\"").unwrap(),
        TextureFormat::Rg32Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg32sint\"").unwrap(),
        TextureFormat::Rg32Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rg32float\"").unwrap(),
        TextureFormat::Rg32Float
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba16uint\"").unwrap(),
        TextureFormat::Rgba16Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba16sint\"").unwrap(),
        TextureFormat::Rgba16Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba16unorm\"").unwrap(),
        TextureFormat::Rgba16Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba16snorm\"").unwrap(),
        TextureFormat::Rgba16Snorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba16float\"").unwrap(),
        TextureFormat::Rgba16Float
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba32uint\"").unwrap(),
        TextureFormat::Rgba32Uint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba32sint\"").unwrap(),
        TextureFormat::Rgba32Sint
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgba32float\"").unwrap(),
        TextureFormat::Rgba32Float
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"stencil8\"").unwrap(),
        TextureFormat::Stencil8
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"depth32float\"").unwrap(),
        TextureFormat::Depth32Float
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"depth16unorm\"").unwrap(),
        TextureFormat::Depth16Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"depth32float-stencil8\"").unwrap(),
        TextureFormat::Depth32FloatStencil8
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"depth24plus\"").unwrap(),
        TextureFormat::Depth24Plus
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"depth24plus-stencil8\"").unwrap(),
        TextureFormat::Depth24PlusStencil8
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"rgb9e5ufloat\"").unwrap(),
        TextureFormat::Rgb9e5Ufloat
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc1-rgba-unorm\"").unwrap(),
        TextureFormat::Bc1RgbaUnorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc1-rgba-unorm-srgb\"").unwrap(),
        TextureFormat::Bc1RgbaUnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc2-rgba-unorm\"").unwrap(),
        TextureFormat::Bc2RgbaUnorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc2-rgba-unorm-srgb\"").unwrap(),
        TextureFormat::Bc2RgbaUnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc3-rgba-unorm\"").unwrap(),
        TextureFormat::Bc3RgbaUnorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc3-rgba-unorm-srgb\"").unwrap(),
        TextureFormat::Bc3RgbaUnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc4-r-unorm\"").unwrap(),
        TextureFormat::Bc4RUnorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc4-r-snorm\"").unwrap(),
        TextureFormat::Bc4RSnorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc5-rg-unorm\"").unwrap(),
        TextureFormat::Bc5RgUnorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc5-rg-snorm\"").unwrap(),
        TextureFormat::Bc5RgSnorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc6h-rgb-ufloat\"").unwrap(),
        TextureFormat::Bc6hRgbUfloat
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc6h-rgb-float\"").unwrap(),
        TextureFormat::Bc6hRgbFloat
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc7-rgba-unorm\"").unwrap(),
        TextureFormat::Bc7RgbaUnorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"bc7-rgba-unorm-srgb\"").unwrap(),
        TextureFormat::Bc7RgbaUnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"etc2-rgb8unorm\"").unwrap(),
        TextureFormat::Etc2Rgb8Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"etc2-rgb8unorm-srgb\"").unwrap(),
        TextureFormat::Etc2Rgb8UnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"etc2-rgb8a1unorm\"").unwrap(),
        TextureFormat::Etc2Rgb8A1Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"etc2-rgb8a1unorm-srgb\"").unwrap(),
        TextureFormat::Etc2Rgb8A1UnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"etc2-rgba8unorm\"").unwrap(),
        TextureFormat::Etc2Rgba8Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"etc2-rgba8unorm-srgb\"").unwrap(),
        TextureFormat::Etc2Rgba8UnormSrgb
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"eac-r11unorm\"").unwrap(),
        TextureFormat::EacR11Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"eac-r11snorm\"").unwrap(),
        TextureFormat::EacR11Snorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"eac-rg11unorm\"").unwrap(),
        TextureFormat::EacRg11Unorm
    );
    assert_eq!(
        serde_json::from_str::<TextureFormat>("\"eac-rg11snorm\"").unwrap(),
        TextureFormat::EacRg11Snorm
    );
}

/// Color write mask. Disabled color channels will not be written to.
///
/// Corresponds to [WebGPU `GPUColorWriteFlags`](
/// https://gpuweb.github.io/gpuweb/#typedefdef-gpucolorwriteflags).
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ColorWrites(u32);

bitflags::bitflags! {
    impl ColorWrites: u32 {
        /// Enable red channel writes
        const RED = 1 << 0;
        /// Enable green channel writes
        const GREEN = 1 << 1;
        /// Enable blue channel writes
        const BLUE = 1 << 2;
        /// Enable alpha channel writes
        const ALPHA = 1 << 3;
        /// Enable red, green, and blue channel writes
        const COLOR = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits();
        /// Enable writes to all channels.
        const ALL = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits() | Self::ALPHA.bits();
    }
}

impl Default for ColorWrites {
    fn default() -> Self {
        Self::ALL
    }
}

/// Passed to `Device::poll` to control how and if it should block.
#[derive(Clone, Debug)]
pub enum PollType<T> {
    /// On wgpu-core based backends, block until the given submission has
    /// completed execution, and any callbacks have been invoked.
    ///
    /// On WebGPU, this has no effect. Callbacks are invoked from the
    /// window event loop.
    WaitForSubmissionIndex(T),
    /// Same as `WaitForSubmissionIndex` but waits for the most recent submission.
    Wait,
    /// Check the device for a single time without blocking.
    Poll,
}

impl<T> PollType<T> {
    /// Construct a [`Self::Wait`] variant
    #[must_use]
    pub fn wait() -> Self {
        // This function seems a little silly, but it is useful to allow
        // <https://github.com/gfx-rs/wgpu/pull/5012> to be split up, as
        // it has meaning in that PR.
        Self::Wait
    }

    /// Construct a [`Self::WaitForSubmissionIndex`] variant
    #[must_use]
    pub fn wait_for(submission_index: T) -> Self {
        // This function seems a little silly, but it is useful to allow
        // <https://github.com/gfx-rs/wgpu/pull/5012> to be split up, as
        // it has meaning in that PR.
        Self::WaitForSubmissionIndex(submission_index)
    }

    /// This `PollType` represents a wait of some kind.
    #[must_use]
    pub fn is_wait(&self) -> bool {
        match *self {
            Self::WaitForSubmissionIndex(..) | Self::Wait => true,
            Self::Poll => false,
        }
    }

    /// Map on the wait index type.
    #[must_use]
    pub fn map_index<U, F>(self, func: F) -> PollType<U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Self::WaitForSubmissionIndex(i) => PollType::WaitForSubmissionIndex(func(i)),
            Self::Wait => PollType::Wait,
            Self::Poll => PollType::Poll,
        }
    }
}

/// Error states after a device poll
#[derive(Debug)]
#[cfg_attr(feature = "std", derive(thiserror::Error))]
pub enum PollError {
    /// The requested Wait timed out before the submission was completed.
    #[cfg_attr(
        feature = "std",
        error("The requested Wait timed out before the submission was completed.")
    )]
    Timeout,
    /// The requested Wait was given a wrong submission index.
    #[cfg_attr(
        feature = "std",
        error("Tried to wait using a submission index ({0}) that has not been returned by a successful submission (last successful submission: {1})")
    )]
    WrongSubmissionIndex(u64, u64),
}

/// Status of device poll operation.
#[derive(Debug, PartialEq, Eq)]
pub enum PollStatus {
    /// There are no active submissions in flight as of the beginning of the poll call.
    /// Other submissions may have been queued on other threads during the call.
    ///
    /// This implies that the given Wait was satisfied before the timeout.
    QueueEmpty,

    /// The requested Wait was satisfied before the timeout.
    WaitSucceeded,

    /// This was a poll.
    Poll,
}

impl PollStatus {
    /// Returns true if the result is [`Self::QueueEmpty`].
    #[must_use]
    pub fn is_queue_empty(&self) -> bool {
        matches!(self, Self::QueueEmpty)
    }

    /// Returns true if the result is either [`Self::WaitSucceeded`] or [`Self::QueueEmpty`].
    #[must_use]
    pub fn wait_finished(&self) -> bool {
        matches!(self, Self::WaitSucceeded | Self::QueueEmpty)
    }
}

/// State of the stencil operation (fixed-pipeline stage).
///
/// For use in [`DepthStencilState`].
///
/// Corresponds to a portion of [WebGPU `GPUDepthStencilState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpudepthstencilstate).
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        (self.front != StencilFaceState::IGNORE || self.back != StencilFaceState::IGNORE)
            && (self.read_mask != 0 || self.write_mask != 0)
    }
    /// Returns true if the state doesn't mutate the target values.
    #[must_use]
    pub fn is_read_only(&self, cull_mode: Option<Face>) -> bool {
        // The rules are defined in step 7 of the "Device timeline initialization steps"
        // subsection of the "Render Pipeline Creation" section of WebGPU
        // (link to the section: https://gpuweb.github.io/gpuweb/#render-pipeline-creation)

        if self.write_mask == 0 {
            return true;
        }

        let front_ro = cull_mode == Some(Face::Front) || self.front.is_read_only();
        let back_ro = cull_mode == Some(Face::Back) || self.back.is_read_only();

        front_ro && back_ro
    }
    /// Returns true if the stencil state uses the reference value for testing.
    #[must_use]
    pub fn needs_ref_value(&self) -> bool {
        self.front.needs_ref_value() || self.back.needs_ref_value()
    }
}

/// Describes the biasing setting for the depth target.
///
/// For use in [`DepthStencilState`].
///
/// Corresponds to a portion of [WebGPU `GPUDepthStencilState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpudepthstencilstate).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.constant != 0 || self.slope_scale != 0.0
    }
}

impl Hash for DepthBiasState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.constant.hash(state);
        self.slope_scale.to_bits().hash(state);
        self.clamp.to_bits().hash(state);
    }
}

impl PartialEq for DepthBiasState {
    fn eq(&self, other: &Self) -> bool {
        (self.constant == other.constant)
            && (self.slope_scale.to_bits() == other.slope_scale.to_bits())
            && (self.clamp.to_bits() == other.clamp.to_bits())
    }
}

impl Eq for DepthBiasState {}

/// Operation to perform to the output attachment at the start of a render pass.
///
/// Corresponds to [WebGPU `GPULoadOp`](https://gpuweb.github.io/gpuweb/#enumdef-gpuloadop),
/// plus the corresponding clearValue.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum LoadOp<V> {
    /// Loads the specified value for this attachment into the render pass.
    ///
    /// On some GPU hardware (primarily mobile), "clear" is significantly cheaper
    /// because it avoids loading data from main memory into tile-local memory.
    ///
    /// On other GPU hardware, there isn’t a significant difference.
    ///
    /// As a result, it is recommended to use "clear" rather than "load" in cases
    /// where the initial value doesn’t matter
    /// (e.g. the render target will be cleared using a skybox).
    Clear(V) = 0,
    /// Loads the existing value for this attachment into the render pass.
    Load = 1,
}

impl<V> LoadOp<V> {
    /// Returns true if variants are same (ignoring clear value)
    pub fn eq_variant<T>(&self, other: LoadOp<T>) -> bool {
        matches!(
            (self, other),
            (LoadOp::Clear(_), LoadOp::Clear(_)) | (LoadOp::Load, LoadOp::Load)
        )
    }
}

impl<V: Default> Default for LoadOp<V> {
    fn default() -> Self {
        Self::Clear(Default::default())
    }
}

/// Operation to perform to the output attachment at the end of a render pass.
///
/// Corresponds to [WebGPU `GPUStoreOp`](https://gpuweb.github.io/gpuweb/#enumdef-gpustoreop).
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum StoreOp {
    /// Stores the resulting value of the render pass for this attachment.
    #[default]
    Store = 0,
    /// Discards the resulting value of the render pass for this attachment.
    ///
    /// The attachment will be treated as uninitialized afterwards.
    /// (If only either Depth or Stencil texture-aspects is set to `Discard`,
    /// the respective other texture-aspect will be preserved.)
    ///
    /// This can be significantly faster on tile-based render hardware.
    ///
    /// Prefer this if the attachment is not read by subsequent passes.
    Discard = 1,
}

/// Pair of load and store operations for an attachment aspect.
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// separate `loadOp` and `storeOp` fields are used instead.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Operations<V> {
    /// How data should be read through this attachment.
    pub load: LoadOp<V>,
    /// Whether data will be written to through this attachment.
    ///
    /// Note that resolve textures (if specified) are always written to,
    /// regardless of this setting.
    pub store: StoreOp,
}

impl<V: Default> Default for Operations<V> {
    #[inline]
    fn default() -> Self {
        Self {
            load: LoadOp::<V>::default(),
            store: StoreOp::default(),
        }
    }
}

/// Describes the depth/stencil state in a render pipeline.
///
/// Corresponds to [WebGPU `GPUDepthStencilState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpudepthstencilstate).
#[repr(C)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DepthStencilState {
    /// Format of the depth/stencil buffer, must be special depth format. Must match the format
    /// of the depth/stencil attachment in [`CommandEncoder::begin_render_pass`][CEbrp].
    ///
    /// [CEbrp]: ../wgpu/struct.CommandEncoder.html#method.begin_render_pass
    pub format: TextureFormat,
    /// If disabled, depth will not be written to.
    pub depth_write_enabled: bool,
    /// Comparison function used to compare depth values in the depth test.
    pub depth_compare: CompareFunction,
    /// Stencil state.
    #[cfg_attr(feature = "serde", serde(default))]
    pub stencil: StencilState,
    /// Depth bias state.
    #[cfg_attr(feature = "serde", serde(default))]
    pub bias: DepthBiasState,
}

impl DepthStencilState {
    /// Returns true if the depth testing is enabled.
    #[must_use]
    pub fn is_depth_enabled(&self) -> bool {
        self.depth_compare != CompareFunction::Always || self.depth_write_enabled
    }

    /// Returns true if the state doesn't mutate the depth buffer.
    #[must_use]
    pub fn is_depth_read_only(&self) -> bool {
        !self.depth_write_enabled
    }

    /// Returns true if the state doesn't mutate the stencil.
    #[must_use]
    pub fn is_stencil_read_only(&self, cull_mode: Option<Face>) -> bool {
        self.stencil.is_read_only(cull_mode)
    }

    /// Returns true if the state doesn't mutate either depth or stencil of the target.
    #[must_use]
    pub fn is_read_only(&self, cull_mode: Option<Face>) -> bool {
        self.is_depth_read_only() && self.is_stencil_read_only(cull_mode)
    }
}

/// Format of indices used with pipeline.
///
/// Corresponds to [WebGPU `GPUIndexFormat`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpuindexformat).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum IndexFormat {
    /// Indices are 16 bit unsigned integers.
    Uint16 = 0,
    /// Indices are 32 bit unsigned integers.
    #[default]
    Uint32 = 1,
}

impl IndexFormat {
    /// Returns the size in bytes of the index format
    pub fn byte_size(&self) -> usize {
        match self {
            IndexFormat::Uint16 => 2,
            IndexFormat::Uint32 => 4,
        }
    }
}

/// Operation to perform on the stencil value.
///
/// Corresponds to [WebGPU `GPUStencilOperation`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpustenciloperation).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum StencilOperation {
    /// Keep stencil value unchanged.
    #[default]
    Keep = 0,
    /// Set stencil value to zero.
    Zero = 1,
    /// Replace stencil value with value provided in most recent call to
    /// [`RenderPass::set_stencil_reference`][RPssr].
    ///
    /// [RPssr]: ../wgpu/struct.RenderPass.html#method.set_stencil_reference
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

/// Describes stencil state in a render pipeline.
///
/// If you are not using stencil state, set this to [`StencilFaceState::IGNORE`].
///
/// Corresponds to [WebGPU `GPUStencilFaceState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpustencilfacestate).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct StencilFaceState {
    /// Comparison function that determines if the fail_op or pass_op is used on the stencil buffer.
    pub compare: CompareFunction,
    /// Operation that is performed when stencil test fails.
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
    #[must_use]
    pub fn needs_ref_value(&self) -> bool {
        self.compare.needs_ref_value()
            || self.fail_op == StencilOperation::Replace
            || self.depth_fail_op == StencilOperation::Replace
            || self.pass_op == StencilOperation::Replace
    }

    /// Returns true if the face state doesn't mutate the target values.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.pass_op == StencilOperation::Keep
            && self.depth_fail_op == StencilOperation::Keep
            && self.fail_op == StencilOperation::Keep
    }
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self::IGNORE
    }
}

/// Comparison function used for depth and stencil operations.
///
/// Corresponds to [WebGPU `GPUCompareFunction`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpucomparefunction).
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum CompareFunction {
    /// Function never passes
    Never = 1,
    /// Function passes if new value less than existing value
    Less = 2,
    /// Function passes if new value is equal to existing value. When using
    /// this compare function, make sure to mark your Vertex Shader's `@builtin(position)`
    /// output as `@invariant` to prevent artifacting.
    Equal = 3,
    /// Function passes if new value is less than or equal to existing value
    LessEqual = 4,
    /// Function passes if new value is greater than existing value
    Greater = 5,
    /// Function passes if new value is not equal to existing value. When using
    /// this compare function, make sure to mark your Vertex Shader's `@builtin(position)`
    /// output as `@invariant` to prevent artifacting.
    NotEqual = 6,
    /// Function passes if new value is greater than or equal to existing value
    GreaterEqual = 7,
    /// Function always passes
    Always = 8,
}

impl CompareFunction {
    /// Returns true if the comparison depends on the reference value.
    #[must_use]
    pub fn needs_ref_value(self) -> bool {
        match self {
            Self::Never | Self::Always => false,
            _ => true,
        }
    }
}

/// Whether a vertex buffer is indexed by vertex or by instance.
///
/// Consider a call to [`RenderPass::draw`] like this:
///
/// ```ignore
/// render_pass.draw(vertices, instances)
/// ```
///
/// where `vertices` is a `Range<u32>` of vertex indices, and
/// `instances` is a `Range<u32>` of instance indices.
///
/// For this call, `wgpu` invokes the vertex shader entry point once
/// for every possible `(v, i)` pair, where `v` is drawn from
/// `vertices` and `i` is drawn from `instances`. These invocations
/// may happen in any order, and will usually run in parallel.
///
/// Each vertex buffer has a step mode, established by the
/// [`step_mode`] field of its [`VertexBufferLayout`], given when the
/// pipeline was created. Buffers whose step mode is [`Vertex`] use
/// `v` as the index into their contents, whereas buffers whose step
/// mode is [`Instance`] use `i`. The indicated buffer element then
/// contributes zero or more attribute values for the `(v, i)` vertex
/// shader invocation to use, based on the [`VertexBufferLayout`]'s
/// [`attributes`] list.
///
/// You can visualize the results from all these vertex shader
/// invocations as a matrix with a row for each `i` from `instances`,
/// and with a column for each `v` from `vertices`. In one sense, `v`
/// and `i` are symmetrical: both are used to index vertex buffers and
/// provide attribute values.  But the key difference between `v` and
/// `i` is that line and triangle primitives are built from the values
/// of each row, along which `i` is constant and `v` varies, not the
/// columns.
///
/// An indexed draw call works similarly:
///
/// ```ignore
/// render_pass.draw_indexed(indices, base_vertex, instances)
/// ```
///
/// The only difference is that `v` values are drawn from the contents
/// of the index buffer&mdash;specifically, the subrange of the index
/// buffer given by `indices`&mdash;instead of simply being sequential
/// integers, as they are in a `draw` call.
///
/// A non-instanced call, where `instances` is `0..1`, is simply a
/// matrix with only one row.
///
/// Corresponds to [WebGPU `GPUVertexStepMode`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpuvertexstepmode).
///
/// [`RenderPass::draw`]: ../wgpu/struct.RenderPass.html#method.draw
/// [`VertexBufferLayout`]: ../wgpu/struct.VertexBufferLayout.html
/// [`step_mode`]: ../wgpu/struct.VertexBufferLayout.html#structfield.step_mode
/// [`attributes`]: ../wgpu/struct.VertexBufferLayout.html#structfield.attributes
/// [`Vertex`]: VertexStepMode::Vertex
/// [`Instance`]: VertexStepMode::Instance
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum VertexStepMode {
    /// Vertex data is advanced every vertex.
    #[default]
    Vertex = 0,
    /// Vertex data is advanced every instance.
    Instance = 1,
}

/// Vertex inputs (attributes) to shaders.
///
/// These are used to specify the individual attributes within a [`VertexBufferLayout`].
/// See its documentation for an example.
///
/// The [`vertex_attr_array!`] macro can help create these with appropriate offsets.
///
/// Corresponds to [WebGPU `GPUVertexAttribute`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuvertexattribute).
///
/// [`vertex_attr_array!`]: ../wgpu/macro.vertex_attr_array.html
/// [`VertexBufferLayout`]: ../wgpu/struct.VertexBufferLayout.html
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct VertexAttribute {
    /// Format of the input
    pub format: VertexFormat,
    /// Byte offset of the start of the input
    pub offset: BufferAddress,
    /// Location for this input. Must match the location in the shader.
    pub shader_location: ShaderLocation,
}

/// Vertex Format for a [`VertexAttribute`] (input).
///
/// Corresponds to [WebGPU `GPUVertexFormat`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpuvertexformat).
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum VertexFormat {
    /// One unsigned byte (u8). `u32` in shaders.
    Uint8 = 0,
    /// Two unsigned bytes (u8). `vec2<u32>` in shaders.
    Uint8x2 = 1,
    /// Four unsigned bytes (u8). `vec4<u32>` in shaders.
    Uint8x4 = 2,
    /// One signed byte (i8). `i32` in shaders.
    Sint8 = 3,
    /// Two signed bytes (i8). `vec2<i32>` in shaders.
    Sint8x2 = 4,
    /// Four signed bytes (i8). `vec4<i32>` in shaders.
    Sint8x4 = 5,
    /// One unsigned byte (u8). [0, 255] converted to float [0, 1] `f32` in shaders.
    Unorm8 = 6,
    /// Two unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec2<f32>` in shaders.
    Unorm8x2 = 7,
    /// Four unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec4<f32>` in shaders.
    Unorm8x4 = 8,
    /// One signed byte (i8). [&minus;127, 127] converted to float [&minus;1, 1] `f32` in shaders.
    Snorm8 = 9,
    /// Two signed bytes (i8). [&minus;127, 127] converted to float [&minus;1, 1] `vec2<f32>` in shaders.
    Snorm8x2 = 10,
    /// Four signed bytes (i8). [&minus;127, 127] converted to float [&minus;1, 1] `vec4<f32>` in shaders.
    Snorm8x4 = 11,
    /// One unsigned short (u16). `u32` in shaders.
    Uint16 = 12,
    /// Two unsigned shorts (u16). `vec2<u32>` in shaders.
    Uint16x2 = 13,
    /// Four unsigned shorts (u16). `vec4<u32>` in shaders.
    Uint16x4 = 14,
    /// One signed short (u16). `i32` in shaders.
    Sint16 = 15,
    /// Two signed shorts (i16). `vec2<i32>` in shaders.
    Sint16x2 = 16,
    /// Four signed shorts (i16). `vec4<i32>` in shaders.
    Sint16x4 = 17,
    /// One unsigned short (u16). [0, 65535] converted to float [0, 1] `f32` in shaders.
    Unorm16 = 18,
    /// Two unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec2<f32>` in shaders.
    Unorm16x2 = 19,
    /// Four unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec4<f32>` in shaders.
    Unorm16x4 = 20,
    /// One signed short (i16). [&minus;32767, 32767] converted to float [&minus;1, 1] `f32` in shaders.
    Snorm16 = 21,
    /// Two signed shorts (i16). [&minus;32767, 32767] converted to float [&minus;1, 1] `vec2<f32>` in shaders.
    Snorm16x2 = 22,
    /// Four signed shorts (i16). [&minus;32767, 32767] converted to float [&minus;1, 1] `vec4<f32>` in shaders.
    Snorm16x4 = 23,
    /// One half-precision float (no Rust equiv). `f32` in shaders.
    Float16 = 24,
    /// Two half-precision floats (no Rust equiv). `vec2<f32>` in shaders.
    Float16x2 = 25,
    /// Four half-precision floats (no Rust equiv). `vec4<f32>` in shaders.
    Float16x4 = 26,
    /// One single-precision float (f32). `f32` in shaders.
    Float32 = 27,
    /// Two single-precision floats (f32). `vec2<f32>` in shaders.
    Float32x2 = 28,
    /// Three single-precision floats (f32). `vec3<f32>` in shaders.
    Float32x3 = 29,
    /// Four single-precision floats (f32). `vec4<f32>` in shaders.
    Float32x4 = 30,
    /// One unsigned int (u32). `u32` in shaders.
    Uint32 = 31,
    /// Two unsigned ints (u32). `vec2<u32>` in shaders.
    Uint32x2 = 32,
    /// Three unsigned ints (u32). `vec3<u32>` in shaders.
    Uint32x3 = 33,
    /// Four unsigned ints (u32). `vec4<u32>` in shaders.
    Uint32x4 = 34,
    /// One signed int (i32). `i32` in shaders.
    Sint32 = 35,
    /// Two signed ints (i32). `vec2<i32>` in shaders.
    Sint32x2 = 36,
    /// Three signed ints (i32). `vec3<i32>` in shaders.
    Sint32x3 = 37,
    /// Four signed ints (i32). `vec4<i32>` in shaders.
    Sint32x4 = 38,
    /// One double-precision float (f64). `f32` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64 = 39,
    /// Two double-precision floats (f64). `vec2<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x2 = 40,
    /// Three double-precision floats (f64). `vec3<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x3 = 41,
    /// Four double-precision floats (f64). `vec4<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x4 = 42,
    /// Three unsigned 10-bit integers and one 2-bit integer, packed into a 32-bit integer (u32). [0, 1024] converted to float [0, 1] `vec4<f32>` in shaders.
    #[cfg_attr(feature = "serde", serde(rename = "unorm10-10-10-2"))]
    Unorm10_10_10_2 = 43,
    /// Four unsigned 8-bit integers, packed into a 32-bit integer (u32). [0, 255] converted to float [0, 1] `vec4<f32>` in shaders.
    #[cfg_attr(feature = "serde", serde(rename = "unorm8x4-bgra"))]
    Unorm8x4Bgra = 44,
}

impl VertexFormat {
    /// Returns the byte size of the format.
    #[must_use]
    pub const fn size(&self) -> u64 {
        match self {
            Self::Uint8 | Self::Sint8 | Self::Unorm8 | Self::Snorm8 => 1,
            Self::Uint8x2
            | Self::Sint8x2
            | Self::Unorm8x2
            | Self::Snorm8x2
            | Self::Uint16
            | Self::Sint16
            | Self::Unorm16
            | Self::Snorm16
            | Self::Float16 => 2,
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
            | Self::Sint32
            | Self::Unorm10_10_10_2
            | Self::Unorm8x4Bgra => 4,
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

    /// Returns the size read by an acceleration structure build of the vertex format. This is
    /// slightly different from [`Self::size`] because the alpha component of 4-component formats
    /// are not read in an acceleration structure build, allowing for a smaller stride.
    #[must_use]
    pub const fn min_acceleration_structure_vertex_stride(&self) -> u64 {
        match self {
            Self::Float16x2 | Self::Snorm16x2 => 4,
            Self::Float32x3 => 12,
            Self::Float32x2 => 8,
            // This is the minimum value from DirectX
            // > A16 component is ignored, other data can be packed there, such as setting vertex stride to 6 bytes
            //
            // https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#d3d12_raytracing_geometry_triangles_desc
            //
            // Vulkan does not express a minimum stride.
            Self::Float16x4 | Self::Snorm16x4 => 6,
            _ => unreachable!(),
        }
    }

    /// Returns the alignment required for `wgpu::BlasTriangleGeometry::vertex_stride`
    #[must_use]
    pub const fn acceleration_structure_stride_alignment(&self) -> u64 {
        match self {
            Self::Float16x4 | Self::Float16x2 | Self::Snorm16x4 | Self::Snorm16x2 => 2,
            Self::Float32x2 | Self::Float32x3 => 4,
            _ => unreachable!(),
        }
    }
}

bitflags::bitflags! {
    /// Different ways that you can use a buffer.
    ///
    /// The usages determine what kind of memory the buffer is allocated from and what
    /// actions the buffer can partake in.
    ///
    /// Specifying only usages the application will actually perform may increase performance.
    /// Additionally, on the WebGL backend, there are restrictions on [`BufferUsages::INDEX`];
    /// see [`DownlevelFlags::UNRESTRICTED_INDEX_BUFFER`] for more information.
    ///
    /// Corresponds to [WebGPU `GPUBufferUsageFlags`](
    /// https://gpuweb.github.io/gpuweb/#typedefdef-gpubufferusageflags).
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct BufferUsages: u32 {
        /// Allow a buffer to be mapped for reading using [`Buffer::map_async`] + [`Buffer::get_mapped_range`].
        /// This does not include creating a buffer with [`BufferDescriptor::mapped_at_creation`] set.
        ///
        /// If [`Features::MAPPABLE_PRIMARY_BUFFERS`] isn't enabled, the only other usage a buffer
        /// may have is COPY_DST.
        const MAP_READ = 1 << 0;
        /// Allow a buffer to be mapped for writing using [`Buffer::map_async`] + [`Buffer::get_mapped_range_mut`].
        /// This does not include creating a buffer with [`BufferDescriptor::mapped_at_creation`] set.
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
        /// Allow a buffer to be the destination buffer for a [`CommandEncoder::resolve_query_set`] operation.
        const QUERY_RESOLVE = 1 << 9;
        /// Allows a buffer to be used as input for a bottom level acceleration structure build
        const BLAS_INPUT = 1 << 10;
        /// Allows a buffer to be used as input for a top level acceleration structure build
        const TLAS_INPUT = 1 << 11;
    }
}

bitflags::bitflags! {
    /// Similar to `BufferUsages`, but used only for `CommandEncoder::transition_resources`.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct BufferUses: u16 {
        /// The argument to a read-only mapping.
        const MAP_READ = 1 << 0;
        /// The argument to a write-only mapping.
        const MAP_WRITE = 1 << 1;
        /// The source of a hardware copy.
        /// cbindgen:ignore
        const COPY_SRC = 1 << 2;
        /// The destination of a hardware copy.
        /// cbindgen:ignore
        const COPY_DST = 1 << 3;
        /// The index buffer used for drawing.
        const INDEX = 1 << 4;
        /// A vertex buffer used for drawing.
        const VERTEX = 1 << 5;
        /// A uniform buffer bound in a bind group.
        const UNIFORM = 1 << 6;
        /// A read-only storage buffer used in a bind group.
        /// cbindgen:ignore
        const STORAGE_READ_ONLY = 1 << 7;
        /// A read-write buffer used in a bind group.
        /// cbindgen:ignore
        const STORAGE_READ_WRITE = 1 << 8;
        /// The indirect or count buffer in a indirect draw or dispatch.
        const INDIRECT = 1 << 9;
        /// A buffer used to store query results.
        const QUERY_RESOLVE = 1 << 10;
        /// Buffer used for acceleration structure building.
        const ACCELERATION_STRUCTURE_SCRATCH = 1 << 11;
        /// Buffer used for bottom level acceleration structure building.
        const BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT = 1 << 12;
        /// Buffer used for top level acceleration structure building.
        const TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT = 1 << 13;
        /// A buffer used to store the compacted size of an acceleration structure
        const ACCELERATION_STRUCTURE_QUERY = 1 << 14;
        /// The combination of states that a buffer may be in _at the same time_.
        const INCLUSIVE = Self::MAP_READ.bits() | Self::COPY_SRC.bits() |
            Self::INDEX.bits() | Self::VERTEX.bits() | Self::UNIFORM.bits() |
            Self::STORAGE_READ_ONLY.bits() | Self::INDIRECT.bits() | Self::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT.bits() | Self::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT.bits();
        /// The combination of states that a buffer must exclusively be in.
        const EXCLUSIVE = Self::MAP_WRITE.bits() | Self::COPY_DST.bits() | Self::STORAGE_READ_WRITE.bits() | Self::ACCELERATION_STRUCTURE_SCRATCH.bits();
        /// The combination of all usages that the are guaranteed to be be ordered by the hardware.
        /// If a usage is ordered, then if the buffer state doesn't change between draw calls, there
        /// are no barriers needed for synchronization.
        const ORDERED = Self::INCLUSIVE.bits() | Self::MAP_WRITE.bits();
    }
}

/// A buffer transition for use with `CommandEncoder::transition_resources`.
#[derive(Debug)]
pub struct BufferTransition<T> {
    /// The buffer to transition.
    pub buffer: T,
    /// The new state to transition to.
    pub state: BufferUses,
}

/// Describes a [`Buffer`](../wgpu/struct.Buffer.html).
///
/// Corresponds to [WebGPU `GPUBufferDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubufferdescriptor).
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BufferDescriptor<L> {
    /// Debug label of a buffer. This will show up in graphics debuggers for easy identification.
    pub label: L,
    /// Size of a buffer, in bytes.
    pub size: BufferAddress,
    /// Usages of a buffer. If the buffer is used in any way that isn't specified here, the operation
    /// will panic.
    ///
    /// Specifying only usages the application will actually perform may increase performance.
    /// Additionally, on the WebGL backend, there are restrictions on [`BufferUsages::INDEX`];
    /// see [`DownlevelFlags::UNRESTRICTED_INDEX_BUFFER`] for more information.
    pub usage: BufferUsages,
    /// Allows a buffer to be mapped immediately after they are made. It does not have to be [`BufferUsages::MAP_READ`] or
    /// [`BufferUsages::MAP_WRITE`], all buffers are allowed to be mapped at creation.
    ///
    /// If this is `true`, [`size`](#structfield.size) must be a multiple of
    /// [`COPY_BUFFER_ALIGNMENT`].
    pub mapped_at_creation: bool,
}

impl<L> BufferDescriptor<L> {
    /// Takes a closure and maps the label of the buffer descriptor into another.
    #[must_use]
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> BufferDescriptor<K> {
        BufferDescriptor {
            label: fun(&self.label),
            size: self.size,
            usage: self.usage,
            mapped_at_creation: self.mapped_at_creation,
        }
    }
}

/// Describes a [`CommandEncoder`](../wgpu/struct.CommandEncoder.html).
///
/// Corresponds to [WebGPU `GPUCommandEncoderDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucommandencoderdescriptor).
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommandEncoderDescriptor<L> {
    /// Debug label for the command encoder. This will show up in graphics debuggers for easy identification.
    pub label: L,
}

impl<L> CommandEncoderDescriptor<L> {
    /// Takes a closure and maps the label of the command encoder descriptor into another.
    #[must_use]
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

/// Timing and queueing with which frames are actually displayed to the user.
///
/// Use this as part of a [`SurfaceConfiguration`] to control the behavior of
/// [`SurfaceTexture::present()`].
///
/// Some modes are only supported by some backends.
/// You can use one of the `Auto*` modes, [`Fifo`](Self::Fifo),
/// or choose one of the supported modes from [`SurfaceCapabilities::present_modes`].
///
/// [presented]: ../wgpu/struct.SurfaceTexture.html#method.present
/// [`SurfaceTexture::present()`]: ../wgpu/struct.SurfaceTexture.html#method.present
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PresentMode {
    /// Chooses the first supported mode out of:
    ///
    /// 1. [`FifoRelaxed`](Self::FifoRelaxed)
    /// 2. [`Fifo`](Self::Fifo)
    ///
    /// Because of the fallback behavior, this is supported everywhere.
    AutoVsync = 0,

    /// Chooses the first supported mode out of:
    ///
    /// 1. [`Immediate`](Self::Immediate)
    /// 2. [`Mailbox`](Self::Mailbox)
    /// 3. [`Fifo`](Self::Fifo)
    ///
    /// Because of the fallback behavior, this is supported everywhere.
    AutoNoVsync = 1,

    /// Presentation frames are kept in a First-In-First-Out queue approximately 3 frames
    /// long. Every vertical blanking period, the presentation engine will pop a frame
    /// off the queue to display. If there is no frame to display, it will present the same
    /// frame again until the next vblank.
    ///
    /// When a present command is executed on the GPU, the presented image is added on the queue.
    ///
    /// Calls to [`Surface::get_current_texture()`] will block until there is a spot in the queue.
    ///
    /// * **Tearing:** No tearing will be observed.
    /// * **Supported on**: All platforms.
    /// * **Also known as**: "Vsync On"
    ///
    /// This is the [default](Self::default) value for `PresentMode`.
    /// If you don't know what mode to choose, choose this mode.
    ///
    /// [`Surface::get_current_texture()`]: ../wgpu/struct.Surface.html#method.get_current_texture
    #[default]
    Fifo = 2,

    /// Presentation frames are kept in a First-In-First-Out queue approximately 3 frames
    /// long. Every vertical blanking period, the presentation engine will pop a frame
    /// off the queue to display. If there is no frame to display, it will present the
    /// same frame until there is a frame in the queue. The moment there is a frame in the
    /// queue, it will immediately pop the frame off the queue.
    ///
    /// When a present command is executed on the GPU, the presented image is added on the queue.
    ///
    /// Calls to [`Surface::get_current_texture()`] will block until there is a spot in the queue.
    ///
    /// * **Tearing**:
    ///   Tearing will be observed if frames last more than one vblank as the front buffer.
    /// * **Supported on**: AMD on Vulkan.
    /// * **Also known as**: "Adaptive Vsync"
    ///
    /// [`Surface::get_current_texture()`]: ../wgpu/struct.Surface.html#method.get_current_texture
    FifoRelaxed = 3,

    /// Presentation frames are not queued at all. The moment a present command
    /// is executed on the GPU, the presented image is swapped onto the front buffer
    /// immediately.
    ///
    /// * **Tearing**: Tearing can be observed.
    /// * **Supported on**: Most platforms except older DX12 and Wayland.
    /// * **Also known as**: "Vsync Off"
    Immediate = 4,

    /// Presentation frames are kept in a single-frame queue. Every vertical blanking period,
    /// the presentation engine will pop a frame from the queue. If there is no frame to display,
    /// it will present the same frame again until the next vblank.
    ///
    /// When a present command is executed on the GPU, the frame will be put into the queue.
    /// If there was already a frame in the queue, the new frame will _replace_ the old frame
    /// on the queue.
    ///
    /// * **Tearing**: No tearing will be observed.
    /// * **Supported on**: DX12 on Windows 10, NVidia on Vulkan and Wayland on Vulkan.
    /// * **Also known as**: "Fast Vsync"
    Mailbox = 5,
}

/// Specifies how the alpha channel of the textures should be handled during
/// compositing.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum CompositeAlphaMode {
    /// Chooses either `Opaque` or `Inherit` automatically，depending on the
    /// `alpha_mode` that the current surface can support.
    Auto = 0,
    /// The alpha channel, if it exists, of the textures is ignored in the
    /// compositing process. Instead, the textures is treated as if it has a
    /// constant alpha of 1.0.
    Opaque = 1,
    /// The alpha channel, if it exists, of the textures is respected in the
    /// compositing process. The non-alpha channels of the textures are
    /// expected to already be multiplied by the alpha channel by the
    /// application.
    PreMultiplied = 2,
    /// The alpha channel, if it exists, of the textures is respected in the
    /// compositing process. The non-alpha channels of the textures are not
    /// expected to already be multiplied by the alpha channel by the
    /// application; instead, the compositor will multiply the non-alpha
    /// channels of the texture by the alpha channel during compositing.
    PostMultiplied = 3,
    /// The alpha channel, if it exists, of the textures is unknown for processing
    /// during compositing. Instead, the application is responsible for setting
    /// the composite alpha blending mode using native WSI command. If not set,
    /// then a platform-specific default will be used.
    Inherit = 4,
}

impl Default for CompositeAlphaMode {
    fn default() -> Self {
        Self::Auto
    }
}

bitflags::bitflags! {
    /// Different ways that you can use a texture.
    ///
    /// The usages determine what kind of memory the texture is allocated from and what
    /// actions the texture can partake in.
    ///
    /// Corresponds to [WebGPU `GPUTextureUsageFlags`](
    /// https://gpuweb.github.io/gpuweb/#typedefdef-gputextureusageflags).
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct TextureUsages: u32 {
        //
        // ---- Start numbering at 1 << 0 ----
        //
        // WebGPU features:
        //
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
        /// Allows a texture to be an output attachment of a render pass.
        const RENDER_ATTACHMENT = 1 << 4;

        //
        // ---- Restart Numbering for Native Features ---
        //
        // Native Features:
        //
        /// Allows a texture to be used with image atomics. Requires [`Features::TEXTURE_ATOMIC`].
        const STORAGE_ATOMIC = 1 << 16;
    }
}

bitflags::bitflags! {
    /// Similar to `TextureUsages`, but used only for `CommandEncoder::transition_resources`.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct TextureUses: u16 {
        /// The texture is in unknown state.
        const UNINITIALIZED = 1 << 0;
        /// Ready to present image to the surface.
        const PRESENT = 1 << 1;
        /// The source of a hardware copy.
        /// cbindgen:ignore
        const COPY_SRC = 1 << 2;
        /// The destination of a hardware copy.
        /// cbindgen:ignore
        const COPY_DST = 1 << 3;
        /// Read-only sampled or fetched resource.
        const RESOURCE = 1 << 4;
        /// The color target of a renderpass.
        const COLOR_TARGET = 1 << 5;
        /// Read-only depth stencil usage.
        const DEPTH_STENCIL_READ = 1 << 6;
        /// Read-write depth stencil usage
        const DEPTH_STENCIL_WRITE = 1 << 7;
        /// Read-only storage texture usage. Corresponds to a UAV in d3d, so is exclusive, despite being read only.
        /// cbindgen:ignore
        const STORAGE_READ_ONLY = 1 << 8;
        /// Write-only storage texture usage.
        /// cbindgen:ignore
        const STORAGE_WRITE_ONLY = 1 << 9;
        /// Read-write storage texture usage.
        /// cbindgen:ignore
        const STORAGE_READ_WRITE = 1 << 10;
        /// Image atomic enabled storage.
        /// cbindgen:ignore
        const STORAGE_ATOMIC = 1 << 11;
        /// The combination of states that a texture may be in _at the same time_.
        /// cbindgen:ignore
        const INCLUSIVE = Self::COPY_SRC.bits() | Self::RESOURCE.bits() | Self::DEPTH_STENCIL_READ.bits();
        /// The combination of states that a texture must exclusively be in.
        /// cbindgen:ignore
        const EXCLUSIVE = Self::COPY_DST.bits() | Self::COLOR_TARGET.bits() | Self::DEPTH_STENCIL_WRITE.bits() | Self::STORAGE_READ_ONLY.bits() | Self::STORAGE_WRITE_ONLY.bits() | Self::STORAGE_READ_WRITE.bits() | Self::STORAGE_ATOMIC.bits() | Self::PRESENT.bits();
        /// The combination of all usages that the are guaranteed to be be ordered by the hardware.
        /// If a usage is ordered, then if the texture state doesn't change between draw calls, there
        /// are no barriers needed for synchronization.
        /// cbindgen:ignore
        const ORDERED = Self::INCLUSIVE.bits() | Self::COLOR_TARGET.bits() | Self::DEPTH_STENCIL_WRITE.bits() | Self::STORAGE_READ_ONLY.bits();

        /// Flag used by the wgpu-core texture tracker to say a texture is in different states for every sub-resource
        const COMPLEX = 1 << 12;
        /// Flag used by the wgpu-core texture tracker to say that the tracker does not know the state of the sub-resource.
        /// This is different from UNINITIALIZED as that says the tracker does know, but the texture has not been initialized.
        const UNKNOWN = 1 << 13;
    }
}

/// A texture transition for use with `CommandEncoder::transition_resources`.
#[derive(Debug)]
pub struct TextureTransition<T> {
    /// The texture to transition.
    pub texture: T,
    /// An optional selector to transition only part of the texture.
    ///
    /// If None, the entire texture will be transitioned.
    pub selector: Option<TextureSelector>,
    /// The new state to transition to.
    pub state: TextureUses,
}

/// Specifies a particular set of subresources in a texture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureSelector {
    /// Range of mips to use.
    pub mips: Range<u32>,
    /// Range of layers to use.
    pub layers: Range<u32>,
}

/// Defines the capabilities of a given surface and adapter.
#[derive(Debug)]
pub struct SurfaceCapabilities {
    /// List of supported formats to use with the given adapter. The first format in the vector is preferred.
    ///
    /// Returns an empty vector if the surface is incompatible with the adapter.
    pub formats: Vec<TextureFormat>,
    /// List of supported presentation modes to use with the given adapter.
    ///
    /// Returns an empty vector if the surface is incompatible with the adapter.
    pub present_modes: Vec<PresentMode>,
    /// List of supported alpha modes to use with the given adapter.
    ///
    /// Will return at least one element, [`CompositeAlphaMode::Opaque`] or [`CompositeAlphaMode::Inherit`].
    pub alpha_modes: Vec<CompositeAlphaMode>,
    /// Bitflag of supported texture usages for the surface to use with the given adapter.
    ///
    /// The usage [`TextureUsages::RENDER_ATTACHMENT`] is guaranteed.
    pub usages: TextureUsages,
}

impl Default for SurfaceCapabilities {
    fn default() -> Self {
        Self {
            formats: Vec::new(),
            present_modes: Vec::new(),
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        }
    }
}

/// Configures a [`Surface`] for presentation.
///
/// [`Surface`]: ../wgpu/struct.Surface.html
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SurfaceConfiguration<V> {
    /// The usage of the swap chain. The only usage guaranteed to be supported is [`TextureUsages::RENDER_ATTACHMENT`].
    pub usage: TextureUsages,
    /// The texture format of the swap chain. The only formats that are guaranteed are
    /// [`TextureFormat::Bgra8Unorm`] and [`TextureFormat::Bgra8UnormSrgb`].
    pub format: TextureFormat,
    /// Width of the swap chain. Must be the same size as the surface, and nonzero.
    ///
    /// If this is not the same size as the underlying surface (e.g. if it is
    /// set once, and the window is later resized), the behaviour is defined
    /// but platform-specific, and may change in the future (currently macOS
    /// scales the surface, other platforms may do something else).
    pub width: u32,
    /// Height of the swap chain. Must be the same size as the surface, and nonzero.
    ///
    /// If this is not the same size as the underlying surface (e.g. if it is
    /// set once, and the window is later resized), the behaviour is defined
    /// but platform-specific, and may change in the future (currently macOS
    /// scales the surface, other platforms may do something else).
    pub height: u32,
    /// Presentation mode of the swap chain. Fifo is the only mode guaranteed to be supported.
    /// `FifoRelaxed`, `Immediate`, and `Mailbox` will crash if unsupported, while `AutoVsync` and
    /// `AutoNoVsync` will gracefully do a designed sets of fallbacks if their primary modes are
    /// unsupported.
    pub present_mode: PresentMode,
    /// Desired maximum number of frames that the presentation engine should queue in advance.
    ///
    /// This is a hint to the backend implementation and will always be clamped to the supported range.
    /// As a consequence, either the maximum frame latency is set directly on the swap chain,
    /// or waits on present are scheduled to avoid exceeding the maximum frame latency if supported,
    /// or the swap chain size is set to (max-latency + 1).
    ///
    /// Defaults to 2 when created via `Surface::get_default_config`.
    ///
    /// Typical values range from 3 to 1, but higher values are possible:
    /// * Choose 2 or higher for potentially smoother frame display, as it allows to be at least one frame
    ///   to be queued up. This typically avoids starving the GPU's work queue.
    ///   Higher values are useful for achieving a constant flow of frames to the display under varying load.
    /// * Choose 1 for low latency from frame recording to frame display.
    ///   ⚠️ If the backend does not support waiting on present, this will cause the CPU to wait for the GPU
    ///   to finish all work related to the previous frame when calling `Surface::get_current_texture`,
    ///   causing CPU-GPU serialization (i.e. when `Surface::get_current_texture` returns, the GPU might be idle).
    ///   It is currently not possible to query this. See <https://github.com/gfx-rs/wgpu/issues/2869>.
    /// * A value of 0 is generally not supported and always clamped to a higher value.
    pub desired_maximum_frame_latency: u32,
    /// Specifies how the alpha channel of the textures should be handled during compositing.
    pub alpha_mode: CompositeAlphaMode,
    /// Specifies what view formats will be allowed when calling `Texture::create_view` on the texture returned by `Surface::get_current_texture`.
    ///
    /// View formats of the same format as the texture are always allowed.
    ///
    /// Note: currently, only the srgb-ness is allowed to change. (ex: `Rgba8Unorm` texture + `Rgba8UnormSrgb` view)
    pub view_formats: V,
}

impl<V: Clone> SurfaceConfiguration<V> {
    /// Map `view_formats` of the texture descriptor into another.
    pub fn map_view_formats<M>(&self, fun: impl FnOnce(V) -> M) -> SurfaceConfiguration<M> {
        SurfaceConfiguration {
            usage: self.usage,
            format: self.format,
            width: self.width,
            height: self.height,
            present_mode: self.present_mode,
            desired_maximum_frame_latency: self.desired_maximum_frame_latency,
            alpha_mode: self.alpha_mode,
            view_formats: fun(self.view_formats.clone()),
        }
    }
}

/// Status of the received surface image.
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
    /// The surface status is not known since `Surface::get_current_texture` previously failed.
    Unknown,
}

/// Nanosecond timestamp used by the presentation engine.
///
/// The specific clock depends on the window system integration (WSI) API used.
///
/// <table>
/// <tr>
///     <td>WSI</td>
///     <td>Clock</td>
/// </tr>
/// <tr>
///     <td>IDXGISwapchain</td>
///     <td><a href="https://docs.microsoft.com/en-us/windows/win32/api/profileapi/nf-profileapi-queryperformancecounter">QueryPerformanceCounter</a></td>
/// </tr>
/// <tr>
///     <td>IPresentationManager</td>
///     <td><a href="https://docs.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryinterrupttimeprecise">QueryInterruptTimePrecise</a></td>
/// </tr>
/// <tr>
///     <td>CAMetalLayer</td>
///     <td><a href="https://developer.apple.com/documentation/kernel/1462446-mach_absolute_time">mach_absolute_time</a></td>
/// </tr>
/// <tr>
///     <td>VK_GOOGLE_display_timing</td>
///     <td><a href="https://linux.die.net/man/3/clock_gettime">clock_gettime(CLOCK_MONOTONIC)</a></td>
/// </tr>
/// </table>
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PresentationTimestamp(
    /// Timestamp in nanoseconds.
    pub u128,
);

impl PresentationTimestamp {
    /// A timestamp that is invalid due to the platform not having a timestamp system.
    pub const INVALID_TIMESTAMP: Self = Self(u128::MAX);

    /// Returns true if this timestamp is the invalid timestamp.
    #[must_use]
    pub fn is_invalid(self) -> bool {
        self == Self::INVALID_TIMESTAMP
    }
}

/// RGBA double precision color.
///
/// This is not to be used as a generic color type, only for specific wgpu interfaces.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Color {
    /// Red component of the color
    pub r: f64,
    /// Green component of the color
    pub g: f64,
    /// Blue component of the color
    pub b: f64,
    /// Alpha component of the color
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
///
/// Corresponds to [WebGPU `GPUTextureDimension`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputexturedimension).
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

/// Origin of a copy from a 2D image.
///
/// Corresponds to [WebGPU `GPUOrigin2D`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuorigin2ddict).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Origin2d {
    #[allow(missing_docs)]
    pub x: u32,
    #[allow(missing_docs)]
    pub y: u32,
}

impl Origin2d {
    /// Zero origin.
    pub const ZERO: Self = Self { x: 0, y: 0 };

    /// Adds the third dimension to this origin
    #[must_use]
    pub fn to_3d(self, z: u32) -> Origin3d {
        Origin3d {
            x: self.x,
            y: self.y,
            z,
        }
    }
}

impl core::fmt::Debug for Origin2d {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        (self.x, self.y).fmt(f)
    }
}

/// Origin of a copy to/from a texture.
///
/// Corresponds to [WebGPU `GPUOrigin3D`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuorigin3ddict).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Origin3d {
    /// X position of the origin
    pub x: u32,
    /// Y position of the origin
    pub y: u32,
    /// Z position of the origin
    pub z: u32,
}

impl Origin3d {
    /// Zero origin.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    /// Removes the third dimension from this origin
    #[must_use]
    pub fn to_2d(self) -> Origin2d {
        Origin2d {
            x: self.x,
            y: self.y,
        }
    }
}

impl Default for Origin3d {
    fn default() -> Self {
        Self::ZERO
    }
}

impl core::fmt::Debug for Origin3d {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        (self.x, self.y, self.z).fmt(f)
    }
}

/// Extent of a texture related operation.
///
/// Corresponds to [WebGPU `GPUExtent3D`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuextent3ddict).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct Extent3d {
    /// Width of the extent
    pub width: u32,
    /// Height of the extent
    pub height: u32,
    /// The depth of the extent or the number of array layers
    #[cfg_attr(feature = "serde", serde(default = "default_depth"))]
    pub depth_or_array_layers: u32,
}

impl core::fmt::Debug for Extent3d {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        (self.width, self.height, self.depth_or_array_layers).fmt(f)
    }
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
    /// Calculates the [physical size] backing a texture of the given
    /// format and extent.  This includes padding to the block width
    /// and height of the format.
    ///
    /// This is the texture extent that you must upload at when uploading to _mipmaps_ of compressed textures.
    ///
    /// [physical size]: https://gpuweb.github.io/gpuweb/#physical-miplevel-specific-texture-extent
    #[must_use]
    pub fn physical_size(&self, format: TextureFormat) -> Self {
        let (block_width, block_height) = format.block_dimensions();

        let width = self.width.div_ceil(block_width) * block_width;
        let height = self.height.div_ceil(block_height) * block_height;

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
    #[must_use]
    pub fn max_mips(&self, dim: TextureDimension) -> u32 {
        match dim {
            TextureDimension::D1 => 1,
            TextureDimension::D2 => {
                let max_dim = self.width.max(self.height);
                32 - max_dim.leading_zeros()
            }
            TextureDimension::D3 => {
                let max_dim = self.width.max(self.height.max(self.depth_or_array_layers));
                32 - max_dim.leading_zeros()
            }
        }
    }

    /// Calculates the extent at a given mip level.
    /// Does *not* account for memory size being a multiple of block size.
    ///
    /// <https://gpuweb.github.io/gpuweb/#logical-miplevel-specific-texture-extent>
    #[must_use]
    pub fn mip_level_size(&self, level: u32, dim: TextureDimension) -> Self {
        Self {
            width: u32::max(1, self.width >> level),
            height: match dim {
                TextureDimension::D1 => 1,
                _ => u32::max(1, self.height >> level),
            },
            depth_or_array_layers: match dim {
                TextureDimension::D1 => 1,
                TextureDimension::D2 => self.depth_or_array_layers,
                TextureDimension::D3 => u32::max(1, self.depth_or_array_layers >> level),
            },
        }
    }
}

#[test]
fn test_physical_size() {
    let format = TextureFormat::Bc1RgbaUnormSrgb; // 4x4 blocks
    assert_eq!(
        Extent3d {
            width: 7,
            height: 7,
            depth_or_array_layers: 1
        }
        .physical_size(format),
        Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1
        }
    );
    // Doesn't change, already aligned
    assert_eq!(
        Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1
        }
        .physical_size(format),
        Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1
        }
    );
    let format = TextureFormat::Astc {
        block: AstcBlock::B8x5,
        channel: AstcChannel::Unorm,
    }; // 8x5 blocks
    assert_eq!(
        Extent3d {
            width: 7,
            height: 7,
            depth_or_array_layers: 1
        }
        .physical_size(format),
        Extent3d {
            width: 8,
            height: 10,
            depth_or_array_layers: 1
        }
    );
}

#[test]
fn test_max_mips() {
    // 1D
    assert_eq!(
        Extent3d {
            width: 240,
            height: 1,
            depth_or_array_layers: 1
        }
        .max_mips(TextureDimension::D1),
        1
    );
    // 2D
    assert_eq!(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1
        }
        .max_mips(TextureDimension::D2),
        1
    );
    assert_eq!(
        Extent3d {
            width: 60,
            height: 60,
            depth_or_array_layers: 1
        }
        .max_mips(TextureDimension::D2),
        6
    );
    assert_eq!(
        Extent3d {
            width: 240,
            height: 1,
            depth_or_array_layers: 1000
        }
        .max_mips(TextureDimension::D2),
        8
    );
    // 3D
    assert_eq!(
        Extent3d {
            width: 16,
            height: 30,
            depth_or_array_layers: 60
        }
        .max_mips(TextureDimension::D3),
        6
    );
}

/// Describes a [`TextureView`].
///
/// For use with [`Texture::create_view()`].
///
/// Corresponds to [WebGPU `GPUTextureViewDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputextureviewdescriptor).
///
/// [`TextureView`]: ../wgpu/struct.TextureView.html
/// [`Texture::create_view()`]: ../wgpu/struct.Texture.html#method.create_view
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TextureViewDescriptor<L> {
    /// Debug label of the texture view. This will show up in graphics debuggers for easy identification.
    pub label: L,
    /// Format of the texture view. Either must be the same as the texture format or in the list
    /// of `view_formats` in the texture's descriptor.
    pub format: Option<TextureFormat>,
    /// The dimension of the texture view. For 1D textures, this must be `D1`. For 2D textures it must be one of
    /// `D2`, `D2Array`, `Cube`, and `CubeArray`. For 3D textures it must be `D3`
    pub dimension: Option<TextureViewDimension>,
    /// The allowed usage(s) for the texture view. Must be a subset of the usage flags of the texture.
    /// If not provided, defaults to the full set of usage flags of the texture.
    pub usage: Option<TextureUsages>,
    /// Aspect of the texture. Color textures must be [`TextureAspect::All`].
    pub aspect: TextureAspect,
    /// Base mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    /// If `Some(count)`, `base_mip_level + count` must be less or equal to underlying texture mip count.
    /// If `None`, considered to include the rest of the mipmap levels, but at least 1 in total.
    pub mip_level_count: Option<u32>,
    /// Base array layer.
    pub base_array_layer: u32,
    /// Layer count.
    /// If `Some(count)`, `base_array_layer + count` must be less or equal to the underlying array count.
    /// If `None`, considered to include the rest of the array layers, but at least 1 in total.
    pub array_layer_count: Option<u32>,
}

/// Describes a [`Texture`](../wgpu/struct.Texture.html).
///
/// Corresponds to [WebGPU `GPUTextureDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputexturedescriptor).
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TextureDescriptor<L, V> {
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
    /// Specifies what view formats will be allowed when calling `Texture::create_view` on this texture.
    ///
    /// View formats of the same format as the texture are always allowed.
    ///
    /// Note: currently, only the srgb-ness is allowed to change. (ex: `Rgba8Unorm` texture + `Rgba8UnormSrgb` view)
    pub view_formats: V,
}

impl<L, V> TextureDescriptor<L, V> {
    /// Takes a closure and maps the label of the texture descriptor into another.
    #[must_use]
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> TextureDescriptor<K, V>
    where
        V: Clone,
    {
        TextureDescriptor {
            label: fun(&self.label),
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: self.sample_count,
            dimension: self.dimension,
            format: self.format,
            usage: self.usage,
            view_formats: self.view_formats.clone(),
        }
    }

    /// Maps the label and view formats of the texture descriptor into another.
    #[must_use]
    pub fn map_label_and_view_formats<K, M>(
        &self,
        l_fun: impl FnOnce(&L) -> K,
        v_fun: impl FnOnce(V) -> M,
    ) -> TextureDescriptor<K, M>
    where
        V: Clone,
    {
        TextureDescriptor {
            label: l_fun(&self.label),
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: self.sample_count,
            dimension: self.dimension,
            format: self.format,
            usage: self.usage,
            view_formats: v_fun(self.view_formats.clone()),
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
    /// # type TextureDescriptor<'a> = wgpu::TextureDescriptor<(), &'a [wgpu::TextureFormat]>;
    /// let desc  = TextureDescriptor {
    ///   label: (),
    ///   size: wgpu::Extent3d { width: 100, height: 60, depth_or_array_layers: 1 },
    ///   mip_level_count: 7,
    ///   sample_count: 1,
    ///   dimension: wgpu::TextureDimension::D3,
    ///   format: wgpu::TextureFormat::Rgba8Sint,
    ///   usage: wgpu::TextureUsages::empty(),
    ///   view_formats: &[],
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
    #[must_use]
    pub fn mip_level_size(&self, level: u32) -> Option<Extent3d> {
        if level >= self.mip_level_count {
            return None;
        }

        Some(self.size.mip_level_size(level, self.dimension))
    }

    /// Computes the render extent of this texture.
    ///
    /// <https://gpuweb.github.io/gpuweb/#abstract-opdef-compute-render-extent>
    #[must_use]
    pub fn compute_render_extent(&self, mip_level: u32) -> Extent3d {
        Extent3d {
            width: u32::max(1, self.size.width >> mip_level),
            height: u32::max(1, self.size.height >> mip_level),
            depth_or_array_layers: 1,
        }
    }

    /// Returns the number of array layers.
    ///
    /// <https://gpuweb.github.io/gpuweb/#abstract-opdef-array-layer-count>
    #[must_use]
    pub fn array_layer_count(&self) -> u32 {
        match self.dimension {
            TextureDimension::D1 | TextureDimension::D3 => 1,
            TextureDimension::D2 => self.size.depth_or_array_layers,
        }
    }
}

/// Format of an `ExternalTexture`. This indicates the number of underlying
/// planes used by the `ExternalTexture` as well as each plane's format.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExternalTextureFormat {
    /// Single [`TextureFormat::Rgba8Unorm`] or [`TextureFormat::Bgra8Unorm`] format plane.
    Rgba,
    /// [`TextureFormat::R8Unorm`] Y plane, and [`TextureFormat::Rg8Unorm`]
    /// interleaved CbCr plane.
    Nv12,
    /// Separate [`TextureFormat::R8Unorm`] Y, Cb, and Cr planes.
    Yu12,
}

/// Parameters describing a gamma encoding transfer function in the form
/// tf = { k * linear                   | linear < b
///      { a * pow(linear, 1/g) - (a-1) | linear >= b
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[allow(missing_docs)]
pub struct ExternalTextureTransferFunction {
    pub a: f32,
    pub b: f32,
    pub g: f32,
    pub k: f32,
}

impl Default for ExternalTextureTransferFunction {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: 1.0,
            g: 1.0,
            k: 1.0,
        }
    }
}

/// Describes an [`ExternalTexture`](../wgpu/struct.ExternalTexture.html).
///
/// Note that [`width`] and [`height`] are the values that should be returned by
/// size queries in shader code; they do not necessarily match the dimensions of
/// the underlying plane texture(s). As a special case, if `(width, height)` is
/// `(0, 0)`, the actual size of the first underlying plane should be used instead.
///
/// The size given by [`width`] and [`height`] must be consistent with
/// [`sample_transform`]: they should be the size in texels of the rectangle
/// covered by the square (0,0)..(1,1) after [`sample_transform`] has been applied
/// to it.
///
/// [`width`]: Self::width
/// [`height`]: Self::height
/// [`sample_transform`]: Self::sample_transform
///
/// Corresponds to [WebGPU `GPUExternalTextureDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuexternaltexturedescriptor).
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExternalTextureDescriptor<L> {
    /// Debug label of the external texture. This will show up in graphics
    /// debuggers for easy identification.
    pub label: L,

    /// Width of the external texture.
    pub width: u32,

    /// Height of the external texture.
    pub height: u32,

    /// Format of the external texture.
    pub format: ExternalTextureFormat,

    /// 4x4 column-major matrix with which to convert sampled YCbCr values
    /// to RGBA.
    /// This is ignored when `format` is [`ExternalTextureFormat::Rgba`].
    pub yuv_conversion_matrix: [f32; 16],

    /// 3x3 column-major matrix to transform linear RGB values in the source
    /// color space to linear RGB values in the destination color space. In
    /// combination with [`Self::src_transfer_function`] and
    /// [`Self::dst_transfer_function`] this can be used to ensure that
    /// [`ImageSample`] and [`ImageLoad`] operations return values in the
    /// desired destination color space rather than the source color space of
    /// the underlying planes.
    ///
    /// [`ImageSample`]: https://docs.rs/naga/latest/naga/ir/enum.Expression.html#variant.ImageSample
    /// [`ImageLoad`]: https://docs.rs/naga/latest/naga/ir/enum.Expression.html#variant.ImageLoad
    pub gamut_conversion_matrix: [f32; 9],

    /// Transfer function for the source color space. The *inverse* of this
    /// will be applied to decode non-linear RGB to linear RGB in the source
    /// color space.
    pub src_transfer_function: ExternalTextureTransferFunction,

    /// Transfer function for the destination color space. This will be applied
    /// to encode linear RGB to non-linear RGB in the destination color space.
    pub dst_transfer_function: ExternalTextureTransferFunction,

    /// Transform to apply to [`ImageSample`] coordinates.
    ///
    /// This is a 3x2 column-major matrix representing an affine transform from
    /// normalized texture coordinates to the normalized coordinates that should
    /// be sampled from the external texture's underlying plane(s).
    ///
    /// This transform may scale, translate, flip, and rotate in 90-degree
    /// increments, but the result of transforming the rectangle (0,0)..(1,1)
    /// must be an axis-aligned rectangle that falls within the bounds of
    /// (0,0)..(1,1).
    ///
    /// [`ImageSample`]: https://docs.rs/naga/latest/naga/ir/enum.Expression.html#variant.ImageSample
    pub sample_transform: [f32; 6],

    /// Transform to apply to [`ImageLoad`] coordinates.
    ///
    /// This is a 3x2 column-major matrix representing an affine transform from
    /// non-normalized texel coordinates to the non-normalized coordinates of
    /// the texel that should be loaded from the external texture's underlying
    /// plane 0. For planes 1 and 2, if present, plane 0's coordinates are
    /// scaled according to the textures' relative sizes.
    ///
    /// This transform may scale, translate, flip, and rotate in 90-degree
    /// increments, but the result of transforming the rectangle (0,0)..([`width`],
    /// [`height`]) must be an axis-aligned rectangle that falls within the bounds
    /// of (0,0)..([`width`], [`height`]).
    ///
    /// [`ImageLoad`]: https://docs.rs/naga/latest/naga/ir/enum.Expression.html#variant.ImageLoad
    /// [`width`]: Self::width
    /// [`height`]: Self::height
    pub load_transform: [f32; 6],
}

impl<L> ExternalTextureDescriptor<L> {
    /// Takes a closure and maps the label of the external texture descriptor into another.
    #[must_use]
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> ExternalTextureDescriptor<K> {
        ExternalTextureDescriptor {
            label: fun(&self.label),
            width: self.width,
            height: self.height,
            format: self.format,
            yuv_conversion_matrix: self.yuv_conversion_matrix,
            sample_transform: self.sample_transform,
            load_transform: self.load_transform,
            gamut_conversion_matrix: self.gamut_conversion_matrix,
            src_transfer_function: self.src_transfer_function,
            dst_transfer_function: self.dst_transfer_function,
        }
    }

    /// The number of underlying planes used by the external texture.
    pub fn num_planes(&self) -> usize {
        match self.format {
            ExternalTextureFormat::Rgba => 1,
            ExternalTextureFormat::Nv12 => 2,
            ExternalTextureFormat::Yu12 => 3,
        }
    }
}

/// Describes a `Sampler`.
///
/// For use with `Device::create_sampler`.
///
/// Corresponds to [WebGPU `GPUSamplerDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerdescriptor).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SamplerDescriptor<L> {
    /// Debug label of the sampler. This will show up in graphics debuggers for easy identification.
    pub label: L,
    /// How to deal with out of bounds accesses in the u (i.e. x) direction
    pub address_mode_u: AddressMode,
    /// How to deal with out of bounds accesses in the v (i.e. y) direction
    pub address_mode_v: AddressMode,
    /// How to deal with out of bounds accesses in the w (i.e. z) direction
    pub address_mode_w: AddressMode,
    /// How to filter the texture when it needs to be magnified (made larger)
    pub mag_filter: FilterMode,
    /// How to filter the texture when it needs to be minified (made smaller)
    pub min_filter: FilterMode,
    /// How to filter between mip map levels
    pub mipmap_filter: FilterMode,
    /// Minimum level of detail (i.e. mip level) to use
    pub lod_min_clamp: f32,
    /// Maximum level of detail (i.e. mip level) to use
    pub lod_max_clamp: f32,
    /// If this is enabled, this is a comparison sampler using the given comparison function.
    pub compare: Option<CompareFunction>,
    /// Must be at least 1. If this is not 1, all filter modes must be linear.
    pub anisotropy_clamp: u16,
    /// Border color to use when `address_mode` is [`AddressMode::ClampToBorder`]
    pub border_color: Option<SamplerBorderColor>,
}

impl<L: Default> Default for SamplerDescriptor<L> {
    fn default() -> Self {
        Self {
            label: Default::default(),
            address_mode_u: Default::default(),
            address_mode_v: Default::default(),
            address_mode_w: Default::default(),
            mag_filter: Default::default(),
            min_filter: Default::default(),
            mipmap_filter: Default::default(),
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }
}

/// Selects a subset of the data a [`Texture`] holds.
///
/// Used in [texture views](TextureViewDescriptor) and
/// [texture copy operations](TexelCopyTextureInfo).
///
/// Corresponds to [WebGPU `GPUTextureAspect`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputextureaspect).
///
/// [`Texture`]: ../wgpu/struct.Texture.html
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum TextureAspect {
    /// Depth, Stencil, and Color.
    #[default]
    All,
    /// Stencil.
    StencilOnly,
    /// Depth.
    DepthOnly,
    /// Plane 0.
    Plane0,
    /// Plane 1.
    Plane1,
    /// Plane 2.
    Plane2,
}

/// How edges should be handled in texture addressing.
///
/// Corresponds to [WebGPU `GPUAddressMode`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpuaddressmode).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum AddressMode {
    /// Clamp the value to the edge of the texture
    ///
    /// -0.25 -> 0.0
    /// 1.25  -> 1.0
    #[default]
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

/// Texel mixing mode when sampling between texels.
///
/// Corresponds to [WebGPU `GPUFilterMode`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpufiltermode).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum FilterMode {
    /// Nearest neighbor sampling.
    ///
    /// This creates a pixelated effect when used as a mag filter
    #[default]
    Nearest = 0,
    /// Linear Interpolation
    ///
    /// This makes textures smooth but blurry when used as a mag filter.
    Linear = 1,
}

/// A range of push constant memory to pass to a shader stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PushConstantRange {
    /// Stage push constant range is visible from. Each stage can only be served by at most one range.
    /// One range can serve multiple stages however.
    pub stages: ShaderStages,
    /// Range in push constant memory to use for the stage. Must be less than [`Limits::max_push_constant_size`].
    /// Start and end must be aligned to the 4s.
    pub range: Range<u32>,
}

/// Describes a [`CommandBuffer`](../wgpu/struct.CommandBuffer.html).
///
/// Corresponds to [WebGPU `GPUCommandBufferDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucommandbufferdescriptor).
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CommandBufferDescriptor<L> {
    /// Debug label of this command buffer.
    pub label: L,
}

impl<L> CommandBufferDescriptor<L> {
    /// Takes a closure and maps the label of the command buffer descriptor into another.
    #[must_use]
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CommandBufferDescriptor<K> {
        CommandBufferDescriptor {
            label: fun(&self.label),
        }
    }
}

/// Describes the depth/stencil attachment for render bundles.
///
/// Corresponds to a portion of [WebGPU `GPURenderBundleEncoderDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderbundleencoderdescriptor).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderBundleDepthStencil {
    /// Format of the attachment.
    pub format: TextureFormat,
    /// If the depth aspect of the depth stencil attachment is going to be written to.
    ///
    /// This must match the [`RenderPassDepthStencilAttachment::depth_ops`] of the renderpass this render bundle is executed in.
    /// If `depth_ops` is `Some(..)` this must be false. If it is `None` this must be true.
    ///
    /// [`RenderPassDepthStencilAttachment::depth_ops`]: ../wgpu/struct.RenderPassDepthStencilAttachment.html#structfield.depth_ops
    pub depth_read_only: bool,

    /// If the stencil aspect of the depth stencil attachment is going to be written to.
    ///
    /// This must match the [`RenderPassDepthStencilAttachment::stencil_ops`] of the renderpass this render bundle is executed in.
    /// If `depth_ops` is `Some(..)` this must be false. If it is `None` this must be true.
    ///
    /// [`RenderPassDepthStencilAttachment::stencil_ops`]: ../wgpu/struct.RenderPassDepthStencilAttachment.html#structfield.stencil_ops
    pub stencil_read_only: bool,
}

/// Describes a [`RenderBundle`](../wgpu/struct.RenderBundle.html).
///
/// Corresponds to [WebGPU `GPURenderBundleDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderbundledescriptor).
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RenderBundleDescriptor<L> {
    /// Debug label of the render bundle encoder. This will show up in graphics debuggers for easy identification.
    pub label: L,
}

impl<L> RenderBundleDescriptor<L> {
    /// Takes a closure and maps the label of the render bundle descriptor into another.
    #[must_use]
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
///
/// Corresponds to [WebGPU `GPUTexelCopyBufferLayout`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagedatalayout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TexelCopyBufferLayout {
    /// Offset into the buffer that is the start of the texture. Must be a multiple of texture block size.
    /// For non-compressed textures, this is 1.
    pub offset: BufferAddress,
    /// Bytes per "row" in an image.
    ///
    /// A row is one row of pixels or of compressed blocks in the x direction.
    ///
    /// This value is required if there are multiple rows (i.e. height or depth is more than one pixel or pixel block for compressed textures)
    ///
    /// Must be a multiple of 256 for [`CommandEncoder::copy_buffer_to_texture`][CEcbtt]
    /// and [`CommandEncoder::copy_texture_to_buffer`][CEcttb]. You must manually pad the
    /// image such that this is a multiple of 256. It will not affect the image data.
    ///
    /// [`Queue::write_texture`][Qwt] does not have this requirement.
    ///
    /// Must be a multiple of the texture block size. For non-compressed textures, this is 1.
    ///
    /// [CEcbtt]: ../wgpu/struct.CommandEncoder.html#method.copy_buffer_to_texture
    /// [CEcttb]: ../wgpu/struct.CommandEncoder.html#method.copy_texture_to_buffer
    /// [Qwt]: ../wgpu/struct.Queue.html#method.write_texture
    pub bytes_per_row: Option<u32>,
    /// "Rows" that make up a single "image".
    ///
    /// A row is one row of pixels or of compressed blocks in the x direction.
    ///
    /// An image is one layer in the z direction of a 3D image or 2DArray texture.
    ///
    /// The amount of rows per image may be larger than the actual amount of rows of data.
    ///
    /// Required if there are multiple images (i.e. the depth is more than one).
    pub rows_per_image: Option<u32>,
}

/// Specific type of a buffer binding.
///
/// Corresponds to [WebGPU `GPUBufferBindingType`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpubufferbindingtype).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BufferBindingType {
    /// A buffer for uniform values.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// struct Globals {
    ///     a_uniform: vec2<f32>,
    ///     another_uniform: vec2<f32>,
    /// }
    /// @group(0) @binding(0)
    /// var<uniform> globals: Globals;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(std140, binding = 0)
    /// uniform Globals {
    ///     vec2 aUniform;
    ///     vec2 anotherUniform;
    /// };
    /// ```
    #[default]
    Uniform,
    /// A storage buffer.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var<storage, read_write> my_element: array<vec4<f32>>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout (set=0, binding=0) buffer myStorageBuffer {
    ///     vec4 myElement[];
    /// };
    /// ```
    Storage {
        /// If `true`, the buffer can only be read in the shader,
        /// and it:
        /// - may or may not be annotated with `read` (WGSL).
        /// - must be annotated with `readonly` (GLSL).
        ///
        /// Example WGSL syntax:
        /// ```rust,ignore
        /// @group(0) @binding(0)
        /// var<storage, read> my_element: array<vec4<f32>>;
        /// ```
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

/// Specific type of a sample in a texture binding.
///
/// Corresponds to [WebGPU `GPUTextureSampleType`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gputexturesampletype).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TextureSampleType {
    /// Sampling returns floats.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<f32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    Float {
        /// If this is `false`, the texture can't be sampled with
        /// a filtering sampler.
        ///
        /// Even if this is `true`, it's possible to sample with
        /// a **non-filtering** sampler.
        filterable: bool,
    },
    /// Sampling does the depth reference comparison.
    ///
    /// This is also compatible with a non-filtering sampler.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_depth_2d;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2DShadow t;
    /// ```
    Depth,
    /// Sampling returns signed integers.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<i32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform itexture2D t;
    /// ```
    Sint,
    /// Sampling returns unsigned integers.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<u32>;
    /// ```
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
/// For use in [`BindingType::StorageTexture`].
///
/// Corresponds to [WebGPU `GPUStorageTextureAccess`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpustoragetextureaccess).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum StorageTextureAccess {
    /// The texture can only be written in the shader and it:
    /// - may or may not be annotated with `write` (WGSL).
    /// - must be annotated with `writeonly` (GLSL).
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var my_storage_image: texture_storage_2d<r32float, write>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) writeonly uniform image2D myStorageImage;
    /// ```
    WriteOnly,
    /// The texture can only be read in the shader and it must be annotated with `read` (WGSL) or
    /// `readonly` (GLSL).
    ///
    /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] must be enabled to use this access
    /// mode. This is a native-only extension.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var my_storage_image: texture_storage_2d<r32float, read>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) readonly uniform image2D myStorageImage;
    /// ```
    ReadOnly,
    /// The texture can be both read and written in the shader and must be annotated with
    /// `read_write` in WGSL.
    ///
    /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] must be enabled to use this access
    /// mode.  This is a nonstandard, native-only extension.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var my_storage_image: texture_storage_2d<r32float, read_write>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) uniform image2D myStorageImage;
    /// ```
    ReadWrite,
    /// The texture can be both read and written in the shader via atomics and must be annotated
    /// with `read_write` in WGSL.
    ///
    /// [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] must be enabled to use this access
    /// mode.  This is a nonstandard, native-only extension.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var my_storage_image: texture_storage_2d<r32uint, atomic>;
    /// ```
    Atomic,
}

/// Specific type of a sampler binding.
///
/// For use in [`BindingType::Sampler`].
///
/// Corresponds to [WebGPU `GPUSamplerBindingType`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpusamplerbindingtype).
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

/// Type of a binding in a [bind group layout][`BindGroupLayoutEntry`].
///
/// For each binding in a layout, a [`BindGroup`] must provide a [`BindingResource`] of the
/// corresponding type.
///
/// Corresponds to WebGPU's mutually exclusive fields within [`GPUBindGroupLayoutEntry`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgrouplayoutentry).
///
/// [`BindingResource`]: ../wgpu/enum.BindingResource.html
/// [`BindGroup`]: ../wgpu/struct.BindGroup.html
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BindingType {
    /// A buffer binding.
    ///
    /// Corresponds to [WebGPU `GPUBufferBindingLayout`](
    /// https://gpuweb.github.io/gpuweb/#dictdef-gpubufferbindinglayout).
    Buffer {
        /// Sub-type of the buffer binding.
        ty: BufferBindingType,

        /// Indicates that the binding has a dynamic offset.
        ///
        /// One offset must be passed to [`RenderPass::set_bind_group`][RPsbg]
        /// for each dynamic binding in increasing order of binding number.
        ///
        /// [RPsbg]: ../wgpu/struct.RenderPass.html#method.set_bind_group
        #[cfg_attr(feature = "serde", serde(default))]
        has_dynamic_offset: bool,

        /// The minimum size for a [`BufferBinding`] matching this entry, in bytes.
        ///
        /// If this is `Some(size)`:
        ///
        /// - When calling [`create_bind_group`], the resource at this bind point
        ///   must be a [`BindingResource::Buffer`] whose effective size is at
        ///   least `size`.
        ///
        /// - When calling [`create_render_pipeline`] or [`create_compute_pipeline`],
        ///   `size` must be at least the [minimum buffer binding size] for the
        ///   shader module global at this bind point: large enough to hold the
        ///   global's value, along with one element of a trailing runtime-sized
        ///   array, if present.
        ///
        /// If this is `None`:
        ///
        /// - Each draw or dispatch command checks that the buffer range at this
        ///   bind point satisfies the [minimum buffer binding size].
        ///
        /// [`BufferBinding`]: ../wgpu/struct.BufferBinding.html
        /// [`create_bind_group`]: ../wgpu/struct.Device.html#method.create_bind_group
        /// [`BindingResource::Buffer`]: ../wgpu/enum.BindingResource.html#variant.Buffer
        /// [minimum buffer binding size]: https://www.w3.org/TR/webgpu/#minimum-buffer-binding-size
        /// [`create_render_pipeline`]: ../wgpu/struct.Device.html#method.create_render_pipeline
        /// [`create_compute_pipeline`]: ../wgpu/struct.Device.html#method.create_compute_pipeline
        #[cfg_attr(feature = "serde", serde(default))]
        min_binding_size: Option<BufferSize>,
    },
    /// A sampler that can be used to sample a texture.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var s: sampler;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform sampler s;
    /// ```
    ///
    /// Corresponds to [WebGPU `GPUSamplerBindingLayout`](
    /// https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerbindinglayout).
    Sampler(SamplerBindingType),
    /// A texture binding.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_2d<f32>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    ///
    /// Corresponds to [WebGPU `GPUTextureBindingLayout`](
    /// https://gpuweb.github.io/gpuweb/#dictdef-gputexturebindinglayout).
    Texture {
        /// Sample type of the texture binding.
        sample_type: TextureSampleType,
        /// Dimension of the texture view that is going to be sampled.
        view_dimension: TextureViewDimension,
        /// True if the texture has a sample count greater than 1. If this is true,
        /// the texture must be declared as `texture_multisampled_2d` or
        /// `texture_depth_multisampled_2d` in the shader, and read using `textureLoad`.
        multisampled: bool,
    },
    /// A storage texture.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var my_storage_image: texture_storage_2d<r32float, write>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) writeonly uniform image2D myStorageImage;
    /// ```
    /// Note that the texture format must be specified in the shader, along with the
    /// access mode. For WGSL, the format must be one of the enumerants in the list
    /// of [storage texel formats](https://gpuweb.github.io/gpuweb/wgsl/#storage-texel-formats).
    ///
    /// Corresponds to [WebGPU `GPUStorageTextureBindingLayout`](
    /// https://gpuweb.github.io/gpuweb/#dictdef-gpustoragetexturebindinglayout).
    StorageTexture {
        /// Allowed access to this texture.
        access: StorageTextureAccess,
        /// Format of the texture.
        format: TextureFormat,
        /// Dimension of the texture view that is going to be sampled.
        view_dimension: TextureViewDimension,
    },

    /// A ray-tracing acceleration structure binding.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var as: acceleration_structure;
    /// ```
    ///
    /// or with vertex return enabled
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var as: acceleration_structure<vertex_return>;
    /// ```
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform accelerationStructureEXT as;
    /// ```
    AccelerationStructure {
        /// Whether this acceleration structure can be used to
        /// create a ray query that has flag vertex return in the shader
        ///
        /// If enabled requires [`Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN`]
        vertex_return: bool,
    },

    /// An external texture binding.
    ///
    /// Example WGSL syntax:
    /// ```rust,ignore
    /// @group(0) @binding(0)
    /// var t: texture_external;
    /// ```
    ///
    /// Corresponds to [WebGPU `GPUExternalTextureBindingLayout`](
    /// https://gpuweb.github.io/gpuweb/#dictdef-gpuexternaltexturebindinglayout).
    ///
    /// Requires [`Features::EXTERNAL_TEXTURE`]
    ExternalTexture,
}

impl BindingType {
    /// Returns true for buffer bindings with dynamic offset enabled.
    #[must_use]
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
///
/// Corresponds to [WebGPU `GPUBindGroupLayoutEntry`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgrouplayoutentry).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BindGroupLayoutEntry {
    /// Binding index. Must match shader index and be unique inside a `BindGroupLayout`. A binding
    /// of index 1, would be described as `@group(0) @binding(1)` in shaders.
    pub binding: u32,
    /// Which shader stages can see this binding.
    pub visibility: ShaderStages,
    /// The type of the binding
    pub ty: BindingType,
    /// If the binding is an array of multiple resources. Corresponds to `binding_array<T>` in the shader.
    ///
    /// When this is `Some` the following validation applies:
    /// - Size must be of value 1 or greater.
    /// - When `ty == BindingType::Texture`, [`Features::TEXTURE_BINDING_ARRAY`] must be supported.
    /// - When `ty == BindingType::Sampler`, [`Features::TEXTURE_BINDING_ARRAY`] must be supported.
    /// - When `ty == BindingType::Buffer`, [`Features::BUFFER_BINDING_ARRAY`] must be supported.
    /// - When `ty == BindingType::Buffer` and `ty.ty == BufferBindingType::Storage`, [`Features::STORAGE_RESOURCE_BINDING_ARRAY`] must be supported.
    /// - When `ty == BindingType::StorageTexture`, [`Features::STORAGE_RESOURCE_BINDING_ARRAY`] must be supported.
    /// - When any binding in the group is an array, no `BindingType::Buffer` in the group may have `has_dynamic_offset == true`
    /// - When any binding in the group is an array, no `BindingType::Buffer` in the group may have `ty.ty == BufferBindingType::Uniform`.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    pub count: Option<NonZeroU32>,
}

/// View of a buffer which can be used to copy to/from a texture.
///
/// Corresponds to [WebGPU `GPUTexelCopyBufferInfo`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopybuffer).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TexelCopyBufferInfo<B> {
    /// The buffer to be copied to/from.
    pub buffer: B,
    /// The layout of the texture data in this buffer.
    pub layout: TexelCopyBufferLayout,
}

/// View of a texture which can be used to copy to/from a buffer/texture.
///
/// Corresponds to [WebGPU `GPUTexelCopyTextureInfo`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopytexture).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TexelCopyTextureInfo<T> {
    /// The texture to be copied to/from.
    pub texture: T,
    /// The target mip level of the texture.
    pub mip_level: u32,
    /// The base texel of the texture in the selected `mip_level`. Together
    /// with the `copy_size` argument to copy functions, defines the
    /// sub-region of the texture to copy.
    #[cfg_attr(feature = "serde", serde(default))]
    pub origin: Origin3d,
    /// The copy aspect.
    #[cfg_attr(feature = "serde", serde(default))]
    pub aspect: TextureAspect,
}

impl<T> TexelCopyTextureInfo<T> {
    /// Adds color space and premultiplied alpha information to make this
    /// descriptor tagged.
    pub fn to_tagged(
        self,
        color_space: PredefinedColorSpace,
        premultiplied_alpha: bool,
    ) -> CopyExternalImageDestInfo<T> {
        CopyExternalImageDestInfo {
            texture: self.texture,
            mip_level: self.mip_level,
            origin: self.origin,
            aspect: self.aspect,
            color_space,
            premultiplied_alpha,
        }
    }
}

/// View of an external texture that can be used to copy to a texture.
///
/// Corresponds to [WebGPU `GPUCopyExternalImageSourceInfo`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopyexternalimage).
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[derive(Clone, Debug)]
pub struct CopyExternalImageSourceInfo {
    /// The texture to be copied from. The copy source data is captured at the moment
    /// the copy is issued.
    pub source: ExternalImageSource,
    /// The base texel used for copying from the external image. Together
    /// with the `copy_size` argument to copy functions, defines the
    /// sub-region of the image to copy.
    ///
    /// Relative to the top left of the image.
    ///
    /// Must be [`Origin2d::ZERO`] if [`DownlevelFlags::UNRESTRICTED_EXTERNAL_TEXTURE_COPIES`] is not supported.
    pub origin: Origin2d,
    /// If the Y coordinate of the image should be flipped. Even if this is
    /// true, `origin` is still relative to the top left.
    pub flip_y: bool,
}

/// Source of an external texture copy.
///
/// Corresponds to the [implicit union type on WebGPU `GPUCopyExternalImageSourceInfo.source`](
/// https://gpuweb.github.io/gpuweb/#dom-gpuimagecopyexternalimage-source).
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[derive(Clone, Debug)]
pub enum ExternalImageSource {
    /// Copy from a previously-decoded image bitmap.
    ImageBitmap(web_sys::ImageBitmap),
    /// Copy from an image element.
    HTMLImageElement(web_sys::HtmlImageElement),
    /// Copy from a current frame of a video element.
    HTMLVideoElement(web_sys::HtmlVideoElement),
    /// Copy from an image.
    ImageData(web_sys::ImageData),
    /// Copy from a on-screen canvas.
    HTMLCanvasElement(web_sys::HtmlCanvasElement),
    /// Copy from a off-screen canvas.
    ///
    /// Requires [`DownlevelFlags::UNRESTRICTED_EXTERNAL_TEXTURE_COPIES`]
    OffscreenCanvas(web_sys::OffscreenCanvas),
    /// Copy from a video frame.
    #[cfg(web_sys_unstable_apis)]
    VideoFrame(web_sys::VideoFrame),
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
impl ExternalImageSource {
    /// Gets the pixel, not css, width of the source.
    pub fn width(&self) -> u32 {
        match self {
            ExternalImageSource::ImageBitmap(b) => b.width(),
            ExternalImageSource::HTMLImageElement(i) => i.width(),
            ExternalImageSource::HTMLVideoElement(v) => v.video_width(),
            ExternalImageSource::ImageData(i) => i.width(),
            ExternalImageSource::HTMLCanvasElement(c) => c.width(),
            ExternalImageSource::OffscreenCanvas(c) => c.width(),
            #[cfg(web_sys_unstable_apis)]
            ExternalImageSource::VideoFrame(v) => v.display_width(),
        }
    }

    /// Gets the pixel, not css, height of the source.
    pub fn height(&self) -> u32 {
        match self {
            ExternalImageSource::ImageBitmap(b) => b.height(),
            ExternalImageSource::HTMLImageElement(i) => i.height(),
            ExternalImageSource::HTMLVideoElement(v) => v.video_height(),
            ExternalImageSource::ImageData(i) => i.height(),
            ExternalImageSource::HTMLCanvasElement(c) => c.height(),
            ExternalImageSource::OffscreenCanvas(c) => c.height(),
            #[cfg(web_sys_unstable_apis)]
            ExternalImageSource::VideoFrame(v) => v.display_height(),
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
impl core::ops::Deref for ExternalImageSource {
    type Target = js_sys::Object;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::ImageBitmap(b) => b,
            Self::HTMLImageElement(i) => i,
            Self::HTMLVideoElement(v) => v,
            Self::ImageData(i) => i,
            Self::HTMLCanvasElement(c) => c,
            Self::OffscreenCanvas(c) => c,
            #[cfg(web_sys_unstable_apis)]
            Self::VideoFrame(v) => v,
        }
    }
}

#[cfg(all(
    target_arch = "wasm32",
    feature = "web",
    feature = "fragile-send-sync-non-atomic-wasm",
    not(target_feature = "atomics")
))]
unsafe impl Send for ExternalImageSource {}
#[cfg(all(
    target_arch = "wasm32",
    feature = "web",
    feature = "fragile-send-sync-non-atomic-wasm",
    not(target_feature = "atomics")
))]
unsafe impl Sync for ExternalImageSource {}

/// Color spaces supported on the web.
///
/// Corresponds to [HTML Canvas `PredefinedColorSpace`](
/// https://html.spec.whatwg.org/multipage/canvas.html#predefinedcolorspace).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum PredefinedColorSpace {
    /// sRGB color space
    Srgb,
    /// Display-P3 color space
    DisplayP3,
}

/// View of a texture which can be used to copy to a texture, including
/// color space and alpha premultiplication information.
///
/// Corresponds to [WebGPU `GPUCopyExternalImageDestInfo`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopytexturetagged).
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CopyExternalImageDestInfo<T> {
    /// The texture to be copied to/from.
    pub texture: T,
    /// The target mip level of the texture.
    pub mip_level: u32,
    /// The base texel of the texture in the selected `mip_level`.
    pub origin: Origin3d,
    /// The copy aspect.
    pub aspect: TextureAspect,
    /// The color space of this texture.
    pub color_space: PredefinedColorSpace,
    /// The premultiplication of this texture
    pub premultiplied_alpha: bool,
}

impl<T> CopyExternalImageDestInfo<T> {
    /// Removes the colorspace information from the type.
    pub fn to_untagged(self) -> TexelCopyTextureInfo<T> {
        TexelCopyTextureInfo {
            texture: self.texture,
            mip_level: self.mip_level,
            origin: self.origin,
            aspect: self.aspect,
        }
    }
}

/// Subresource range within an image
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ImageSubresourceRange {
    /// Aspect of the texture. Color textures must be [`TextureAspect::All`][TAA].
    ///
    /// [TAA]: ../wgpu/enum.TextureAspect.html#variant.All
    pub aspect: TextureAspect,
    /// Base mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    /// If `Some(count)`, `base_mip_level + count` must be less or equal to underlying texture mip count.
    /// If `None`, considered to include the rest of the mipmap levels, but at least 1 in total.
    pub mip_level_count: Option<u32>,
    /// Base array layer.
    pub base_array_layer: u32,
    /// Layer count.
    /// If `Some(count)`, `base_array_layer + count` must be less or equal to the underlying array count.
    /// If `None`, considered to include the rest of the array layers, but at least 1 in total.
    pub array_layer_count: Option<u32>,
}

impl ImageSubresourceRange {
    /// Returns if the given range represents a full resource, with a texture of the given
    /// layer count and mip count.
    ///
    /// ```rust
    /// # use wgpu_types as wgpu;
    ///
    /// let range_none = wgpu::ImageSubresourceRange {
    ///     aspect: wgpu::TextureAspect::All,
    ///     base_mip_level: 0,
    ///     mip_level_count: None,
    ///     base_array_layer: 0,
    ///     array_layer_count: None,
    /// };
    /// assert_eq!(range_none.is_full_resource(wgpu::TextureFormat::Stencil8, 5, 10), true);
    ///
    /// let range_some = wgpu::ImageSubresourceRange {
    ///     aspect: wgpu::TextureAspect::All,
    ///     base_mip_level: 0,
    ///     mip_level_count: Some(5),
    ///     base_array_layer: 0,
    ///     array_layer_count: Some(10),
    /// };
    /// assert_eq!(range_some.is_full_resource(wgpu::TextureFormat::Stencil8, 5, 10), true);
    ///
    /// let range_mixed = wgpu::ImageSubresourceRange {
    ///     aspect: wgpu::TextureAspect::StencilOnly,
    ///     base_mip_level: 0,
    ///     // Only partial resource
    ///     mip_level_count: Some(3),
    ///     base_array_layer: 0,
    ///     array_layer_count: None,
    /// };
    /// assert_eq!(range_mixed.is_full_resource(wgpu::TextureFormat::Stencil8, 5, 10), false);
    /// ```
    #[must_use]
    pub fn is_full_resource(
        &self,
        format: TextureFormat,
        mip_levels: u32,
        array_layers: u32,
    ) -> bool {
        // Mip level count and array layer count need to deal with both the None and Some(count) case.
        let mip_level_count = self.mip_level_count.unwrap_or(mip_levels);
        let array_layer_count = self.array_layer_count.unwrap_or(array_layers);

        let aspect_eq = Some(format) == format.aspect_specific_format(self.aspect);

        let base_mip_level_eq = self.base_mip_level == 0;
        let mip_level_count_eq = mip_level_count == mip_levels;

        let base_array_layer_eq = self.base_array_layer == 0;
        let array_layer_count_eq = array_layer_count == array_layers;

        aspect_eq
            && base_mip_level_eq
            && mip_level_count_eq
            && base_array_layer_eq
            && array_layer_count_eq
    }

    /// Returns the mip level range of a subresource range describes for a specific texture.
    #[must_use]
    pub fn mip_range(&self, mip_level_count: u32) -> Range<u32> {
        self.base_mip_level..match self.mip_level_count {
            Some(mip_level_count) => self.base_mip_level + mip_level_count,
            None => mip_level_count,
        }
    }

    /// Returns the layer range of a subresource range describes for a specific texture.
    #[must_use]
    pub fn layer_range(&self, array_layer_count: u32) -> Range<u32> {
        self.base_array_layer..match self.array_layer_count {
            Some(array_layer_count) => self.base_array_layer + array_layer_count,
            None => array_layer_count,
        }
    }
}

/// Color variation to use when sampler addressing mode is [`AddressMode::ClampToBorder`]
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerBorderColor {
    /// [0, 0, 0, 0]
    TransparentBlack,
    /// [0, 0, 0, 1]
    OpaqueBlack,
    /// [1, 1, 1, 1]
    OpaqueWhite,

    /// On the Metal backend, this is equivalent to `TransparentBlack` for
    /// textures that have an alpha component, and equivalent to `OpaqueBlack`
    /// for textures that do not have an alpha component. On other backends,
    /// this is equivalent to `TransparentBlack`. Requires
    /// [`Features::ADDRESS_MODE_CLAMP_TO_ZERO`]. Not supported on the web.
    Zero,
}

/// Describes how to create a `QuerySet`.
///
/// Corresponds to [WebGPU `GPUQuerySetDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuquerysetdescriptor).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Takes a closure and maps the label of the query set descriptor into another.
    #[must_use]
    pub fn map_label<'a, K>(&'a self, fun: impl FnOnce(&'a L) -> K) -> QuerySetDescriptor<K> {
        QuerySetDescriptor {
            label: fun(&self.label),
            ty: self.ty,
            count: self.count,
        }
    }
}

/// Type of query contained in a [`QuerySet`].
///
/// Corresponds to [WebGPU `GPUQueryType`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpuquerytype).
///
/// [`QuerySet`]: ../wgpu/struct.QuerySet.html
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Must be multiplied by [`Queue::get_timestamp_period`][Qgtp] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    ///
    /// [`Features::TIMESTAMP_QUERY`] must be enabled to use this query type.
    ///
    /// [Qgtp]: ../wgpu/struct.Queue.html#method.get_timestamp_period
    Timestamp,
}

bitflags::bitflags! {
    /// Flags for which pipeline data should be recorded in a query.
    ///
    /// Used in [`QueryType`].
    ///
    /// The amount of values written when resolved depends
    /// on the amount of flags set. For example, if 3 flags are set, 3
    /// 64-bit values will be written per query.
    ///
    /// The order they are written is the order they are declared
    /// in these bitflags. For example, if you enabled `CLIPPER_PRIMITIVES_OUT`
    /// and `COMPUTE_SHADER_INVOCATIONS`, it would write 16 bytes,
    /// the first 8 bytes being the primitive out value, the last 8
    /// bytes being the compute shader invocation count.
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

/// Argument buffer layout for `draw_indirect` commands.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct DrawIndirectArgs {
    /// The number of vertices to draw.
    pub vertex_count: u32,
    /// The number of instances to draw.
    pub instance_count: u32,
    /// The Index of the first vertex to draw.
    pub first_vertex: u32,
    /// The instance ID of the first instance to draw.
    ///
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    pub first_instance: u32,
}

impl DrawIndirectArgs {
    /// Returns the bytes representation of the struct, ready to be written in a buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// Argument buffer layout for `draw_indexed_indirect` commands.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct DrawIndexedIndirectArgs {
    /// The number of indices to draw.
    pub index_count: u32,
    /// The number of instances to draw.
    pub instance_count: u32,
    /// The first index within the index buffer.
    pub first_index: u32,
    /// The value added to the vertex index before indexing into the vertex buffer.
    pub base_vertex: i32,
    /// The instance ID of the first instance to draw.
    ///
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    pub first_instance: u32,
}

impl DrawIndexedIndirectArgs {
    /// Returns the bytes representation of the struct, ready to be written in a buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// Argument buffer layout for `dispatch_indirect` commands.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct DispatchIndirectArgs {
    /// The number of work groups in X dimension.
    pub x: u32,
    /// The number of work groups in Y dimension.
    pub y: u32,
    /// The number of work groups in Z dimension.
    pub z: u32,
}

impl DispatchIndirectArgs {
    /// Returns the bytes representation of the struct, ready to be written into a buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// Describes how shader bound checks should be performed.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShaderRuntimeChecks {
    /// Enforce bounds checks in shaders, even if the underlying driver doesn't
    /// support doing so natively.
    ///
    /// When this is `true`, `wgpu` promises that shaders can only read or
    /// write the accessible region of a bindgroup's buffer bindings. If
    /// the underlying graphics platform cannot implement these bounds checks
    /// itself, `wgpu` will inject bounds checks before presenting the
    /// shader to the platform.
    ///
    /// When this is `false`, `wgpu` only enforces such bounds checks if the
    /// underlying platform provides a way to do so itself. `wgpu` does not
    /// itself add any bounds checks to generated shader code.
    ///
    /// Note that `wgpu` users may try to initialize only those portions of
    /// buffers that they anticipate might be read from. Passing `false` here
    /// may allow shaders to see wider regions of the buffers than expected,
    /// making such deferred initialization visible to the application.
    pub bounds_checks: bool,
    ///
    /// If false, the caller MUST ensure that all passed shaders do not contain any infinite loops.
    ///
    /// If it does, backend compilers MAY treat such a loop as unreachable code and draw
    /// conclusions about other safety-critical code paths. This option SHOULD NOT be disabled
    /// when running untrusted code.
    pub force_loop_bounding: bool,
}

impl ShaderRuntimeChecks {
    /// Creates a new configuration where the shader is fully checked.
    #[must_use]
    pub fn checked() -> Self {
        unsafe { Self::all(true) }
    }

    /// Creates a new configuration where none of the checks are performed.
    ///
    /// # Safety
    ///
    /// See the documentation for the `set_*` methods for the safety requirements
    /// of each sub-configuration.
    #[must_use]
    pub fn unchecked() -> Self {
        unsafe { Self::all(false) }
    }

    /// Creates a new configuration where all checks are enabled or disabled. To safely
    /// create a configuration with all checks enabled, use [`ShaderRuntimeChecks::checked`].
    ///
    /// # Safety
    ///
    /// See the documentation for the `set_*` methods for the safety requirements
    /// of each sub-configuration.
    #[must_use]
    pub unsafe fn all(all_checks: bool) -> Self {
        Self {
            bounds_checks: all_checks,
            force_loop_bounding: all_checks,
        }
    }
}

impl Default for ShaderRuntimeChecks {
    fn default() -> Self {
        Self::checked()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for all size defining attributes of a single triangle geometry inside a bottom level acceleration structure.
pub struct BlasTriangleGeometrySizeDescriptor {
    /// Format of a vertex position, must be [`VertexFormat::Float32x3`]
    /// with just [`Features::EXPERIMENTAL_RAY_QUERY`]
    /// but [`Features::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS`] adds more.
    pub vertex_format: VertexFormat,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Format of an index. Only needed if an index buffer is used.
    /// If `index_format` is provided `index_count` is required.
    pub index_format: Option<IndexFormat>,
    /// Number of indices. Only needed if an index buffer is used.
    /// If `index_count` is provided `index_format` is required.
    pub index_count: Option<u32>,
    /// Flags for the geometry.
    pub flags: AccelerationStructureGeometryFlags,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for all size defining attributes of all geometries inside a bottom level acceleration structure.
pub enum BlasGeometrySizeDescriptors {
    /// Triangle geometry version.
    Triangles {
        /// Descriptor for each triangle geometry.
        descriptors: Vec<BlasTriangleGeometrySizeDescriptor>,
    },
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Update mode for acceleration structure builds.
pub enum AccelerationStructureUpdateMode {
    /// Always perform a full build.
    Build,
    /// If possible, perform an incremental update.
    ///
    /// Not advised for major topology changes.
    /// (Useful for e.g. skinning)
    PreferUpdate,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a bottom level acceleration structure.
pub struct CreateBlasDescriptor<L> {
    /// Label for the bottom level acceleration structure.
    pub label: L,
    /// Flags for the bottom level acceleration structure.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the bottom level acceleration structure.
    pub update_mode: AccelerationStructureUpdateMode,
}

impl<L> CreateBlasDescriptor<L> {
    /// Takes a closure and maps the label of the blas descriptor into another.
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CreateBlasDescriptor<K> {
        CreateBlasDescriptor {
            label: fun(&self.label),
            flags: self.flags,
            update_mode: self.update_mode,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a top level acceleration structure.
pub struct CreateTlasDescriptor<L> {
    /// Label for the top level acceleration structure.
    pub label: L,
    /// Number of instances that can be stored in the acceleration structure.
    pub max_instances: u32,
    /// Flags for the bottom level acceleration structure.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the bottom level acceleration structure.
    pub update_mode: AccelerationStructureUpdateMode,
}

impl<L> CreateTlasDescriptor<L> {
    /// Takes a closure and maps the label of the blas descriptor into another.
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> CreateTlasDescriptor<K> {
        CreateTlasDescriptor {
            label: fun(&self.label),
            flags: self.flags,
            update_mode: self.update_mode,
            max_instances: self.max_instances,
        }
    }
}

bitflags::bitflags!(
    /// Flags for acceleration structures
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureFlags: u8 {
        /// Allow for incremental updates (no change in size), currently this is unimplemented
        /// and will build as normal (this is fine, update vs build should be unnoticeable)
        const ALLOW_UPDATE = 1 << 0;
        /// Allow the acceleration structure to be compacted in a copy operation
        /// (`Blas::prepare_for_compaction`, `CommandEncoder::compact_blas`).
        const ALLOW_COMPACTION = 1 << 1;
        /// Optimize for fast ray tracing performance, recommended if the geometry is unlikely
        /// to change (e.g. in a game: non-interactive scene geometry)
        const PREFER_FAST_TRACE = 1 << 2;
        /// Optimize for fast build time, recommended if geometry is likely to change frequently
        /// (e.g. in a game: player model).
        const PREFER_FAST_BUILD = 1 << 3;
        /// Optimize for low memory footprint (both while building and in the output BLAS).
        const LOW_MEMORY = 1 << 4;
        /// Use `BlasTriangleGeometry::transform_buffer` when building a BLAS (only allowed in
        /// BLAS creation)
        const USE_TRANSFORM = 1 << 5;
        /// Allow retrieval of the vertices of the triangle hit by a ray.
        const ALLOW_RAY_HIT_VERTEX_RETURN = 1 << 6;
    }
);

bitflags::bitflags!(
    /// Flags for acceleration structure geometries
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureGeometryFlags: u8 {
        /// Is OPAQUE (is there no alpha test) recommended as currently in naga there is no
        /// candidate intersections yet so currently BLASes without this flag will not have hits.
        /// Not enabling this makes the BLAS unable to be interacted with in WGSL.
        const OPAQUE = 1 << 0;
        /// NO_DUPLICATE_ANY_HIT_INVOCATION, not useful unless using hal with wgpu, ray-tracing
        /// pipelines are not supported in wgpu so any-hit shaders do not exist. For when any-hit
        /// shaders are implemented (or experienced users who combine this with an underlying library:
        /// for any primitive (triangle or AABB) multiple any-hit shaders sometimes may be invoked
        /// (especially in AABBs like a sphere), if this flag in present only one hit on a primitive may
        /// invoke an any-hit shader.
        const NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1;
    }
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// What a copy between acceleration structures should do
pub enum AccelerationStructureCopy {
    /// Directly duplicate an acceleration structure to another
    Clone,
    /// Duplicate and compact an acceleration structure
    Compact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// What type the data of an acceleration structure is
pub enum AccelerationStructureType {
    /// The types of the acceleration structure are triangles
    Triangles,
    /// The types of the acceleration structure are axis aligned bounding boxes
    AABBs,
    /// The types of the acceleration structure are instances
    Instances,
}

/// Alignment requirement for transform buffers used in acceleration structure builds
pub const TRANSFORM_BUFFER_ALIGNMENT: BufferAddress = 16;

/// Alignment requirement for instance buffers used in acceleration structure builds (`build_acceleration_structures_unsafe_tlas`)
pub const INSTANCE_BUFFER_ALIGNMENT: BufferAddress = 16;

pub use send_sync::*;

#[doc(hidden)]
mod send_sync {
    pub trait WasmNotSendSync: WasmNotSend + WasmNotSync {}
    impl<T: WasmNotSend + WasmNotSync> WasmNotSendSync for T {}
    #[cfg(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    ))]
    pub trait WasmNotSend: Send {}
    #[cfg(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    ))]
    impl<T: Send> WasmNotSend for T {}
    #[cfg(not(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    )))]
    pub trait WasmNotSend {}
    #[cfg(not(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    )))]
    impl<T> WasmNotSend for T {}

    #[cfg(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    ))]
    pub trait WasmNotSync: Sync {}
    #[cfg(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    ))]
    impl<T: Sync> WasmNotSync for T {}
    #[cfg(not(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    )))]
    pub trait WasmNotSync {}
    #[cfg(not(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    )))]
    impl<T> WasmNotSync for T {}
}

/// Corresponds to a [`GPUDeviceLostReason`].
///
/// [`GPUDeviceLostReason`]: https://www.w3.org/TR/webgpu/#enumdef-gpudevicelostreason
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceLostReason {
    /// The device was lost for an unspecific reason, including driver errors.
    Unknown = 0,
    /// The device's `destroy` method was called.
    Destroyed = 1,
}

/// Descriptor for a shader module given by any of several sources.
/// These shaders are passed through directly to the underlying api.
/// At least one shader type that may be used by the backend must be `Some` or a panic is raised.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CreateShaderModuleDescriptorPassthrough<'a, L> {
    /// Entrypoint. Unused for Spir-V.
    pub entry_point: String,
    /// Debug label of the shader module. This will show up in graphics debuggers for easy identification.
    pub label: L,
    /// Number of workgroups in each dimension x, y and z. Unused for Spir-V.
    pub num_workgroups: (u32, u32, u32),
    /// Runtime checks that should be enabled.
    pub runtime_checks: ShaderRuntimeChecks,

    /// Binary SPIR-V data, in 4-byte words.
    pub spirv: Option<Cow<'a, [u32]>>,
    /// Shader DXIL source.
    pub dxil: Option<Cow<'a, [u8]>>,
    /// Shader MSL source.
    pub msl: Option<Cow<'a, str>>,
    /// Shader HLSL source.
    pub hlsl: Option<Cow<'a, str>>,
    /// Shader GLSL source (currently unused).
    pub glsl: Option<Cow<'a, str>>,
    /// Shader WGSL source.
    pub wgsl: Option<Cow<'a, str>>,
}

// This is so people don't have to fill in fields they don't use, like num_workgroups,
// entry_point, or other shader languages they didn't compile for
impl<'a, L: Default> Default for CreateShaderModuleDescriptorPassthrough<'a, L> {
    fn default() -> Self {
        Self {
            entry_point: "".into(),
            label: Default::default(),
            num_workgroups: (0, 0, 0),
            runtime_checks: ShaderRuntimeChecks::unchecked(),
            spirv: None,
            dxil: None,
            msl: None,
            hlsl: None,
            glsl: None,
            wgsl: None,
        }
    }
}

impl<'a, L> CreateShaderModuleDescriptorPassthrough<'a, L> {
    /// Takes a closure and maps the label of the shader module descriptor into another.
    pub fn map_label<K>(
        &self,
        fun: impl FnOnce(&L) -> K,
    ) -> CreateShaderModuleDescriptorPassthrough<'a, K> {
        CreateShaderModuleDescriptorPassthrough {
            entry_point: self.entry_point.clone(),
            label: fun(&self.label),
            num_workgroups: self.num_workgroups,
            runtime_checks: self.runtime_checks,
            spirv: self.spirv.clone(),
            dxil: self.dxil.clone(),
            msl: self.msl.clone(),
            hlsl: self.hlsl.clone(),
            glsl: self.glsl.clone(),
            wgsl: self.wgsl.clone(),
        }
    }

    #[cfg(feature = "trace")]
    /// Returns the source data for tracing purpose.
    pub fn trace_data(&self) -> &[u8] {
        if let Some(spirv) = &self.spirv {
            bytemuck::cast_slice(spirv)
        } else if let Some(msl) = &self.msl {
            msl.as_bytes()
        } else if let Some(dxil) = &self.dxil {
            dxil
        } else {
            panic!("No binary data provided to `ShaderModuleDescriptorGeneric`")
        }
    }

    #[cfg(feature = "trace")]
    /// Returns the binary file extension for tracing purpose.
    pub fn trace_binary_ext(&self) -> &'static str {
        if self.spirv.is_some() {
            "spv"
        } else if self.msl.is_some() {
            "msl"
        } else if self.dxil.is_some() {
            "dxil"
        } else {
            panic!("No binary data provided to `ShaderModuleDescriptorGeneric`")
        }
    }
}
