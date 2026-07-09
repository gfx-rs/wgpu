//! `wgpu` is a cross-platform, safe, pure-Rust graphics API. It runs natively on
//! Vulkan, Metal, D3D12, and OpenGL; and on top of WebGL2 and WebGPU on wasm.
//!
//! The API is based on the [WebGPU standard][webgpu], but is a fully native Rust library.
//! It serves as the core of the WebGPU integration in Firefox, Servo, and Deno.
//!
//! [webgpu]: https://gpuweb.github.io/gpuweb/
//!
//! ## Getting Started
//!
//! The main entry point to the API is the [`Instance`] type, from which you can create [`Adapter`], [`Device`], and [`Surface`].
//!
//! If you are new to `wgpu` and graphics programming, we recommend starting with [Learn Wgpu].
//! <!-- Note, "Learn Wgpu" is using the capitalization style in their header, NOT our styling -->
//!
//! Additionally, [WebGPU Fundamentals] is a tutorial for WebGPU which is very similar to our API, minus differences between Rust and Javascript.
//!
//! We have a [wiki](https://github.com/gfx-rs/wgpu/wiki) which has information on useful architecture patterns, debugging tips, and more getting started information.
//!
//! There are examples for this version [available on GitHub](https://github.com/gfx-rs/wgpu/tree/v30/examples#readme).
//!
//! The API is refcounted, so all handles are cloneable, and if you create a resource which references another,
//! it will automatically keep dependent resources alive.
//!
//! `wgpu` uses the coordinate systems of D3D and Metal. Depth ranges from [0, 1].
//!
//! | Render                | Texture                |
//! | --------------------- | ---------------------- |
//! | ![render_coordinates] | ![texture_coordinates] |
//!
//! `wgpu`'s MSRV is **1.87**.
//!
//! [Learn Wgpu]: https://sotrh.github.io/learn-wgpu/
//! [WebGPU Fundamentals]: https://webgpufundamentals.org/
//! [render_coordinates]: https://raw.githubusercontent.com/gfx-rs/wgpu/refs/heads/v30/docs/render_coordinates.png
//! [texture_coordinates]: https://raw.githubusercontent.com/gfx-rs/wgpu/refs/heads/v30/docs/texture_coordinates.png
//!
//! ## Extension Specifications
//!
//! While the core of `wgpu` is based on the WebGPU standard, we also support extensions that allow for features that the standard does not have yet.
//! For high-level documentation on how to use these extensions, see documentation on [`Features`] or the relevant specification:
//!
//! 🧪EXPERIMENTAL🧪 APIs are subject to change and may allow undefined behavior if used incorrectly.
//!
//! - 🧪EXPERIMENTAL🧪 [Ray Tracing](https://github.com/gfx-rs/wgpu/blob/v30/docs/api-specs/ray_tracing.md).
//! - 🧪EXPERIMENTAL🧪 [Mesh Shading](https://github.com/gfx-rs/wgpu/blob/v30/docs/api-specs/mesh_shading.md).
//!
//! ## Shader Support
//!
//! `wgpu` can consume shaders in [WGSL](https://gpuweb.github.io/gpuweb/wgsl/), SPIR-V, and GLSL.
//! Both [HLSL](https://github.com/Microsoft/DirectXShaderCompiler) and [GLSL](https://github.com/KhronosGroup/glslang)
//! have compilers to target SPIR-V. All of these shader languages can be used with any backend as we handle all of the conversions. Additionally, support for these shader inputs is not going away.
//!
//! While WebGPU does not support any shading language other than WGSL, we will automatically convert your
//! non-WGSL shaders if you're running on WebGPU.
//!
//! WGSL is always supported by default, but GLSL and SPIR-V need features enabled to compile in support.
//!
//! To enable WGSL shaders, enable the `wgsl` feature of `wgpu` (enabled by default).
//! To enable SPIR-V shaders, enable the `spirv` feature of `wgpu`.
//! To enable GLSL shaders, enable the `glsl` feature of `wgpu`.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
//!
//! ### Feature Aliases
//!
//! These features aren't actually features on the crate itself, but a convenient shorthand for
//! complicated cases.
//!
//! - **`wgpu_core`** --- Enabled when there is any non-webgpu backend enabled on the platform.
//! - **`naga`** --- Enabled when target `glsl` or `spirv` input is enabled, or when `wgpu_core` is enabled.
//!
//! ## Surface color spaces and HDR output
//!
//! A surface can present in different color spaces to get HDR or wide-gamut
//! output onto the screen. This is configured through
//! [`SurfaceConfiguration::color_space`], and the rest of this section is a
//! concept primer on how it works.
//!
//! ### HDR output in a nutshell
//!
//! By default a surface is standard dynamic range (SDR) with the sRGB gamut: the
//! value `1.0` is the brightest white, anything above it clips, and only colors
//! within sRGB are expressible. Other [`SurfaceColorSpace`]s opt into a **wider
//! gamut** (for example [`DisplayP3`](SurfaceColorSpace::DisplayP3), still SDR
//! but with more saturated colors than sRGB), **high dynamic range** (values
//! above `1.0` drive brighter-than-white output), or both (for example
//! [`Bt2100Pq`](SurfaceColorSpace::Bt2100Pq), aka HDR10), on platforms that
//! support it.
//!
//! Three ideas carry most of the weight:
//!
//! * **Reference white and headroom.** Brightness is measured in *nits*
//!   (cd/m²). On HDR-capable monitors, SDR reference white (plain white, `(1.0, 1.0, 1.0)` in the
//!   extended color spaces) sits *below* the display's peak output on purpose, so
//!   highlights have room above it. That gap is the display's *headroom*.
//! * **The transfer function is a round-trip.** Your shader applies an
//!   encoding transfer function (the OETF) to turn the light it computed into a
//!   stored signal, and the display applies the inverse decoding transfer
//!   function (the EOTF) to turn it back into light. Choosing a color space
//!   chooses which transfer functions both ends use.
//! * **Who applies the encoding transfer function.** wgpu applies it for you
//!   **only** when you render to an `*Srgb` texture view format, where the GPU
//!   runs the sRGB OETF when a value is stored to a texture. For every other
//!   color space (linear extended sRGB, encoded extended sRGB or P3, PQ, HLG)
//!   **the values your shader writes to the surface texture must already be
//!   encoded by you**, along with any gamut conversion; in a typical renderer
//!   this happens in a final tone-mapping or post-processing pass. wgpu hands
//!   the signal to the compositor unchanged; getting this wrong produces a
//!   wrong image with no error.
//!
//! wgpu does **not** tonemap or gamut-map for you. It gives you the surface
//! and, through [`DisplayHdrInfo`], the display's advisory capabilities;
//! *choosing* and *applying* a tone curve is your application's job.
//!
//! ### The practical path
//!
//! 1. **Query capabilities.** Call `Surface::get_capabilities`. To use HDR or
//!    wide-gamut output, read [`SurfaceCapabilities::format_capabilities`] (each
//!    format and the [`SurfaceColorSpaces`] it supports), **not**
//!    [`SurfaceCapabilities::formats`]: the latter lists only formats usable
//!    with [`Auto`](SurfaceColorSpace::Auto), which never selects HDR.
//!    [`SurfaceCapabilities::color_spaces`] is a convenience lookup for one
//!    format.
//! 2. **Optionally query the display.** Call `Surface::display_hdr_info` for the
//!    current [`DisplayHdrInfo`] (peak and SDR-white nits, EDR headroom,
//!    primaries, and a coarse dynamic-range/gamut bucket). Use it to pick a
//!    tone-map target ([`DisplayHdrInfo::tone_map_headroom`]); *whether* HDR is
//!    worthwhile is the capability question from step 1, not this live value.
//!    Every field is advisory and optional; `None` means "cannot tell here",
//!    never "SDR".
//! 3. **Choose a format and color space.** Intersect what you want with what
//!    step 1 advertises, in your own preference order (for example HDR10, then
//!    linear extended sRGB, then encoded extended sRGB, then SDR
//!    [`Srgb`](SurfaceColorSpace::Srgb)). Keep an SDR fallback for when nothing
//!    HDR is advertised, such as when OS HDR is off.
//! 4. **Configure the surface.** Set [`SurfaceConfiguration::color_space`] and
//!    `format`. [`Auto`](SurfaceColorSpace::Auto) (the default) reproduces
//!    wgpu's historical behavior and never picks HDR; any other value must be in
//!    that format's advertised set or configuration fails validation.
//! 5. **Encode what you write to the surface texture.** For an `*Srgb` format,
//!    output linear and the hardware encodes for you. Otherwise the values your
//!    shader writes to the surface texture must already carry the encoding the
//!    chosen color space expects (sRGB, extended sRGB, PQ, or HLG) **and** any
//!    gamut conversion (for example outputting in the BT.2020 gamut for HDR10);
//!    in a typical renderer you do this in a final tone-mapping or
//!    post-processing pass. See the table below.
//! 6. **Present** as usual. If OS HDR is toggled mid-run, you'll see it on the
//!    next `Surface::display_hdr_info` poll; re-query `Surface::get_capabilities`
//!    and re-run the steps above.
//!
//! The standalone [HDR surface example] implements every step, including the
//! encoding transfer function for each color space.
//!
//! ### What to output from your fragment shader
//!
//! What the values your shader writes to the surface texture must contain for
//! each color space, and whether wgpu applies the transfer function for you:
//!
//! | Color space | Typical format | You write | wgpu encodes? |
//! | ----------- | -------------- | --------- | ------------- |
//! | `Srgb`, `*Srgb` format | `{Rgba,Bgra}8UnormSrgb` | linear | **yes** (hardware sRGB OETF on store) |
//! | `Srgb`, non-srgb format | `{Rgba,Bgra}8Unorm` | sRGB-encoded | no; apply the sRGB OETF yourself or use `*Srgb` instead |
//! | `ExtendedSrgbLinear` (scRGB) | `Rgba16Float` | linear, `1.0` = SDR white | no, but no encoding is necessary |
//! | `ExtendedSrgb` | `Rgba16Float` | extended sRGB-encoded | no; apply the extended sRGB OETF yourself |
//! | `DisplayP3` | `Bgra8Unorm` | sRGB-encoded, P3 primaries | no; apply the sRGB OETF (after gamut-mapping to P3) |
//! | `ExtendedDisplayP3` | `Rgba16Float` | extended sRGB-encoded, P3 primaries | no; apply the extended sRGB OETF (after gamut-mapping to P3) |
//! | `Bt2100Pq` (HDR10) | `Rgb10a2Unorm` | PQ-encoded, BT.2020 primaries | no; apply the PQ OETF (after gamut-mapping to BT.2020) |
//! | `Bt2100Hlg` | `Rgb10a2Unorm` | HLG-encoded, BT.2020 primaries | no; apply the HLG OETF (after gamut-mapping to BT.2020) |
//!
//! In short, wgpu applies the transfer function for you only when you render to
//! an `*Srgb` format. In every other case the values your shader writes to the
//! surface texture must already carry both the transfer function and any gamut
//! conversion. The [HDR surface example] implements every encoder in WGSL.
//!
//! ### Glossary
//!
//! * **Chromaticity** --- a color's hue and saturation independent of its
//!   brightness, given as an `(x, y)` coordinate on the CIE 1931 diagram.
//! * **Primaries / gamut** --- the chromaticities of the red, green, and blue a
//!   color space addresses, and so the range of colors it can express. [BT.709]
//!   is the sRGB gamut, [Display P3] is wider, and [BT.2020] is wider still.
//! * **White point** --- the chromaticity of `R = G = B` (what "white" looks
//!   like). Every color space here uses [D65], standard daylight.
//! * **Transfer function (OETF / EOTF)** --- how stored values map to light. The
//!   *OETF* is the encoding transfer function your application applies; the
//!   *EOTF* is the inverse decoding transfer function the display applies.
//! * **SDR / HDR** --- standard dynamic range clips at `1.0` (reference white);
//!   high dynamic range lets values above `1.0` drive brighter-than-white
//!   output.
//! * **Nits and reference white** --- a *nit* (cd/m²) is a unit of brightness;
//!   *reference white* (also called *paper white*, especially on Windows) is the
//!   brightness of SDR plain white, set below the panel's peak so highlights have
//!   room above it.
//! * **Headroom (EDR)** --- how much brighter than current SDR white the display
//!   can go right now, as a multiplier (`1.0` means none). Dynamic; see
//!   [`DisplayHdrInfo::tone_map_headroom`].
//! * **PQ / HLG** --- the two HDR transfer functions: [PQ] (SMPTE ST 2084,
//!   HDR10) encodes absolute luminance, [HLG] (BT.2100) encodes relative
//!   luminance.
//! * **scRGB / extended-range sRGB** --- sRGB extended past 0.0..=1.0 for
//!   HDR: [scRGB] is *linear*
//!   ([`ExtendedSrgbLinear`](SurfaceColorSpace::ExtendedSrgbLinear)), while
//!   [`ExtendedSrgb`](SurfaceColorSpace::ExtendedSrgb) is the same range but
//!   sRGB-*encoded* (gamma), the web's HDR path.
//!
//! [HDR surface example]: https://github.com/gfx-rs/wgpu/tree/v30/examples/standalone/03_hdr_surface
//! [BT.709]: https://www.itu.int/rec/R-REC-BT.709
//! [BT.2020]: https://www.itu.int/rec/R-REC-BT.2020
//! [Display P3]: https://en.wikipedia.org/wiki/DCI-P3#Display_P3
//! [D65]: https://en.wikipedia.org/wiki/Standard_illuminant#D65_values
//! [PQ]: https://en.wikipedia.org/wiki/Perceptual_quantizer
//! [HLG]: https://www.itu.int/rec/R-REC-BT.2100
//! [scRGB]: https://en.wikipedia.org/wiki/ScRGB
//!

#![no_std]
// `-Znext-solver` requires deeper recursion limits (at least for now) to prove Send/Sync
#![recursion_limit = "256"]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_logo_url = "https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/logo.png")]
#![warn(
    clippy::alloc_instead_of_core,
    clippy::allow_attributes,
    clippy::std_instead_of_alloc,
    clippy::std_instead_of_core,
    missing_docs,
    rust_2018_idioms,
    unsafe_op_in_unsafe_fn
)]
#![allow(
    // We need to investiagate these.
    clippy::large_enum_variant,
    // These degrade readability significantly.
    clippy::bool_assert_comparison,
    clippy::bool_comparison,
)]
// NOTE: Keep this in sync with `wgpu-core`.
#![cfg_attr(not(send_sync), allow(clippy::arc_with_non_send_sync))]
#![cfg_attr(not(any(wgpu_core, webgpu)), allow(unused))]

extern crate alloc;
#[cfg(any(std, test))]
extern crate std;
#[cfg(wgpu_core)]
pub extern crate wgpu_core as wgc;
#[cfg(wgpu_core)]
pub extern crate wgpu_hal as hal;
pub extern crate wgpu_types as wgt;

//
//
// Modules
//
//

mod api;
mod backend;
mod cmp;
mod dispatch;
mod macros;
pub mod util;

//
//
// Public re-exports
//
//

#[cfg(custom)]
pub use backend::custom;

pub use api::*;
pub use wgt::{
    AdapterInfo, AddressMode, AllocatorReport, AstcBlock, AstcChannel, Backend, BackendOptions,
    Backends, BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor, BlendOperation,
    BlendState, BufferAddress, BufferBindingType, BufferSize, BufferTextureCopyInfo,
    BufferTransition, BufferUsages, BufferUses, Color, ColorTargetState, ColorWrites,
    CommandBufferDescriptor, CompareFunction, CompositeAlphaMode, CooperativeMatrixProperties,
    CooperativeScalarType, CopyExternalImageDestInfo, CoreCounters, DepthBiasState,
    DepthStencilState, DeviceLostReason, DeviceType, DisplayChromaticity, DisplayCoarseRange,
    DisplayGamut, DisplayHdrInfo, DisplayHeadroom, DisplayLuminance, DownlevelCapabilities,
    DownlevelFlags, DownlevelLimits, Dx12BackendOptions, Dx12Compiler, Dx12SwapchainKind,
    Dx12UseFrameLatencyWaitableObject, DxcShaderModel, DynamicOffset, ExperimentalFeatures,
    Extent3d, ExternalTextureFormat, ExternalTextureTransferFunction, Face, Features, FeaturesWGPU,
    FeaturesWebGPU, FilterMode, ForceShaderModelToken, FrontFace, GlBackendOptions, GlDebugFns,
    GlFenceBehavior, Gles3MinorVersion, HalCounters, ImageSubresourceRange, IndexFormat,
    InstanceDescriptor, InstanceFlags, InternalCounters, Limits, LoadOpDontCare,
    MemoryBudgetThresholds, MemoryHints, MipmapFilterMode, MultisampleState, NoopBackendOptions,
    Origin2d, Origin3d, PassthroughShaderEntryPoint, PipelineStatisticsTypes, PollError,
    PollStatus, PolygonMode, PowerPreference, PredefinedColorSpace, PresentMode,
    PresentationTimestamp, PrimitiveState, PrimitiveTopology, QueryType, RenderBundleDepthStencil,
    RequestAdapterError, SamplerBindingType, SamplerBorderColor, ShaderLocation, ShaderModel,
    ShaderRuntimeChecks, ShaderStages, StencilFaceState, StencilOperation, StencilState,
    StorageTextureAccess, SurfaceCapabilities, SurfaceColorSpace, SurfaceColorSpaces,
    SurfaceFormatCapabilities, SurfaceStatus, TexelCopyBufferLayout, TextureAspect, TextureChannel,
    TextureDimension, TextureFormat, TextureFormatFeatureFlags, TextureFormatFeatures,
    TextureSampleType, TextureTransition, TextureUsages, TextureUses, TextureViewDimension, Trace,
    VertexAttribute, VertexFormat, VertexStepMode, WasmNotSend, WasmNotSendSync, WasmNotSync,
    WriteOnly, WriteOnlyIter, COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT,
    IMMEDIATE_DATA_ALIGNMENT, MAP_ALIGNMENT, MAXIMUM_SUBGROUP_MAX_SIZE, MINIMUM_SUBGROUP_MIN_SIZE,
    QUERY_RESOLVE_BUFFER_ALIGNMENT, QUERY_SET_MAX_QUERIES, QUERY_SIZE, VERTEX_ALIGNMENT,
};

#[expect(deprecated)]
pub use wgt::VERTEX_STRIDE_ALIGNMENT;

// wasm-only types, we try to keep as many types non-platform
// specific, but these need to depend on web-sys.
#[cfg(web)]
pub use wgt::{CopyExternalImageSourceInfo, ExternalImageSource};

/// Re-export of our `naga` dependency.
///
#[cfg(wgpu_core)]
#[cfg_attr(docsrs, doc(cfg(any(wgpu_core, naga))))]
// We re-export wgpu-core's re-export of naga, as we may not have direct access to it.
pub use ::wgc::naga;
/// Re-export of our `naga` dependency.
///
#[cfg(all(not(wgpu_core), naga))]
#[cfg_attr(docsrs, doc(cfg(any(wgpu_core, naga))))]
// If that's not available, we re-export our own.
pub use naga;

/// Re-export of our `raw-window-handle` dependency.
///
pub use raw_window_handle as rwh;

/// Re-export of our `web-sys` dependency.
///
#[cfg(web)]
pub use web_sys;

/// Vendored WebGPU JS-handle types used by the WebGPU backend.
///
/// They are exposed publicly so that interop crates can read the JS handle
/// behind a [`Texture`] / [`Buffer`] / etc. (via [`Texture::as_webgpu`] and
/// siblings), and pass a foreign handle in (via
/// [`Device::create_texture_from_webgpu_handle`]).
///
/// A `web_sys::GpuTexture` from a consumer's own `web-sys` dependency wraps
/// the same JS object as a `wgpu::webgpu::GpuTexture`; convert between them
/// with [`wasm_bindgen::JsCast::unchecked_into`].
#[cfg(webgpu)]
pub mod webgpu {
    pub use crate::backend::webgpu::webgpu_sys::{
        GpuBuffer, GpuDevice, GpuQueue, GpuTexture, GpuTextureView,
    };
    pub use crate::backend::webgpu::DropCallback;
}

#[doc(hidden)]
pub use macros::helpers as __macro_helpers;
