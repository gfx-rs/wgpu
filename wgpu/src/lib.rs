//! A cross-platform graphics and compute library based on [WebGPU](https://gpuweb.github.io/gpuweb/).
//!
//! To start using the API, create an [`Instance`].
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
//! - **`naga`** ---- Enabled when any non-wgsl shader input is enabled.
//!

#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![doc(html_logo_url = "https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/logo.png")]
#![warn(missing_docs, rust_2018_idioms, unsafe_op_in_unsafe_fn)]

//
//
// Modules
//
//

mod api;
mod backend;
mod context;
mod macros;
mod send_sync;
pub mod util;

/// Module to add ray tracing support to wgpu.
/// It adds support for acceleration structures and ray queries.
/// The features [`Features::RAY_QUERY`] and [`Features::RAY_TRACING_ACCELERATION_STRUCTURE`] where added and are required to use this module.
/// This is an experimental feature and is not available on all backends.
/// It is not part of the WebGPU standard and is only natively supported with the Vulkan backend so far.
/// DirectX 12 and Metal support are planned, the API is subject to change.
/// (It should be somewhat stable, unless adding ray tracing to the WebGPU standard requires deep changes.)
///
/// Functions to create and build acceleration structures are provided, In traits on the [`Device`] and [`CommandEncoder`].
/// To use ray queries, add a top level acceleration structure to a bind group, and use the ray query extension in a shader.
/// Naga (our shader compiler) has support for the ray query extension for "wgsl" shaders.
/// For more details see the examples (starting with ray-).
pub mod ray_tracing;

//
//
// Private re-exports
//
//

#[allow(unused_imports)] // WebGPU needs this
use context::Context;
use send_sync::*;

type C = dyn context::DynContext;
#[cfg(send_sync)]
type Data = dyn std::any::Any + Send + Sync;
#[cfg(not(send_sync))]
type Data = dyn std::any::Any;

//
//
// Public re-exports
//
//

pub use api::*;
pub use wgt::{
    AdapterInfo, AddressMode, AstcBlock, AstcChannel, Backend, Backends, BindGroupLayoutEntry,
    BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState, BufferAddress,
    BufferBindingType, BufferSize, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandBufferDescriptor, CompareFunction, CompositeAlphaMode, CoreCounters, DepthBiasState,
    DepthStencilState, DeviceLostReason, DeviceType, DownlevelCapabilities, DownlevelFlags,
    Dx12Compiler, DynamicOffset, Extent3d, Face, Features, FilterMode, FrontFace,
    Gles3MinorVersion, HalCounters, ImageDataLayout, ImageSubresourceRange, IndexFormat,
    InstanceDescriptor, InstanceFlags, InternalCounters, Limits, MaintainResult, MemoryHints,
    MultisampleState, Origin2d, Origin3d, PipelineStatisticsTypes, PolygonMode, PowerPreference,
    PredefinedColorSpace, PresentMode, PresentationTimestamp, PrimitiveState, PrimitiveTopology,
    PushConstantRange, QueryType, RenderBundleDepthStencil, SamplerBindingType, SamplerBorderColor,
    ShaderLocation, ShaderModel, ShaderStages, StencilFaceState, StencilOperation, StencilState,
    StorageTextureAccess, SurfaceCapabilities, SurfaceStatus, TextureAspect, TextureDimension,
    TextureFormat, TextureFormatFeatureFlags, TextureFormatFeatures, TextureSampleType,
    TextureUsages, TextureViewDimension, VertexAttribute, VertexFormat, VertexStepMode,
    WasmNotSend, WasmNotSendSync, WasmNotSync, COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT,
    MAP_ALIGNMENT, PUSH_CONSTANT_ALIGNMENT, QUERY_RESOLVE_BUFFER_ALIGNMENT, QUERY_SET_MAX_QUERIES,
    QUERY_SIZE, VERTEX_STRIDE_ALIGNMENT,
};
// wasm-only types, we try to keep as many types non-platform
// specific, but these need to depend on web-sys.
#[cfg(any(webgpu, webgl))]
pub use wgt::{ExternalImageSource, ImageCopyExternalImage};

//
//
// Re-exports of dependencies
//
//

/// Re-export of our `wgpu-core` dependency.
///
#[cfg(wgpu_core)]
pub use ::wgc as core;

/// Re-export of our `wgpu-hal` dependency.
///
///
#[cfg(wgpu_core)]
pub use ::hal;

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
#[cfg(any(webgl, webgpu))]
pub use web_sys;
