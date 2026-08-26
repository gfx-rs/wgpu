//! This crate contains types for the remoting version of wgpu-core for browser implementing WebGPU.
//!
//! It contains types that are used for IPC communication between the browser's untrusted content process
//! and the browser's trusted GPU process and [`IdentityHub`](crate::identity::IdentityHub) for Id generation.
//! All in all it contains all types needed for content process.
//!
//! All IPC types implement `serde::Serialize` and `serde::Deserialize` so that they can be sent over IPC.
//!
//! Types are defined entirely separately from wgpu-core's (and eventually wgpu-types's) public types,
//! so that even as experimental features like raytracing and native wpgu features that are not standard WebGPU,
//! cannot be expressed in untrusted (thus potentially malicious) content processes,
//! as Serde's standard deserialization rejects invalid values of the IPC types.
//!
//! For more information about the remoting architecture, see the wgpu-core-remote crate's documentation.
extern crate alloc;
extern crate wgpu_types as wgt;

use alloc::borrow::Cow;

pub type Index = u32;
pub type Epoch = u32;
pub type SubmissionIndex = u64;
pub type SubmittedWorkDoneClosure = Box<dyn FnOnce() + Send + 'static>;

pub mod id;
pub mod identity;

pub mod binding_model;
pub mod encoders;
pub mod ffi;
pub mod pipelines;

pub type Label<'a> = Option<Cow<'a, str>>;

/// Options for requesting adapter.
///
/// Corresponds to [WebGPU `GPURequestAdapterOptions`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions).
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RequestAdapterOptions {
    /// Power preference for the adapter.
    pub power_preference: wgt::PowerPreference,
    /// Indicates that only a fallback adapter can be returned. This is generally a "software"
    /// implementation on the system.
    pub force_fallback_adapter: bool,
}

assert_ffi_safe!(RequestAdapterOptions);

pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;
pub type QueueDescriptor<'a> = wgt::QueueDescriptor<Label<'a>>;
pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;
pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>, Vec<wgt::TextureFormat>>;
pub type ExternalTextureDescriptor<'a> = wgt::ExternalTextureDescriptor<Label<'a>>;

/// Describes a `TextureView`.
///
/// Corresponds to [WebGPU `GPUTextureViewDescriptor`](https://gpuweb.github.io/gpuweb/#dictdef-gputextureviewdescriptor).
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TextureViewDescriptor<'a> {
    /// Debug label of the texture view.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Format of the texture view, or `None` for the same format as the texture
    /// itself.
    ///
    /// At this time, it must be the same the underlying format of the texture.
    pub format: Option<wgt::TextureFormat>,
    /// The dimension of the texture view.
    ///
    /// - For 1D textures, this must be `D1`.
    /// - For 2D textures it must be one of `D2`, `D2Array`, `Cube`, or `CubeArray`.
    /// - For 3D textures it must be `D3`.
    pub dimension: Option<wgt::TextureViewDimension>,
    /// The allowed usage(s) for the texture view. Must be a subset of the usage flags of the texture.
    /// If not provided, defaults to the full set of usage flags of the texture.
    pub usage: Option<wgt::TextureUsages>,
    /// Range within the texture that is accessible via this view.
    pub range: wgt::ImageSubresourceRange,
}

/// Describes a `Sampler`
///
/// Corresponds to [WebGPU `GPUSamplerDescriptor`](https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerdescriptor).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SamplerDescriptor<'a> {
    /// Debug label of the sampler.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// How to deal with out of bounds accesses in the u (i.e. x) direction
    pub address_modes: [wgt::AddressMode; 3],
    /// How to filter the texture when it needs to be magnified (made larger)
    pub mag_filter: wgt::FilterMode,
    /// How to filter the texture when it needs to be minified (made smaller)
    pub min_filter: wgt::FilterMode,
    /// How to filter between mip map levels
    pub mipmap_filter: wgt::MipmapFilterMode,
    /// Minimum level of detail (i.e. mip level) to use
    pub lod_min_clamp: f32,
    /// Maximum level of detail (i.e. mip level) to use
    pub lod_max_clamp: f32,
    /// If this is enabled, this is a comparison sampler using the given comparison function.
    pub compare: Option<wgt::CompareFunction>,
    /// Must be at least 1. If this is not 1, all filter modes must be linear.
    pub anisotropy_clamp: u16,
}

/// Describes a pipeline layout.
///
/// A `PipelineLayoutDescriptor` can be used to create a pipeline layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// cbindgen:ignore
pub struct PipelineLayoutDescriptor<'a> {
    /// Debug label of the pipeline layout.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Bind groups that this pipeline uses. The first entry will provide all the bindings for
    /// "set = 0", second entry will provide all the bindings for "set = 1" etc.
    pub bind_group_layouts: Cow<'a, [Option<id::BindGroupLayoutId>]>,
    /// The number of bytes of immediate data that are allocated for use
    /// in the shader. The `var<immediate>`s in the shader attached to
    /// this pipeline must be equal or smaller than this size.
    ///
    /// If this value is non-zero, [`wgt::Features::IMMEDIATES`] must be enabled.
    pub immediate_size: u32,
}

/// Corresponds to [WebGPU `GPUShaderModuleDescriptor`](https://gpuweb.github.io/gpuweb/#dictdef-gpushadermoduledescriptor).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShaderModuleDescriptor<'a> {
    pub label: Label<'a>,
    pub code: Cow<'a, str>,
}

pub type QuerySetDescriptor<'a> = wgt::QuerySetDescriptor<Label<'a>>;
