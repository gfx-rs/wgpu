use alloc::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::ffi::FfiOption;
use crate::id::{BindGroupLayoutId, BufferId, ExternalTextureId, SamplerId, TextureViewId};
use crate::{assert_ffi_safe, Label};

/// Corresponds to [`GPUBufferBinding`](https://www.w3.org/TR/webgpu/#dictdef-gpubufferbinding).
#[repr(C)]
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct BufferBinding {
    pub buffer: BufferId,
    pub offset: wgt::BufferAddress,

    /// Size of the binding. If `None`, the binding spans from `offset` to the
    /// end of the buffer.
    ///
    /// We use `BufferAddress` to allow a size of zero on this `wgpu_core` type,
    /// because JavaScript bindings cannot readily express `Option<NonZeroU64>`.
    /// The `wgpu` API uses `Option<BufferSize>` (i.e. `NonZeroU64`) for this
    /// field.
    pub size: FfiOption<wgt::BufferAddress>,
}

assert_ffi_safe!(BufferBinding);

/// Corresponds to [`GPUBindingResource`](https://www.w3.org/TR/webgpu/#typedefdef-gpubindingresource).
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BindingResource {
    Buffer(BufferBinding),
    Sampler(SamplerId),
    TextureView(TextureViewId),
    ExternalTexture(ExternalTextureId),
}

assert_ffi_safe!(BindingResource);

/// Bindable resource and the slot to bind it to.
///
/// This corresponds to [`GPUBindGroupEntry`](https://www.w3.org/TR/webgpu/#dictdef-gpubindgroupentry).
#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupEntry {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    pub resource: BindingResource,
}

assert_ffi_safe!(BindGroupEntry);

/// Describes a group of bindings and the resources to be bound.
///
/// This corresponds to [`GPUBindGroupDescriptor`](https://www.w3.org/TR/webgpu/#dictdef-gpubindgroupdescriptor).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupDescriptor<'a> {
    /// Debug label of the bind group.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The BindGroupLayout that corresponds to this bind group.
    pub layout: BindGroupLayoutId,
    /// The resources to bind to this bind group.
    pub entries: Cow<'a, [BindGroupEntry]>,
}

/// A buffer binding.
///
/// Corresponds to [WebGPU `GPUBufferBindingLayout`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubufferbindinglayout).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct BufferBindingLayout {
    /// Sub-type of the buffer binding.
    pub ty: wgt::BufferBindingType,

    /// Indicates that the binding has a dynamic offset.
    ///
    /// One offset must be passed to [`RenderPass::set_bind_group`][RPsbg]
    /// for each dynamic binding in increasing order of binding number.
    ///
    #[serde(default)]
    pub has_dynamic_offset: bool,

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
    /// [minimum buffer binding size]: https://www.w3.org/TR/webgpu/#minimum-buffer-binding-size
    #[serde(default)]
    pub min_binding_size: Option<wgt::BufferSize>,
}

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SamplerBindingLayout {
    /// Type of a sampler binding.
    pub ty: wgt::SamplerBindingType,
}

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct TextureBindingLayout {
    /// Sample type of the texture binding.
    pub sample_type: wgt::TextureSampleType,
    /// Dimension of the texture view that is going to be sampled.
    pub view_dimension: wgt::TextureViewDimension,
    /// True if the texture has a sample count greater than 1. If this is true,
    /// the texture must be declared as `texture_multisampled_2d` or
    /// `texture_depth_multisampled_2d` in the shader, and read using `textureLoad`.
    pub multisampled: bool,
}

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct StorageTextureBindingLayout {
    /// Allowed access to this texture.
    pub access: wgt::StorageTextureAccess,
    /// Format of the texture.
    pub format: wgt::TextureFormat,
    /// Dimension of the texture view that is going to be sampled.
    pub view_dimension: wgt::TextureViewDimension,
}

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AccelerationStructureBindingLayout {
    /// Whether this acceleration structure can be used to
    /// create a ray query that has flag vertex return in the shader
    ///
    /// If enabled requires [`Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN`]
    pub vertex_return: bool,
}

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ExternalTextureBindingLayout;

/// Describes a single binding inside a bind group.
///
/// Corresponds to [WebGPU `GPUBindGroupLayoutEntry`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgrouplayoutentry).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BindGroupLayoutEntry {
    /// Binding index. Must match shader index and be unique inside a `BindGroupLayout`. A binding
    /// of index 1, would be described as `@group(0) @binding(1)` in shaders.
    pub binding: u32,
    /// Which shader stages can see this binding.
    pub visibility: wgt::ShaderStagesWebGPU,
    // The type of the binding
    pub buffer: Option<BufferBindingLayout>,
    pub sampler: Option<SamplerBindingLayout>,
    pub texture: Option<TextureBindingLayout>,
    pub storage_texture: Option<StorageTextureBindingLayout>,
    pub external_texture: Option<ExternalTextureBindingLayout>,
}

/// Corresponds to [`GPUBindGroupLayoutDescriptor`](https://www.w3.org/TR/webgpu/#dictdef-gpubindgrouplayoutdescriptor).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Array of entries in this BindGroupLayout
    pub entries: Cow<'a, [BindGroupLayoutEntry]>,
}
