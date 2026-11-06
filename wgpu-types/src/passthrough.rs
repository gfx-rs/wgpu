//! Passthrough shader descriptors

// TODO: undo this
#![allow(missing_docs)]

use alloc::borrow::Cow;
use nt::ShaderStage;

use crate::{Features, PrimitiveTopology};

/// An entry point may be subject to naming restrictions depending on the language used.
///
/// This is used so that each backend can produce the same entry point under different
/// names.
#[derive(Debug, Clone)]
pub struct EntryPointMapping<'a, I: core::fmt::Debug + Clone + Default> {
    /// The name of the entry point as it appears in the passthrough source.
    pub name_in_source: Cow<'a, str>,
    /// The name of the entry point as it will be referred to when using the created
    /// shader module.
    pub name_in_shader_module: Cow<'a, str>,
    /// Extra information about the entry point, differs per-backend.
    pub entry_point_info: I,
}

#[derive(Debug, Clone, Default)]
pub struct Dx12EntryPointInfo {
    // TODO: populate
}

/// Passthrough from an HLSL string
#[derive(Debug, Clone)]
pub struct HlslSource<'a> {
    /// The HLSL source to pass through
    pub source: Cow<'a, str>,
    /// The list of entry point names as they will be referred to after
    /// shader module creation.
    pub entry_point_map: Cow<'a, [EntryPointMapping<'a, Dx12EntryPointInfo>]>,

    pub shader_model: (u8, u8),
}

/// Passthrough from a DXIL blob
#[derive(Debug, Clone)]
pub struct DxilSource<'a> {
    /// The DXIL source to pass through
    pub source: Cow<'a, [u8]>,
    /// The name of the entry point as it will be referred to afte shader
    /// module creation.
    pub entry_point: Cow<'a, str>,
    /// Extra information about the entry point
    pub info: Dx12EntryPointInfo,

    pub shader_model: (u8, u8),
}

/// Passthrough shader source for the DX12 backend
#[derive(Debug, Clone)]
pub enum Dx12Source<'a> {
    /// HLSL passthrough
    Hlsl(Cow<'a, [HlslSource<'a>]>),
    /// DXIL passthrough
    Dxil(Cow<'a, [DxilSource<'a>]>),
}

#[derive(Debug, Clone, Default)]
pub struct VulkanEntryPointInfo {
    // TODO: populate
}

/// Passthrough from a SPIR-V blob
#[derive(Debug, Clone)]
pub struct SpirvSource<'a> {
    /// The SPIR-V source to pass through
    pub source: Cow<'a, [u32]>,
    /// The list of entry point names as they will be referred to after
    /// shader module creation.
    pub entry_point_map: Cow<'a, [EntryPointMapping<'a, VulkanEntryPointInfo>]>,

    /// Only used for validation
    pub spirv_version: (u8, u8),
    /// Only used for validation
    pub needed_capabilities: Cow<'a, [u32]>,
    /// Only used for validation
    pub needed_extensions: Cow<'a, [Cow<'a, str>]>,
}

/// Passthrough shader source for the Vulkan backend
#[derive(Debug, Clone)]
pub struct VulkanSource<'a> {
    /// The list of SPIR-V blobs
    pub sources: Cow<'a, [SpirvSource<'a>]>,
}

#[derive(Debug, Clone, Default)]
pub struct GlesEntryPointInfo {
    // TODO: populate
}

/// Passthrough from a GLSL string
#[derive(Debug, Clone)]
pub struct GlslSource<'a> {
    /// The GLSL source to pass through
    pub source: Cow<'a, str>,
    /// The name of the entry point as it will be referred to afte shader
    /// module creation.
    pub entry_point: Cow<'a, str>,
    /// Extra information about the entry point
    pub info: GlesEntryPointInfo,

    pub version: nt::glsl::Version,
}

/// Passthrough shader source for the GL/GLES backend
#[derive(Debug, Clone)]
pub struct GlesSource<'a> {
    /// The list of GLSL strings
    pub sources: Cow<'a, [GlslSource<'a>]>,
}

#[derive(Debug, Clone, Default)]
pub struct MetalEntryPointInfo {
    // TODO: populate
}

/// Passthrough from a Metal Shading Language string
#[derive(Debug, Clone)]
pub struct MslSource<'a> {
    /// The Metal Shading Language source to pass through
    pub source: Cow<'a, str>,
    /// The list of entry point names as they will be referred to after
    /// shader module creation.
    pub entry_point_map: Cow<'a, [EntryPointMapping<'a, MetalEntryPointInfo>]>,

    /// The MSL language version/metal API version required
    pub metal_version: (u8, u8),
}

/// Passthrough from a metallib blob
#[derive(Debug, Clone)]
pub struct MetallibSource<'a> {
    /// The metallib source to pass through
    pub source: Cow<'a, [u8]>,
    /// The list of entry point names as they will be referred to after
    /// shader module creation.
    pub entry_point_map: Cow<'a, [EntryPointMapping<'a, MetalEntryPointInfo>]>,

    /// The MSL language version/metal API version required
    pub metal_version: (u8, u8),
}

/// Passthrough shader source for the Metal backend
#[derive(Debug, Clone)]
pub enum MetalSource<'a> {
    /// Metal Shading Language passthrough
    Msl(Cow<'a, [MslSource<'a>]>),
    /// Metallib passthrough
    Metallib(Cow<'a, [MetallibSource<'a>]>),
}

#[derive(Debug, Clone, Default)]
pub struct WebGPUEntryPointInfo {
    // TODO: populate
}

/// Passthrough from a WGSL string
#[derive(Debug, Clone)]
pub struct WgslSource<'a> {
    /// The WGSL source to pass through
    pub source: Cow<'a, str>,
    /// The list of entry point names as they will be referred to after
    /// shader module creation.
    pub entry_point_map: Cow<'a, [EntryPointMapping<'a, WebGPUEntryPointInfo>]>,

    pub enable_extensions: Cow<'a, [Cow<'a, str>]>,
    pub language_extensions: Cow<'a, [Cow<'a, str>]>,
}

/// Passthrough shader source for the WebGPU backend
#[derive(Debug, Clone)]
pub struct WebGPUSource<'a> {
    /// The list of WGSL strings
    pub sources: Cow<'a, [WgslSource<'a>]>,
}

#[derive(Debug, Clone)]
pub enum BackendSource<'a> {
    Vulkan(VulkanSource<'a>),
    Dx12(Dx12Source<'a>),
    Metal(MetalSource<'a>),
    Gles(GlesSource<'a>),
    WebGPU(WebGPUSource<'a>),
}

/// The sources for a passthrough shader.
///
/// The first usable source is taken.
#[derive(Debug, Clone, Default)]
pub struct PassthroughShaderSource<'a> {
    pub sources: Cow<'a, [BackendSource<'a>]>,
}

#[derive(Debug, Clone)]
pub struct PassthroughEntryPointInfo<'a> {
    /// The entry point's name, as it will be used from the resulting shader module.
    pub name: Cow<'a, str>,
    /// The shader stage.
    pub stage: ShaderStage,
    /// Only relevant for some shader stages.
    ///
    /// Necessary for Metal backend passthrough. Therefore required for all relevant stages.
    pub workgroup_size: Option<(u32, u32, u32)>,
    /// Only necessary for vertex/mesh shader output.
    ///
    /// Necessary for GLES backend passthrough.
    pub clip_distance_count: u32,

    /// Information about the entry point interface, and validation information.
    pub interface: PassthroughEntryPointInterfaceInfo<'a>,
}

#[derive(Debug, Clone, Default)]
pub struct PassthroughEntryPointInterfaceInfo<'a> {
    /// The features this shader relies on.
    pub used_features: Features,
    /// List of inputs
    pub inputs: Option<Cow<'a, [Varying]>>,
    /// List of outputs
    pub outputs: Option<Cow<'a, [Varying]>>,
    /// Size in bytes of the immediate data
    pub immediate_data_size: u32,
    /// Array length of the clip distances builtin. On GLES, these are needed for
    /// clip distances to be usable.
    pub clip_disances_array_length: u32,
    /// List of used combinations of textures and samplers.
    pub texture_sampler_pairs: Cow<'a, [(nt::ResourceBinding, nt::ResourceBinding)]>,
    /// Needed for derived pipeline layouts.
    pub bindings: Option<ResourceBindings>,
    /// Only relevant for mesh and task shaders. A size of zero means no payload.
    pub task_payload_size: Option<u32>,
    pub total_workgroup_memory: u32,
    pub mesh_info: Option<MeshEntryPointInfo>,
}

#[derive(Debug, Clone)]
pub enum Varying {
    UserDefined {
        location: u32,
        // TODO: replace with NumericType
        ty: nt::VertexFormat,
        // TODO: interpolation, sampling mode, per-primitive, blend source index
    },
    Builtin {
        // TODO: builtin
    },
}

#[derive(Debug, Clone)]
pub struct ResourceBindings {
    // TODO: populate
}

#[derive(Debug, Clone)]
pub struct MeshEntryPointInfo {
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub topology: PrimitiveTopology,
}

#[derive(Debug, Clone)]
pub struct PassthroughShaderDescriptor<'a, L> {
    pub label: L,
    pub entry_points: Cow<'a, [PassthroughEntryPointInfo<'a>]>,
    pub source: PassthroughShaderSource<'a>,
}
