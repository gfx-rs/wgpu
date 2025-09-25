#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wgpu_types::{
    BindGroupLayoutEntry, CreateShaderModuleDescriptorPassthrough, PushConstantRange,
    ShaderRuntimeChecks,
};

// Reexport certain configuration options so that not every user has to also depend on naga themselves
#[cfg(feature = "glsl-out")]
pub use naga::back::glsl;
#[cfg(feature = "hlsl-out")]
pub use naga::back::hlsl;
#[cfg(feature = "msl-out")]
pub use naga::back::msl;
#[cfg(feature = "spv-out")]
pub use naga::back::spv;
#[cfg(feature = "wgsl-out")]
pub use naga::back::wgsl;

/// A range of push constant memory to pass to a shader stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PipelineLayout<
    'a,
    A: IntoIterator<Item = &'a BindGroupLayoutEntry>,
    B: IntoIterator<Item = A>,
    C: IntoIterator<Item = &'a PushConstantRange>,
> {
    pub bind_groups: B,
    pub push_constant_ranges: C,
}

pub enum PrecompiledShaderInput<'a> {
    #[cfg(feature = "wgsl-in")]
    Wgsl(&'a str),
    #[cfg(feature = "spv-in")]
    Spirv(&'a [u32]),
    #[cfg(feature = "glsl-in")]
    Glsl(&'a str),
    Naga(&'a naga::Module, &'a naga::valid::ModuleInfo),
}

pub enum Dx12OutputMode {
    Hlsl,
    Dxc,
    Fxc,
}

pub struct CompileShaderDescriptor<'a> {
    pub input: PrecompiledShaderInput<'a>,
    pub entry_point: String,
    pub runtime_checks: Option<ShaderRuntimeChecks>,
}

pub struct PipelineCompileInfo<
    'a,
    A: IntoIterator<Item = &'a BindGroupLayoutEntry>,
    B: IntoIterator<Item = A>,
    C: IntoIterator<Item = &'a PushConstantRange>,
> {
    pub layout: &'a PipelineLayout<'a, A, B, C>,
    /// Leave as None to exclude spirv output
    #[cfg(feature = "spv-out")]
    pub spirv_options: Option<naga::back::spv::Options<'a>>,
    /// Leave as None to exclude wgsl output
    #[cfg(feature = "wgsl-out")]
    pub wgsl_options: Option<naga::back::wgsl::WriterFlags>,
    /// Leave as None to exclude glsl output
    #[cfg(feature = "glsl-out")]
    pub glsl_options: Option<naga::back::glsl::Options>,
    /// Leave as None to exclude hlsl/dxil output
    #[cfg(feature = "hlsl-out")]
    pub hlsl_options: Option<naga::back::hlsl::Options>,
    /// Leave as None to exclude msl output
    #[cfg(feature = "msl-out")]
    pub msl_options: Option<naga::back::msl::Options>,
    #[cfg(feature = "hlsl-out")]
    /// If `hlsl_options` is not None, whether or not to compile the outputted HLSL into DXIL
    pub dx12_output_mode: Dx12OutputMode,
}

pub struct CompileRenderPipelineInfo<
    'a,
    A: IntoIterator<Item = &'a BindGroupLayoutEntry>,
    B: IntoIterator<Item = A>,
    C: IntoIterator<Item = &'a PushConstantRange>,
> {
    pub vertex: CompileShaderDescriptor<'a>,
    pub fragment: Option<CompileShaderDescriptor<'a>>,
    pub pipeline_info: PipelineCompileInfo<'a, A, B, C>,
}
pub struct CompiledRenderPipelineShaders<'a> {
    pub vertex: CreateShaderModuleDescriptorPassthrough<'a, ()>,
    pub fragment: Option<CreateShaderModuleDescriptorPassthrough<'a, ()>>,
}

pub struct CompileComputePipelineInfo<
    'a,
    A: IntoIterator<Item = &'a BindGroupLayoutEntry>,
    B: IntoIterator<Item = A>,
    C: IntoIterator<Item = &'a PushConstantRange>,
> {
    pub compute: CompileShaderDescriptor<'a>,
    pub pipeline_info: PipelineCompileInfo<'a, A, B, C>,
}
pub struct CompiledComputePipelineShaders<'a> {
    pub compute: CreateShaderModuleDescriptorPassthrough<'a, ()>,
}

pub struct CompileMeshPipelineInfo<
    'a,
    A: IntoIterator<Item = &'a BindGroupLayoutEntry>,
    B: IntoIterator<Item = A>,
    C: IntoIterator<Item = &'a PushConstantRange>,
> {
    pub task: Option<CompileShaderDescriptor<'a>>,
    pub mesh: CompileShaderDescriptor<'a>,
    pub fragment: Option<CompileShaderDescriptor<'a>>,
    pub pipeline_info: PipelineCompileInfo<'a, A, B, C>,
}
pub struct CompiledMeshPipelineShaders<'a> {
    pub task: Option<CreateShaderModuleDescriptorPassthrough<'a, ()>>,
    pub mesh: CreateShaderModuleDescriptorPassthrough<'a, ()>,
    pub fragment: Option<CreateShaderModuleDescriptorPassthrough<'a, ()>>,
}

/// Write out the passthrough to literal rust code that can then be parsed by rustc.
pub fn passthrough_descriptor_to_rust_code(
    desc: &CreateShaderModuleDescriptorPassthrough<'_, ()>,
    out: &mut impl std::fmt::Write,
) {
    todo!()
}

/// Error during compiling of shaders
#[derive(Error, Debug)]
pub enum Error {}

pub fn compile_render_pipeline_shaders<
    'a,
    A: IntoIterator<Item = &'a BindGroupLayoutEntry>,
    B: IntoIterator<Item = A>,
    C: IntoIterator<Item = &'a PushConstantRange>,
>(
    info: &CompileRenderPipelineInfo<'a, A, B, C>,
) -> Result<CompiledRenderPipelineShaders<'static>, Error> {
    todo!()
}

pub fn compile_mesh_pipeline_shaders<
    'a,
    A: IntoIterator<Item = &'a BindGroupLayoutEntry>,
    B: IntoIterator<Item = A>,
    C: IntoIterator<Item = &'a PushConstantRange>,
>(
    info: &CompileMeshPipelineInfo<'a, A, B, C>,
) -> Result<CompiledMeshPipelineShaders<'static>, Error> {
    todo!()
}

pub fn compile_compute_pipeline_shaders<
    'a,
    A: IntoIterator<Item = &'a BindGroupLayoutEntry>,
    B: IntoIterator<Item = A>,
    C: IntoIterator<Item = &'a PushConstantRange>,
>(
    info: &CompileComputePipelineInfo<'a, A, B, C>,
) -> Result<CompiledComputePipelineShaders<'static>, Error> {
    todo!()
}
