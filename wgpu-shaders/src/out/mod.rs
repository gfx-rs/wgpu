#[cfg(feature = "glsl-out")]
pub mod glsl;
#[cfg(feature = "msl-out")]
pub mod msl;
#[cfg(feature = "spv-out")]
pub mod spv;

pub enum ShaderCompilationError {
    EntryPoint,
    PipelineConstants(String),
    Linkage(String),
}
