mod keywords;
mod writer;

use std::fmt::Error as FmtError;
use thiserror::Error;

pub use writer::Writer;

pub const DEFAULT_SHADER_MODEL: u16 = 50;
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct ShaderModel(u16);

impl ShaderModel {
    pub fn new(shader_model: u16) -> Self {
        Self(shader_model)
    }
}

impl Default for ShaderModel {
    fn default() -> Self {
        Self(DEFAULT_SHADER_MODEL)
    }
}

/// Structure that contains the configuration used in the [`Writer`](Writer)
#[derive(Debug, Clone)]
pub struct Options {
    /// The hlsl shader model to be used
    pub shader_model: ShaderModel,
    /// The vertex entry point name in generated shader
    pub vertex_entry_point_name: String,
    /// The fragment entry point name in generated shader
    pub fragment_entry_point_name: String,
    /// The comput entry point name in generated shader
    pub compute_entry_point_name: String,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            shader_model: ShaderModel(50),
            vertex_entry_point_name: String::from("vert_main"),
            fragment_entry_point_name: String::from("frag_main"),
            compute_entry_point_name: String::from("comp_main"),
        }
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    IoError(#[from] FmtError),
    #[error("BuiltIn {0:?} is not supported")]
    UnsupportedShaderModel(ShaderModel),
    #[error("A scalar with an unsupported width was requested: {0:?} {1:?}")]
    UnsupportedScalar(crate::ScalarKind, crate::Bytes),
    #[error("{0}")]
    Unimplemented(String), // TODO: Error used only during development
    #[error("{0}")]
    Custom(String),
}

pub fn write_string(
    module: &crate::Module,
    info: &crate::valid::ModuleInfo,
    options: &Options,
) -> Result<String, Error> {
    let mut w = Writer::new(String::new(), options);
    w.write(module, info)?;
    let output = w.finish();
    Ok(output)
}
