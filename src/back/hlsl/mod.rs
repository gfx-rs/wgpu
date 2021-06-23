mod keywords;
mod writer;

use std::fmt::Error as FmtError;
use thiserror::Error;

pub use writer::Writer;

/// A HLSL shader model version.
#[allow(non_snake_case, non_camel_case_types)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum ShaderModel {
    V5_0,
    V5_1,
    V6_0,
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
            shader_model: ShaderModel::V5_0,
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
