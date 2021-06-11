mod keywords;
mod writer;

use std::fmt::Error as FmtError;
use thiserror::Error;

pub use writer::Writer;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct ShaderModel(u16);

impl Default for ShaderModel {
    fn default() -> Self {
        ShaderModel(50)
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
    #[error("BuiltIn {0:?} is not supported")]
    UnsupportedBuiltIn(crate::BuiltIn),
    #[error("{0}")]
    Unimplemented(String), // TODO: Error used only during development
}

pub fn write_string(
    module: &crate::Module,
    info: &crate::valid::ModuleInfo,
    shader_model: ShaderModel,
) -> Result<String, Error> {
    let mut w = Writer::new(String::new(), shader_model);
    w.write(module, info)?;
    let output = w.finish();
    Ok(output)
}
