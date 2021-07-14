//! HLSL shading language backend
//!
//! # Supported shader model versions:
//! - 5.0
//! - 5.1
//! - 6.0
//!

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

impl ShaderModel {
    pub fn to_str(self) -> &'static str {
        match self {
            Self::V5_0 => "5_0",
            Self::V5_1 => "5_1",
            Self::V6_0 => "6_0",
        }
    }
}

impl crate::ShaderStage {
    pub fn to_hlsl_str(self) -> &'static str {
        match self {
            Self::Vertex => "vs",
            Self::Fragment => "ps",
            Self::Compute => "cs",
        }
    }
}

/// Structure that contains the configuration used in the [`Writer`](Writer)
#[derive(Debug, Clone)]
pub struct Options {
    /// The hlsl shader model to be used
    pub shader_model: ShaderModel,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            shader_model: ShaderModel::V5_0,
        }
    }
}

/// Structure that contains a reflection info
pub struct ReflectionInfo {
    /// Real name of entry point allowed by the `hlsl` compiler.
    /// For example:
    /// the entry point with the name `line` is valid for `wgsl`, but not valid for `hlsl`, because `line` is a reserved keyword.
    pub entry_points: Vec<String>,
    // TODO: locations
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
