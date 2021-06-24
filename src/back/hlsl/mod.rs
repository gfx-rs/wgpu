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
    pub fn to_profile_string(self, stage: crate::ShaderStage) -> String {
        let stage_prefix = match stage {
            crate::ShaderStage::Vertex => "vs_",
            crate::ShaderStage::Fragment => "ps_",
            crate::ShaderStage::Compute => "cs_",
        };

        let version = match self {
            Self::V5_0 => "5_0",
            Self::V5_1 => "5_1",
            Self::V6_0 => "6_0",
        };

        format!("{}{}", stage_prefix, version)
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

pub struct ReflectionInfo {
    /// Information about all entry points (stage, name).
    pub entry_points: Vec<(crate::ShaderStage, String)>,
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
