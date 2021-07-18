//! HLSL shading language backend
//!
//! # Supported shader model versions:
//! - 5.0
//! - 5.1
//! - 6.0
//!

mod image;
mod keywords;
mod writer;

use std::fmt::Error as FmtError;
use thiserror::Error;

pub use writer::Writer;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct BindTarget {
    pub space: u8,
    pub register: u8,
}

// Using `BTreeMap` instead of `HashMap` so that we can hash itself.
pub type BindingMap = std::collections::BTreeMap<crate::ResourceBinding, BindTarget>;

/// A HLSL shader model version.
#[allow(non_snake_case, non_camel_case_types)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
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

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum EntryPointError {
    #[error("mapping of {0:?} is missing")]
    MissingBinding(crate::ResourceBinding),
}

/// Structure that contains the configuration used in the [`Writer`](Writer)
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct Options {
    /// The hlsl shader model to be used
    pub shader_model: ShaderModel,
    /// Map of resources association to binding locations.
    pub binding_map: BindingMap,
    /// Don't panic on missing bindings, instead generate any HLSL.
    pub fake_missing_bindings: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            shader_model: ShaderModel::V5_0,
            binding_map: BindingMap::default(),
            fake_missing_bindings: true,
        }
    }
}

impl Options {
    fn resolve_resource_binding(
        &self,
        res_binding: &crate::ResourceBinding,
    ) -> Result<BindTarget, EntryPointError> {
        match self.binding_map.get(res_binding) {
            Some(target) => Ok(target.clone()),
            None if self.fake_missing_bindings => Ok(BindTarget {
                space: res_binding.group as u8,
                register: res_binding.binding as u8,
            }),
            None => Err(EntryPointError::MissingBinding(res_binding.clone())),
        }
    }
}

/// Structure that contains a reflection info
pub struct ReflectionInfo {
    /// Mapping of the entry point names. Each item in the array
    /// corresponds to an entry point index. The real entry point name may be different if one of the
    /// reserved words are used.
    ///
    ///Note: Some entry points may fail translation because of missing bindings.
    pub entry_point_names: Vec<Result<String, EntryPointError>>,
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
