//! HLSL shading language backend
//!
//! # Supported shader model versions:
//! - 5.0
//! - 5.1
//! - 6.0
//!
//! All matrix construction/deconstruction is row based in HLSL. This means that when
//! we construct a matrix from column vectors, our matrix will be implicitly transposed.
//! The inverse transposition happens when we call `[0]` to get the zeroth column vector.
//!
//! Because all of our matrices are implicitly transposed, we flip arguments to `mul`. `mat * vec`
//! becomes `vec * mat`, etc. This acts as the inverse transpose making the results identical.
//!
//! The only time we don't get this implicit transposition is when reading matrices from Uniforms/Push Constants.
//! To deal with this, we add `row_major` to all declarations of matrices in Uniforms/Push Constants.
//!
//! Finally because all of our matrices are transposed, if you use `mat3x4`, it'll become `float4x3` in HLSL.

mod conv;
mod help;
mod keywords;
mod storage;
mod writer;

use std::fmt::Error as FmtError;
use thiserror::Error;

use crate::proc;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct BindTarget {
    pub space: u8,
    pub register: u32,
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

impl crate::ImageDimension {
    fn to_hlsl_str(self) -> &'static str {
        match self {
            Self::D1 => "1D",
            Self::D2 => "2D",
            Self::D3 => "3D",
            Self::Cube => "Cube",
        }
    }
}

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

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
    /// Add special constants to `SV_VertexIndex` and `SV_InstanceIndex`,
    /// to make them work like in Vulkan/Metal, with help of the host.
    pub special_constants_binding: Option<BindTarget>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            shader_model: ShaderModel::V5_1,
            binding_map: BindingMap::default(),
            fake_missing_bindings: true,
            special_constants_binding: None,
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
                register: res_binding.binding,
            }),
            None => Err(EntryPointError::MissingBinding(res_binding.clone())),
        }
    }
}

/// Structure that contains a reflection info
#[derive(Default)]
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

pub struct Writer<'a, W> {
    out: W,
    names: crate::FastHashMap<proc::NameKey, String>,
    namer: proc::Namer,
    /// HLSL backend options
    options: &'a Options,
    /// Information about entry point arguments and result types.
    entry_point_io: Vec<writer::EntryPointInterface>,
    /// Set of expressions that have associated temporary variables
    named_expressions: crate::NamedExpressions,
    wrapped_array_lengths: crate::FastHashSet<help::WrappedArrayLength>,
    wrapped_image_queries: crate::FastHashSet<help::WrappedImageQuery>,
    temp_access_chain: Vec<storage::SubAccess>,
}
