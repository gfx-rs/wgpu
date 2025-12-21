/*!
Backend for [GLSL][glsl] (OpenGL Shading Language).

[glsl]: https://www.khronos.org/registry/OpenGL/index_gl.php
*/

use crate::{
    back,
    proc::{self, NameKey},
    valid, Handle, ShaderStage, TypeInner,
};
use std::fmt::{Error as FmtError, Write};

pub use writer::Writer;

mod conv;
mod features;
mod keywords;
mod writer;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    FmtError(#[from] FmtError),
    #[error("Custom backend error: {0}")]
    Custom(String),
    #[error("{0}")]
    Validation(#[from] valid::ValidationError),
    #[error("Missing features: {0:?}")]
    MissingFeatures(Vec<Features>),
    #[error("Unsupported external: {0}")]
    UnsupportedExternal(String),
    #[error("Version {0:?} is not supported")]
    VersionNotSupported,
    #[error("Entry point not found")]
    EntryPointNotFound,
    #[error("Override")]
    Override,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Version {
    /** OpenGL 3.3+ */
    Desktop(u16),
    /** OpenGL ES 3.0+ */
    Embedded { version: u16, is_webgl: bool },
}

impl Version {
    /// Returns true if the version is ES
    pub const fn is_es(&self) -> bool {
        match *self {
            Version::Desktop(_) => false,
            Version::Embedded { .. } => true,
        }
    }

    /// Returns true if the version is WebGL
    pub const fn is_webgl(&self) -> bool {
        match *self {
            Version::Embedded { is_webgl, .. } => is_webgl,
            _ => false,
        }
    }

    /// Returns true if the version supports `std140` layout
    pub const fn supports_std140_layout(&self) -> bool {
        match *self {
            Version::Desktop(_) => true,
            Version::Embedded { version, .. } => version >= 300,
        }
    }

    /// Returns true if the version supports `std430` layout
    pub const fn supports_std430_layout(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 430,
            Version::Embedded { version, .. } => version >= 310,
        }
    }

    /// Returns true if the version supports explicit locations
    pub const fn supports_explicit_locations(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 330,
            Version::Embedded { version, .. } => version >= 300,
        }
    }

    /// Returns true if the version supports explicit locations for I/O
    pub const fn supports_io_locations(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 410,
            Version::Embedded { version, .. } => version >= 310,
        }
    }

    /// Returns true if the version supports early depth tests
    pub const fn supports_early_depth_test(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 420,
            Version::Embedded { version, .. } => version >= 310,
        }
    }

    /// Returns true if the version supports derivative control
    pub const fn supports_derivative_control(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 450,
            Version::Embedded { .. } => false,
        }
    }

    /// Returns true if the version supports fma function
    pub const fn supports_fma_function(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 400,
            Version::Embedded { version, .. } => version >= 320,
        }
    }

    /// Returns true if the version supports integer functions
    pub const fn supports_integer_functions(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 400,
            Version::Embedded { version, .. } => version >= 310,
        }
    }

    /// Returns true if the version supports 64-bit integers
    pub const fn supports_64_bit_integers(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 400,
            Version::Embedded { .. } => false,
        }
    }

    /// Returns true if the version supports 64-bit floats
    pub const fn supports_64_bit_floats(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 400,
            Version::Embedded { .. } => false,
        }
    }

    /// Returns true if the version supports texture samples query
    pub const fn supports_texture_samples_query(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 450,
            Version::Embedded { version, .. } => version >= 310,
        }
    }

    /// Returns true if the version supports pack/unpack 4x8 functions
    pub const fn supports_pack_unpack_4x8(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 410,
            Version::Embedded { version, .. } => version >= 310,
        }
    }

    /// Returns true if the version supports pack/unpack snorm 2x16 functions
    pub const fn supports_pack_unpack_snorm_2x16(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 420,
            Version::Embedded { version, .. } => version >= 310,
        }
    }

    /// Returns true if the version supports pack/unpack unorm 2x16 functions
    pub const fn supports_pack_unpack_unorm_2x16(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 410,
            Version::Embedded { version, .. } => version >= 300,
        }
    }

    /// Returns true if the version supports pack/unpack half 2x16 functions
    pub const fn supports_pack_unpack_half_2x16(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 420,
            Version::Embedded { version, .. } => version >= 300,
        }
    }

    /// Returns true if the version supports frexp function
    pub const fn supports_frexp_function(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 400,
            Version::Embedded { .. } => false,
        }
    }

    /// Checks if the version is supported
    pub const fn is_supported(&self) -> bool {
        match *self {
            Version::Desktop(version) => version >= 330,
            Version::Embedded { version, .. } => version >= 300,
        }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Version::Desktop(v) => write!(f, "{} core", v),
            Version::Embedded { version, is_webgl } => {
                if is_webgl {
                    write!(f, "{} es", version)
                } else {
                    write!(f, "{} es", version)
                }
            }
        }
    }
}

bitflags::bitflags! {
    /// Configuration flags for the [`Writer`].
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct WriterFlags: u32 {
        /// Flip the Y coordinate of the vertex position.
        const ADJUST_COORDINATE_SPACE = 0x1;
        /// Support `gl_PointSize` on the vertex shader.
        const FORCE_POINT_SIZE = 0x2;
        /// Include unused global variables and functions.
        const INCLUDE_UNUSED_ITEMS = 0x4;
        /// Draw parameters.
        const DRAW_PARAMETERS = 0x8;
    }
}

/// Configuration used for the [`Writer`].
#[derive(Debug, Clone)]
pub struct Options {
    /// The GLSL version to be used.
    pub version: Version,
    /// The flags to be used.
    pub writer_flags: WriterFlags,
    /// The binding map to be used.
    pub binding_map: back::BindingMap,
    /// Whether to zero-initialize workgroup memory.
    pub zero_initialize_workgroup_memory: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            version: Version::Desktop(330),
            writer_flags: WriterFlags::empty(),
            binding_map: Default::default(),
            zero_initialize_workgroup_memory: true,
        }
    }
}

/// Structure containing the reflection info
pub struct ReflectionInfo {
    /// Mapping of the entry point names to the reflection info
    pub entry_points: crate::FastHashMap<String, EntryPointReflectionInfo>,
    /// Mapping of the global variables to the reflection info
    pub globals: crate::FastHashMap<Handle<crate::GlobalVariable>, GlobalReflectionInfo>,
}

/// Structure containing the reflection info for an entry point
pub struct EntryPointReflectionInfo {
    /// Mapping of the uniform variables to the reflection info
    pub uniforms: crate::FastHashMap<String, u32>,
}

/// Structure containing the reflection info for a global variable
pub struct GlobalReflectionInfo {
    /// The name of the global variable
    pub name: String,
}

/// Structure containing the pipeline options
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    /// The shader stage of the entry point
    pub shader_stage: ShaderStage,
    /// The name of the entry point
    pub entry_point: String,
    /// The multiview configuration
    pub multiview: Option<core::num::NonZeroU32>,
}

use features::Features;

/// Helper function to check if a value can be initialized
fn is_value_init_supported(module: &crate::Module, ty: Handle<crate::Type>) -> bool {
    match module.types[ty].inner {
        TypeInner::Scalar { .. }
        | TypeInner::Vector { .. }
        | TypeInner::Matrix { .. }
        | TypeInner::Atomic { .. } => true,
        TypeInner::Array { base, size, .. } => {
            size != crate::ArraySize::Dynamic && is_value_init_supported(module, base)
        }
        TypeInner::Struct { ref members, .. } => members
            .iter()
            .all(|member| is_value_init_supported(module, member.ty)),
        _ => false,
    }
}
