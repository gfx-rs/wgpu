extern crate alloc;

extern crate wgpu_shader_types as wst;
extern crate wgpu_types as wgt;

#[cfg(feature = "glsl-out")]
pub mod glsl;
#[cfg(feature = "msl-out")]
pub mod msl;
#[cfg(feature = "spv-out")]
pub mod spv;

pub use wgpu_shader_types::ShaderStage;

use alloc::borrow::Cow;
use core::fmt;

#[derive(Debug, Clone)]
pub struct DebugSource {
    pub file_name: Cow<'static, str>,
    pub source_code: Cow<'static, str>,
}

/// Naga shader module.
#[derive(Default)]
pub struct NagaShader {
    /// Shader module IR.
    #[cfg(feature = "naga-dep")]
    pub(crate) module: Cow<'static, naga::Module>,
    /// Analysis information of the module.
    #[cfg(feature = "naga-dep")]
    pub(crate) info: naga::valid::ModuleInfo,
    /// Source codes for debug
    #[cfg(feature = "naga-dep")]
    pub(crate) debug_source: Option<DebugSource>,
    /// Private field to keep this non-constructible
    #[cfg(not(feature = "naga-dep"))]
    _p: (),
}

impl NagaShader {
    pub fn has_overrides(&self) -> bool {
        #[cfg(feature = "naga-dep")]
        {
            !self.module.overrides.is_empty()
        }
        #[cfg(not(feature = "naga-dep"))]
        {
            unreachable!()
        }
    }
}

// Custom implementation avoids the need to generate Debug impl code
// for the whole Naga module and info.
impl fmt::Debug for NagaShader {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Naga shader")
    }
}

/// Shader input.
#[allow(clippy::large_enum_variant)]
pub enum ShaderInput<'a> {
    Naga(NagaShader),
    MetalLib {
        file: &'a [u8],
        num_workgroups: (u32, u32, u32),
    },
    Msl {
        shader: &'a str,
        num_workgroups: (u32, u32, u32),
    },
    SpirV(&'a [u32]),
    Dxil {
        shader: &'a [u8],
        num_workgroups: (u32, u32, u32),
    },
    Hlsl {
        shader: &'a str,
        num_workgroups: (u32, u32, u32),
    },
    Glsl {
        shader: &'a str,
        num_workgroups: (u32, u32, u32),
    },
}

pub enum ShaderCompilationError {
    EntryPoint,
    PipelineConstants(String),
    Linkage(String),
}
