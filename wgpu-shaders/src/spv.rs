use alloc::borrow::Cow;
use naga::back::spv::Options;

pub struct SpvCompileOptionsDesc {
    pub lang_version: (u8, u8),
}

#[derive(Clone, Debug)]
pub struct SpvCompileOptions {
    #[cfg(feature = "naga-dep")]
    options: Options<'static>,
    // TODO
}
impl SpvCompileOptions {
    pub fn new(desc: SpvCompileOptionsDesc) -> Self {
        Self {
            #[cfg(feature = "naga-dep")]
            options: Options {
                lang_version: desc.lang_version,
                // TODO
                ..Default::default()
            },
        }
    }
}

pub struct SpvShaderDesc<'a> {
    pub options: &'a SpvCompileOptions,
    pub runtime_checks: wgt::ShaderRuntimeChecks,
    pub entry_point: Option<(String, wst::ShaderStage)>,
}

impl super::NagaShader {
    pub fn compile_spv(&self, desc: SpvShaderDesc) -> Vec<u32> {
        todo!()
    }
}
