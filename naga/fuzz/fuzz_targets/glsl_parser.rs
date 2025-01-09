#![cfg_attr(enable_fuzzing, no_main)]

#[cfg(enable_fuzzing)]
mod fuzz {
    use arbitrary::Arbitrary;
    use libfuzzer_sys::fuzz_target;
    use naga::{
        front::glsl::{Frontend, Options},
        FastHashMap, ShaderStage,
    };

    #[derive(Debug, Arbitrary)]
    enum ShaderStageProxy {
        Vertex,
        Fragment,
        Compute,
    }

    impl From<ShaderStageProxy> for ShaderStage {
        fn from(proxy: ShaderStageProxy) -> Self {
            match proxy {
                ShaderStageProxy::Vertex => ShaderStage::Vertex,
                ShaderStageProxy::Fragment => ShaderStage::Fragment,
                ShaderStageProxy::Compute => ShaderStage::Compute,
            }
        }
    }

    #[derive(Debug, Arbitrary)]
    struct OptionsProxy {
        pub stage: ShaderStageProxy,
        pub defines: FastHashMap<String, String>,
    }

    impl From<OptionsProxy> for Options {
        fn from(proxy: OptionsProxy) -> Self {
            Options {
                stage: proxy.stage.into(),
                defines: proxy.defines,
            }
        }
    }

    fuzz_target!(|data: (OptionsProxy, String)| {
        let (options, source) = data;
        // Ensure the parser can handle potentially malformed strings without crashing.
        let mut parser = Frontend::default();
        let _result = parser.parse(&options.into(), &source);
    });
}

#[cfg(not(enable_fuzzing))]
fn main() {}
