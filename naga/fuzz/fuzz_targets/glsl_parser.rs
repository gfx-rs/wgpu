#![cfg_attr(all(fuzzable_platform, fuzzing), no_main)]

#[cfg(all(fuzzable_platform, fuzzing))]
mod fuzz {
    use std::iter::FromIterator;

    use arbitrary::Arbitrary;
    use libfuzzer_sys::fuzz_target;
    use naga::{
        front::glsl::{Frontend, Options},
        FastHashMap, ShaderStage,
    };

    #[derive(Debug, Arbitrary)]
    struct OptionsProxy {
        pub stage: ShaderStage,
        pub defines: std::collections::HashMap<String, String>,
    }

    impl From<OptionsProxy> for Options {
        fn from(proxy: OptionsProxy) -> Self {
            Options {
                stage: proxy.stage,
                // NOTE: This is a workaround needed due to lack of rust-fuzz/arbitrary support for hashbrown.
                defines: FastHashMap::from_iter(
                    proxy
                        .defines
                        .keys()
                        .map(|k| (k.clone(), proxy.defines.get(&k.clone()).unwrap().clone())),
                ),
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

#[cfg(not(all(fuzzable_platform, fuzzing)))]
fn main() {}
