#![cfg_attr(all(fuzzable_platform, fuzzing), no_main)]

#[cfg(all(fuzzable_platform, fuzzing))]
mod fuzz {
    use libfuzzer_sys::fuzz_target;
    use naga::front::spv::{Frontend, Options};

    fuzz_target!(|data: Vec<u32>| {
        // Ensure the parser can handle potentially malformed data without crashing.
        let options = Options::default();
        let _result = Frontend::new(data.into_iter(), &options).parse();
    });
}

#[cfg(not(all(fuzzable_platform, fuzzing)))]
fn main() {}
