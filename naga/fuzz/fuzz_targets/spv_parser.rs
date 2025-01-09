#![cfg_attr(enable_fuzzing, no_main)]

#[cfg(enable_fuzzing)]
mod fuzz {
    use libfuzzer_sys::fuzz_target;
    use naga::front::spv::{Frontend, Options};

    fuzz_target!(|data: Vec<u32>| {
        // Ensure the parser can handle potentially malformed data without crashing.
        let options = Options::default();
        let _result = Frontend::new(data.into_iter(), &options).parse();
    });
}

#[cfg(not(enable_fuzzing))]
fn main() {}
