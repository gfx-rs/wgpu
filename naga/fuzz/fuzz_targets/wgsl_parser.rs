#![cfg_attr(enable_fuzzing, no_main)]

#[cfg(enable_fuzzing)]
mod fuzz {
    use libfuzzer_sys::fuzz_target;
    use naga::front::wgsl::Frontend;

    fuzz_target!(|data: String| {
        // Ensure the parser can handle potentially malformed strings without crashing.
        let _result = Frontend::new().parse(&data);
    });
}

#[cfg(not(enable_fuzzing))]
fn main() {}
