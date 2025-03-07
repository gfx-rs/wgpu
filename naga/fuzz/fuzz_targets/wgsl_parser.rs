#![cfg_attr(all(fuzzable_platform, fuzzing), no_main)]

#[cfg(all(fuzzable_platform, fuzzing))]
mod fuzz {
    use libfuzzer_sys::fuzz_target;
    use naga::front::wgsl::Frontend;

    fuzz_target!(|data: String| {
        // Ensure the parser can handle potentially malformed strings without crashing.
        let _result = Frontend::new().parse(&data);
    });
}

#[cfg(not(all(fuzzable_platform, fuzzing)))]
fn main() {}
