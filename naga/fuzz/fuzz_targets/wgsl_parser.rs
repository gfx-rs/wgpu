#![no_main]
use libfuzzer_sys::fuzz_target;
use naga::front::wgsl::Frontend;

fuzz_target!(|data: String| {
    // Ensure the parser can handle potentially malformed strings without crashing.
    let _result = Frontend::new().parse(&data);
});
