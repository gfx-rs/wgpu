#![no_main]
#[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
mod fuzz {
    use libfuzzer_sys::fuzz_target;
    use naga::front::wgsl::Frontend;

    fuzz_target!(|data: String| {
        // Ensure the parser can handle potentially malformed strings without crashing.
        let _result = Frontend::new().parse(&data);
    });
}
