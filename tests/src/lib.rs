//! Test utilities for the wgpu repository.

#![allow(clippy::arc_with_non_send_sync)] // False positive on wasm

mod config;
mod expectations;
pub mod image;
mod init;
mod isolation;
pub mod native;
mod params;
mod poll;
mod report;
mod run;

#[cfg(target_arch = "wasm32")]
pub use init::initialize_html_canvas;

pub use self::image::ComparisonType;
pub use config::GpuTestConfiguration;
#[doc(hidden)]
pub use ctor;
pub use expectations::{FailureApplicationReasons, FailureBehavior, FailureCase, FailureReason};
pub use init::{initialize_adapter, initialize_device, initialize_instance};
pub use params::TestParameters;
pub use run::{execute_test, TestingContext};
pub use wgpu_macros::gpu_test;

/// Run some code in an error scope and assert that validation fails.
///
/// Note that errors related to commands for the GPU (i.e. raised by methods on
/// GPUCommandEncoder, GPURenderPassEncoder, GPUComputePassEncoder,
/// GPURenderBundleEncoder) are usually not raised immediately. They are raised
/// only when `finish()` is called on the command encoder. Tests of such error
/// cases should call `fail` with a closure that calls `finish()`, not with a
/// closure that encodes the actual command.
pub fn fail<T>(
    device: &wgpu::Device,
    callback: impl FnOnce() -> T,
    expected_msg_substring: Option<&'static str>,
) -> T {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = callback();
    let validation_error = pollster::block_on(device.pop_error_scope())
        .expect("expected validation error in callback, but no validation error was emitted");
    if let Some(expected_msg_substring) = expected_msg_substring {
        let lowered_expected = expected_msg_substring.to_lowercase();
        let lowered_actual = validation_error.to_string().to_lowercase();
        assert!(
            lowered_actual.contains(&lowered_expected),
            concat!(
                "expected validation error case-insensitively containing {}, ",
                "but it was not present in actual error message:\n{}"
            ),
            expected_msg_substring,
            validation_error
        );
    }

    result
}

/// Run some code in an error scope and assert that validation succeeds.
#[track_caller]
pub fn valid<T>(device: &wgpu::Device, callback: impl FnOnce() -> T) -> T {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = callback();
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        panic!(
            "`valid` block at {} encountered wgpu error:\n{error}",
            std::panic::Location::caller()
        );
    }

    result
}

/// Run some code in an error scope and assert that validation succeeds or fails depending on the
/// provided `should_fail` boolean.
pub fn fail_if<T>(
    device: &wgpu::Device,
    should_fail: bool,
    callback: impl FnOnce() -> T,
    expected_msg_substring: Option<&'static str>,
) -> T {
    if should_fail {
        fail(device, callback, expected_msg_substring)
    } else {
        valid(device, callback)
    }
}

fn did_fill_error_scope<T>(
    device: &wgpu::Device,
    callback: impl FnOnce() -> T,
    filter: wgpu::ErrorFilter,
) -> (bool, T) {
    device.push_error_scope(filter);
    let result = callback();
    let validation_error = pollster::block_on(device.pop_error_scope());
    let failed = validation_error.is_some();

    (failed, result)
}

/// Returns true if the provided callback fails validation.
pub fn did_fail<T>(device: &wgpu::Device, callback: impl FnOnce() -> T) -> (bool, T) {
    did_fill_error_scope(device, callback, wgpu::ErrorFilter::Validation)
}

/// Returns true if the provided callback encounters an out-of-memory error.
pub fn did_oom<T>(device: &wgpu::Device, callback: impl FnOnce() -> T) -> (bool, T) {
    did_fill_error_scope(device, callback, wgpu::ErrorFilter::OutOfMemory)
}

/// Adds the necessary main function for our gpu test harness.
#[macro_export]
macro_rules! gpu_test_main {
    () => {
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
        #[cfg(target_arch = "wasm32")]
        fn main() {}

        #[cfg(not(target_arch = "wasm32"))]
        fn main() -> $crate::native::MainResult {
            $crate::native::main()
        }
    };
}
