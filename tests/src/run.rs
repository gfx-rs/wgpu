use std::panic::AssertUnwindSafe;

use futures_lite::FutureExt;
use wgpu::{Adapter, Device, Queue};

use crate::{
    init::{initialize_adapter, initialize_device},
    isolation,
    params::TestInfo,
    report::AdapterReport,
    GpuTestConfiguration,
};

pub struct TestingContext {
    pub adapter: Adapter,
    pub adapter_info: wgt::AdapterInfo,
    pub adapter_downlevel_capabilities: wgt::DownlevelCapabilities,
    pub device: Device,
    pub device_features: wgt::Features,
    pub device_limits: wgt::Limits,
    pub queue: Queue,
}

pub async fn execute_test(
    config: GpuTestConfiguration,
    test_info: Option<TestInfo>,
    adapter_index: usize,
) {
    // If we get information externally, skip based on that information before we do anything.
    if let Some(TestInfo { skip: true, .. }) = test_info {
        return;
    }

    // We don't actually care if it fails
    #[cfg(not(target_arch = "wasm32"))]
    let _ = env_logger::try_init();
    #[cfg(target_arch = "wasm32")]
    let _ = console_log::init_with_level(log::Level::Info);

    let _test_guard = isolation::OneTestPerProcessGuard::new();

    let (adapter, _surface_guard) = initialize_adapter(adapter_index).await;

    let adapter_info = adapter.get_info();
    let adapter_downlevel_capabilities = adapter.get_downlevel_capabilities();

    let test_info = test_info.unwrap_or_else(|| {
        let adapter_report = AdapterReport::from_adapter(&adapter);
        TestInfo::from_configuration(&config, &adapter_report)
    });

    // We are now guaranteed to have information about this test, so skip if we need to.
    if test_info.skip {
        log::info!("TEST RESULT: SKIPPED");
        return;
    }

    let (device, queue) = pollster::block_on(initialize_device(
        &adapter,
        config.params.required_features,
        config.params.required_limits.clone(),
    ));

    let context = TestingContext {
        adapter,
        adapter_info,
        adapter_downlevel_capabilities,
        device,
        device_features: config.params.required_features,
        device_limits: config.params.required_limits.clone(),
        queue,
    };

    // Run the test, and catch panics (possibly due to failed assertions).
    let panicked = AssertUnwindSafe((config.test.as_ref().unwrap())(context))
        .catch_unwind()
        .await
        .is_err();

    // Check whether any validation errors were reported during the test run.
    cfg_if::cfg_if!(
        if #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))] {
            let canary_set = wgpu::hal::VALIDATION_CANARY.get_and_reset();
        } else if #[cfg(all(target_arch = "wasm32", feature = "webgl"))] {
            let canary_set = _surface_guard.unwrap().check_for_unreported_errors();
        } else {
            // TODO: WebGPU
            let canary_set = false;
        }
    );

    // Summarize reasons for actual failure, if any.
    let failure_cause = match (panicked, canary_set) {
        (true, true) => Some("PANIC AND VALIDATION ERROR"),
        (true, false) => Some("PANIC"),
        (false, true) => Some("VALIDATION ERROR"),
        (false, false) => None,
    };

    // Compare actual results against expectations.
    match (failure_cause, test_info.expected_failure_reason) {
        // The test passed, as expected.
        (None, None) => log::info!("TEST RESULT: PASSED"),
        // The test failed unexpectedly.
        (Some(cause), None) => {
            panic!("UNEXPECTED TEST FAILURE DUE TO {cause}")
        }
        // The test passed unexpectedly.
        (None, Some(reason)) => {
            panic!("UNEXPECTED TEST PASS: {reason:?}");
        }
        // The test failed, as expected.
        (Some(cause), Some(reason_expected)) => {
            log::info!(
                "TEST RESULT: EXPECTED FAILURE DUE TO {} (expected because of {:?})",
                cause,
                reason_expected
            );
        }
    }
}
