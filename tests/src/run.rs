use std::panic::AssertUnwindSafe;

use futures_lite::FutureExt;
use wgpu::{Adapter, Device, Instance, Queue};

use crate::{
    expectations::{expectations_match_failures, ExpectationMatchResult, FailureResult},
    init::{init_logger, initialize_adapter, initialize_device},
    isolation,
    params::TestInfo,
    report::AdapterReport,
    GpuTestConfiguration,
};

/// Parameters and resources hadned to the test function.
pub struct TestingContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub adapter_info: wgpu::AdapterInfo,
    pub adapter_downlevel_capabilities: wgpu::DownlevelCapabilities,
    pub device: Device,
    pub device_features: wgpu::Features,
    pub device_limits: wgpu::Limits,
    pub queue: Queue,
}

/// Execute the given test configuration with the given adapter report.
///
/// If test_info is specified, will use the information whether to skip the test.
/// If it is not, we'll create the test info from the adapter itself.
pub async fn execute_test(
    adapter_report: Option<&AdapterReport>,
    config: GpuTestConfiguration,
    test_info: Option<TestInfo>,
) {
    // If we get information externally, skip based on that information before we do anything.
    if let Some(TestInfo { skip: true, .. }) = test_info {
        return;
    }

    init_logger();

    let _test_guard = isolation::OneTestPerProcessGuard::new();

    let (instance, adapter, _surface_guard) =
        initialize_adapter(adapter_report, config.params.force_fxc).await;

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

    // Print the name of the test.
    log::info!("TEST: {}", config.name);

    let (device, queue) = pollster::block_on(initialize_device(
        &adapter,
        config.params.required_features,
        config.params.required_limits.clone(),
    ));

    let context = TestingContext {
        instance,
        adapter,
        adapter_info,
        adapter_downlevel_capabilities,
        device,
        device_features: config.params.required_features,
        device_limits: config.params.required_limits.clone(),
        queue,
    };

    let mut failures = Vec::new();

    // Run the test, and catch panics (possibly due to failed assertions).
    let panic_res = AssertUnwindSafe((config.test.as_ref().unwrap())(context))
        .catch_unwind()
        .await;

    if let Err(panic) = panic_res {
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str));

        let result = FailureResult::panic();

        let result = if let Some(panic_str) = message {
            result.with_message(panic_str)
        } else {
            result
        };

        failures.push(result)
    }

    // Check whether any validation errors were reported during the test run.
    cfg_if::cfg_if!(
        if #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))] {
            failures.extend(wgpu::hal::VALIDATION_CANARY.get_and_reset().into_iter().map(|msg| FailureResult::validation_error().with_message(msg)));
        } else if #[cfg(all(target_arch = "wasm32", feature = "webgl"))] {
            if _surface_guard.unwrap().check_for_unreported_errors() {
                failures.push(FailureResult::validation_error());
            }
        } else {
        }
    );

    // The call to matches_failure will log.
    if expectations_match_failures(&test_info.failures, failures) == ExpectationMatchResult::Panic {
        panic!(
            "{}: test {:?} did not behave as expected",
            config.location, config.name
        );
    }
    // Print the name of the test.
    log::info!("TEST FINISHED: {}", config.name);
}
