use std::{io::Write, panic::AssertUnwindSafe};

use futures_lite::FutureExt;
use wgpu::{Adapter, Device, Instance, Queue};
use wgpu_test_metadata::{
    AdapterKey, EventPhase, GpuHarnessEvent, GpuTestKey, GPU_TEST_EVENT_PREFIX,
};

use crate::{
    expectations::{
        expectations_match_failures, expected_failure_signatures, ExpectationMatchResult,
        FailureResult,
    },
    init::{init_logger, initialize_adapter, initialize_device},
    isolation,
    params::TestInfo,
    report::AdapterReport,
    GpuTestConfiguration,
};

fn adapter_key(info: &wgpu::AdapterInfo) -> AdapterKey {
    AdapterKey {
        backend: format!("{:?}", info.backend),
        vendor: info.vendor,
        device: info.device,
        name: info.name.clone(),
        driver: info.driver.clone(),
    }
}

fn test_key(test_path: &str, info: &wgpu::AdapterInfo) -> GpuTestKey {
    GpuTestKey {
        test_path: test_path.to_owned(),
        adapter: adapter_key(info),
    }
}

fn emit_gpu_test_event(event: &GpuHarnessEvent) {
    if let Ok(json) = serde_json::to_string(event) {
        println!("{GPU_TEST_EVENT_PREFIX}{json}");
        let _ = std::io::stdout().flush();
    }
}

#[derive(Hash)]
/// Parameters and resources handed to the test function.
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
    let mut test_info = test_info;
    // If we get information externally and know we should skip, avoid adapter/device setup.
    if let (Some(test_info), Some(adapter_report)) = (test_info.as_ref(), adapter_report) {
        if test_info.skip {
            let expected_failure_signatures = expected_failure_signatures(&test_info.failures);
            let key = test_key(&config.name, &adapter_report.info);

            emit_gpu_test_event(&GpuHarnessEvent {
                version: 1,
                phase: EventPhase::Before,
                key: key.clone(),
                inline_expect_fail: test_info.inline_expect_fail,
                inline_expect_crash: test_info.inline_expect_crash,
                skip: test_info.skip,
                skip_due_to_unsupported: test_info.skip_due_to_unsupported,
                skip_due_to_expectation: test_info.skip_due_to_expectation,
                expected_failure_signatures,
                actual_success: None,
                actual_failure_signatures: Vec::new(),
                expectation_verdict: None,
            });

            emit_gpu_test_event(&GpuHarnessEvent {
                version: 1,
                phase: EventPhase::After,
                key,
                inline_expect_fail: test_info.inline_expect_fail,
                inline_expect_crash: test_info.inline_expect_crash,
                skip: test_info.skip,
                skip_due_to_unsupported: test_info.skip_due_to_unsupported,
                skip_due_to_expectation: test_info.skip_due_to_expectation,
                expected_failure_signatures: Vec::new(),
                actual_success: None,
                actual_failure_signatures: Vec::new(),
                expectation_verdict: Some("skipped".to_owned()),
            });
            return;
        }
    }

    init_logger();

    let _test_guard = isolation::OneTestPerProcessGuard::new();

    let (instance, adapter, _surface_guard) =
        initialize_adapter(adapter_report, &config.params).await;

    let adapter_info = adapter.get_info();
    let adapter_downlevel_capabilities = adapter.get_downlevel_capabilities();

    let test_info = test_info.take().unwrap_or_else(|| {
        let adapter_report = AdapterReport::from_adapter(&adapter);
        TestInfo::from_configuration(&config, &adapter_report)
    });

    let event_key = test_key(&config.name, &adapter_info);
    let expected_failure_signatures = expected_failure_signatures(&test_info.failures);
    emit_gpu_test_event(&GpuHarnessEvent {
        version: 1,
        phase: EventPhase::Before,
        key: event_key.clone(),
        inline_expect_fail: test_info.inline_expect_fail,
        inline_expect_crash: test_info.inline_expect_crash,
        skip: test_info.skip,
        skip_due_to_unsupported: test_info.skip_due_to_unsupported,
        skip_due_to_expectation: test_info.skip_due_to_expectation,
        expected_failure_signatures,
        actual_success: None,
        actual_failure_signatures: Vec::new(),
        expectation_verdict: None,
    });

    // We are now guaranteed to have information about this test, so skip if we need to.
    if test_info.skip {
        emit_gpu_test_event(&GpuHarnessEvent {
            version: 1,
            phase: EventPhase::After,
            key: event_key,
            inline_expect_fail: test_info.inline_expect_fail,
            inline_expect_crash: test_info.inline_expect_crash,
            skip: test_info.skip,
            skip_due_to_unsupported: test_info.skip_due_to_unsupported,
            skip_due_to_expectation: test_info.skip_due_to_expectation,
            expected_failure_signatures: Vec::new(),
            actual_success: None,
            actual_failure_signatures: Vec::new(),
            expectation_verdict: Some("skipped".to_owned()),
        });
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

    let actual_success = failures.is_empty();
    let actual_failure_signatures = failures
        .iter()
        .map(FailureResult::to_signature)
        .collect::<Vec<_>>();

    let expectation_match = expectations_match_failures(&test_info.failures, failures);
    let expectation_verdict = match expectation_match {
        ExpectationMatchResult::Panic => "panic",
        ExpectationMatchResult::Complete => "complete",
    };

    emit_gpu_test_event(&GpuHarnessEvent {
        version: 1,
        phase: EventPhase::After,
        key: event_key,
        inline_expect_fail: test_info.inline_expect_fail,
        inline_expect_crash: test_info.inline_expect_crash,
        skip: false,
        skip_due_to_unsupported: false,
        skip_due_to_expectation: false,
        expected_failure_signatures: Vec::new(),
        actual_success: Some(actual_success),
        actual_failure_signatures,
        expectation_verdict: Some(expectation_verdict.to_owned()),
    });

    // The call to matches_failure will log.
    if expectation_match == ExpectationMatchResult::Panic {
        panic!(
            "{}: test {:?} did not behave as expected",
            config.location, config.name
        );
    }
    // Print the name of the test.
    log::info!("TEST FINISHED: {}", config.name);
}
