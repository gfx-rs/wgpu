use std::{future::Future, pin::Pin};

use arrayvec::ArrayVec;

use crate::{
    infra::{report::AdapterReport, GpuTestConfiguration},
    initialize_test,
};

pub(super) struct SingleTest {
    name: String,
    future: Pin<Box<dyn Future<Output = Result<(), libtest_mimic::Failed>> + Send + Sync>>,
}

impl SingleTest {
    pub fn from_gpu_test(
        test: GpuTestConfiguration,
        adapter: &AdapterReport,
        adapter_index: usize,
    ) -> Self {
        let base_name = test.name;
        let backend = &adapter.info.backend;
        let device_name = &adapter.info.name;

        // Figure out if we should skip the test and if so, why.
        let mut skipped_reasons: ArrayVec<_, 4> = ArrayVec::new();
        let missing_features = test.params.required_features - adapter.features;
        if !missing_features.is_empty() {
            skipped_reasons.push("Features");
        }

        if !test.params.required_limits.check_limits(&adapter.limits) {
            skipped_reasons.push("Limits");
        }

        let missing_downlevel_flags =
            test.params.required_downlevel_caps.flags - adapter.downlevel_caps.flags;
        if !missing_downlevel_flags.is_empty() {
            skipped_reasons.push("Downlevel Flags");
        }

        if test.params.required_downlevel_caps.shader_model > adapter.downlevel_caps.shader_model {
            skipped_reasons.push("Shader Model");
        }

        // Produce a lower-case version of the adapter info, for comparison against
        // `parameters.skips` and `parameters.failures`.
        let adapter_lowercase_info = wgt::AdapterInfo {
            name: adapter.info.name.to_lowercase(),
            driver: adapter.info.driver.to_lowercase(),
            ..adapter.info.clone()
        };

        // Check if we should skip the test altogether.
        let skip_reason = test
            .params
            .skips
            .iter()
            .find_map(|case| case.applies_to(&adapter_lowercase_info));

        let expected_failure = test
            .params
            .failures
            .iter()
            .find_map(|case| case.applies_to(&adapter_lowercase_info));

        let mut skip = false;
        let running_msg = if let Some(reasons) = skip_reason {
            skip = true;

            let names: ArrayVec<_, 4> = reasons
                .iter_names()
                .map(|(name, _)| name)
                .collect();
            let names_text = names.join(" | ");

            format!("Skipped Failure: {}", names_text)
        } else if !skipped_reasons.is_empty() {
            skip = true;
            format!("Skipped: {}", skipped_reasons.join(" | "))
        } else if let Some(failure_resasons) = expected_failure {
            let names: ArrayVec<_, 4> = failure_resasons
                .iter_names()
                .map(|(name, _)| name)
                .collect();
            let names_text = names.join(" | ");

            format!("Executed Failure: {}", names_text)
        } else {
            String::from("Executed")
        };

        let full_name =
            format!("[{running_msg}] [{backend:?}/{device_name}/{adapter_index}] {base_name}");

        Self {
            name: full_name,
            future: Box::pin(async move {
                if skip {
                    return Ok(());
                }
                initialize_test(
                    test.params,
                    expected_failure,
                    adapter_index,
                    test.test.expect("Test must be specified"),
                )
                .await;
                Ok(())
            }),
        }
    }

    pub fn into_trial(self) -> libtest_mimic::Trial {
        libtest_mimic::Trial::test(self.name, || pollster::block_on(self.future))
    }
}
