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
        let params = test.params;

        let base_name = test.name;
        let backend = &adapter.info.backend;
        let device_name = &adapter.info.name;

        // Figure out if we should skip the test and if so, why.
        let mut skipped_reasons: ArrayVec<_, 4> = ArrayVec::new();
        let missing_features = params.required_features - adapter.features;
        if !missing_features.is_empty() {
            skipped_reasons.push("Features");
        }

        if !params.required_limits.check_limits(&adapter.limits) {
            skipped_reasons.push("Limits");
        }

        let missing_downlevel_flags =
            params.required_downlevel_caps.flags - adapter.downlevel_caps.flags;
        if !missing_downlevel_flags.is_empty() {
            skipped_reasons.push("Downlevel Flags");
        }

        if params.required_downlevel_caps.shader_model > adapter.downlevel_caps.shader_model {
            skipped_reasons.push("Shader Model");
        }

        let expected_failure = params.to_failure_reasons(&adapter.info);

        let mut should_skip = false;
        let running_msg = if !skipped_reasons.is_empty() {
            should_skip = true;
            format!("Skipped: {}", skipped_reasons.join(" | "))
        } else if let Some((failure_resasons, skip)) = expected_failure {
            should_skip |= skip;
            let names: ArrayVec<_, 4> = failure_resasons
                .iter_names()
                .map(|(name, _)| name)
                .collect();
            let names_text = names.join(" | ");

            let skip_text = if skip { "Skipped " } else { "Executed " };
            format!("{skip_text}Failure: {}", names_text)
        } else {
            String::from("Executed")
        };

        let full_name =
            format!("[{running_msg}] [{backend:?}/{device_name}/{adapter_index}] {base_name}");

        Self {
            name: full_name,
            future: Box::pin(async move {
                if should_skip {
                    return Ok(());
                }
                initialize_test(
                    params,
                    expected_failure.map(|(reasons, _)| reasons),
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
