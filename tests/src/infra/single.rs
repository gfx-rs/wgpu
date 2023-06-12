use std::sync::Arc;

use arrayvec::ArrayVec;

use crate::{
    infra::{report::AdapterReport, GpuTest},
    initialize_test, TestParameters,
};

pub fn run_test(
    test: Arc<dyn GpuTest + Send + Sync>,
    adapter: &AdapterReport,
) -> libtest_mimic::Trial {
    let params = TestParameters::default();
    let params = test.parameters(params);

    let base_name = test.name();
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

    let full_name = format!("[{backend:?}/{device_name}] [{running_msg}] {base_name}");

    libtest_mimic::Trial::test(full_name, move || {
        if should_skip {
            return Ok(());
        }
        initialize_test(
            params,
            expected_failure.map(|(reasons, _)| reasons),
            |ctx| test.run(ctx),
        );
        Ok(())
    })
}
