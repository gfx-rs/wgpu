use arrayvec::ArrayVec;
use wgt::{DownlevelCapabilities, DownlevelFlags, Features, Limits};

use crate::{
    report::AdapterReport, FailureApplicationReasons, FailureBehavior, FailureCase,
    GpuTestConfiguration,
};

const LOWEST_DOWNLEVEL_PROPERTIES: wgpu::DownlevelCapabilities = DownlevelCapabilities {
    flags: wgt::DownlevelFlags::empty(),
    limits: wgt::DownlevelLimits {},
    shader_model: wgt::ShaderModel::Sm2,
};

/// This information determines if a test should run.
#[derive(Clone)]
pub struct TestParameters {
    pub required_features: Features,
    pub required_downlevel_caps: DownlevelCapabilities,
    pub required_limits: Limits,

    /// On Dx12, specifically test against the Fxc compiler.
    ///
    /// For testing workarounds to Fxc bugs.
    pub force_fxc: bool,

    /// Conditions under which this test should be skipped.
    pub skips: Vec<FailureCase>,

    /// Conditions under which this test should be run, but is expected to fail.
    pub failures: Vec<FailureCase>,
}

impl Default for TestParameters {
    fn default() -> Self {
        Self {
            required_features: Features::empty(),
            required_downlevel_caps: LOWEST_DOWNLEVEL_PROPERTIES,
            required_limits: Limits::downlevel_webgl2_defaults(),
            force_fxc: false,
            skips: Vec::new(),
            failures: Vec::new(),
        }
    }
}

// Builder pattern to make it easier
impl TestParameters {
    /// Set of common features that most internal tests require for compute and readback.
    pub fn test_features_limits(self) -> Self {
        self.downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults())
    }

    /// Set the list of features this test requires.
    pub fn features(mut self, features: Features) -> Self {
        self.required_features |= features;
        self
    }

    pub fn downlevel_flags(mut self, downlevel_flags: DownlevelFlags) -> Self {
        self.required_downlevel_caps.flags |= downlevel_flags;
        self
    }

    /// Set the limits needed for the test.
    pub fn limits(mut self, limits: Limits) -> Self {
        self.required_limits = limits;
        self
    }

    pub fn force_fxc(mut self, force_fxc: bool) -> Self {
        self.force_fxc = force_fxc;
        self
    }

    /// Mark the test as always failing, but not to be skipped.
    pub fn expect_fail(mut self, when: FailureCase) -> Self {
        self.failures.push(when);
        self
    }

    /// Mark the test as always failing, and needing to be skipped.
    pub fn skip(mut self, when: FailureCase) -> Self {
        self.skips.push(when);
        self
    }
}

/// Information about a test, including if if it should be skipped.
pub struct TestInfo {
    pub skip: bool,
    pub failure_application_reasons: FailureApplicationReasons,
    pub failures: Vec<FailureCase>,
    pub running_msg: String,
}

impl TestInfo {
    pub(crate) fn from_configuration(test: &GpuTestConfiguration, adapter: &AdapterReport) -> Self {
        // Figure out if a test is unsupported, and why.
        let mut unsupported_reasons: ArrayVec<_, 4> = ArrayVec::new();
        let missing_features = test.params.required_features - adapter.features;
        if !missing_features.is_empty() {
            unsupported_reasons.push("Features");
        }

        if !test.params.required_limits.check_limits(&adapter.limits) {
            unsupported_reasons.push("Limits");
        }

        let missing_downlevel_flags =
            test.params.required_downlevel_caps.flags - adapter.downlevel_caps.flags;
        if !missing_downlevel_flags.is_empty() {
            unsupported_reasons.push("Downlevel Flags");
        }

        if test.params.required_downlevel_caps.shader_model > adapter.downlevel_caps.shader_model {
            unsupported_reasons.push("Shader Model");
        }

        // Produce a lower-case version of the adapter info, for comparison against
        // `parameters.skips` and `parameters.failures`.
        let adapter_lowercase_info = wgt::AdapterInfo {
            name: adapter.info.name.to_lowercase(),
            driver: adapter.info.driver.to_lowercase(),
            ..adapter.info.clone()
        };

        // Check if we should skip the test altogether.
        let skip_application_reason = test
            .params
            .skips
            .iter()
            .find_map(|case| case.applies_to_adapter(&adapter_lowercase_info));

        let mut applicable_cases = Vec::with_capacity(test.params.failures.len());
        let mut failure_application_reasons = FailureApplicationReasons::empty();
        let mut flaky = false;
        for failure in &test.params.failures {
            if let Some(reasons) = failure.applies_to_adapter(&adapter_lowercase_info) {
                failure_application_reasons.insert(reasons);
                applicable_cases.push(failure.clone());
                flaky |= matches!(failure.behavior, FailureBehavior::Ignore);
            }
        }

        let mut skip = false;
        let running_msg = if let Some(reasons) = skip_application_reason {
            skip = true;

            let names: ArrayVec<_, 4> = reasons.iter_names().map(|(name, _)| name).collect();
            let names_text = names.join(" | ");

            format!("Skipped Failure: {names_text}")
        } else if !unsupported_reasons.is_empty() {
            skip = true;
            format!("Unsupported: {}", unsupported_reasons.join(" | "))
        } else if !failure_application_reasons.is_empty() {
            if cfg!(target_arch = "wasm32") {
                skip = true;
            }

            let names: ArrayVec<_, 4> = failure_application_reasons
                .iter_names()
                .map(|(name, _)| name)
                .collect();
            let names_text = names.join(" & ");
            let flaky_text = if flaky { " Flaky " } else { " " };

            format!("Executed{flaky_text}Failure: {names_text}")
        } else {
            String::from("Executed")
        };

        Self {
            skip,
            failure_application_reasons,
            failures: applicable_cases,
            running_msg,
        }
    }
}
