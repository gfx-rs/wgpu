/// Conditions under which a test should fail or be skipped.
///
/// By passing a `FailureCase` to [`TestParameters::expect_fail`][expect_fail], you can
/// mark a test as expected to fail under the indicated conditions. By
/// passing it to [`TestParameters::skip`][skip], you can request that the
/// test be skipped altogether.
///
/// If a field is `None`, then that field does not restrict matches. For
/// example:
///
/// ```
/// # use wgpu_test::*;
/// FailureCase {
///     backends: Some(wgpu::Backends::DX11 | wgpu::Backends::DX12),
///     vendor: None,
///     adapter: Some("RTX"),
///     driver: None,
///     reason: FailureReason::ValidationError(Some("Some error substring")),
///     behavior: FailureBehavior::AssertFailure,
/// }
/// # ;
/// ```
///
/// This applies to all cards with `"RTX'` in their name on either
/// Direct3D backend, no matter the vendor ID or driver name.
///
/// The strings given here need only appear as a substring in the
/// corresponding [`AdapterInfo`] fields. The comparison is
/// case-insensitive.
///
/// The default value of `FailureCase` applies to any test case. That
/// is, there are no criteria to constrain the match.
///
/// [skip]: super::TestParameters::skip
/// [expect_fail]: super::TestParameters::expect_fail
/// [`AdapterInfo`]: wgt::AdapterInfo
#[derive(Default, Clone)]
pub struct FailureCase {
    /// Backends expected to fail, or `None` for any backend.
    ///
    /// If this is `None`, or if the test is using one of the backends
    /// in `backends`, then this `FailureCase` applies.
    pub backends: Option<wgpu::Backends>,

    /// Vendor expected to fail, or `None` for any vendor.
    ///
    /// If `Some`, this must match [`AdapterInfo::device`], which is
    /// usually the PCI device id. Otherwise, this `FailureCase`
    /// applies regardless of vendor.
    ///
    /// [`AdapterInfo::device`]: wgt::AdapterInfo::device
    pub vendor: Option<u32>,

    /// Name of adaper expected to fail, or `None` for any adapter name.
    ///
    /// If this is `Some(s)` and `s` is a substring of
    /// [`AdapterInfo::name`], then this `FailureCase` applies. If
    /// this is `None`, the adapter name isn't considered.
    ///
    /// [`AdapterInfo::name`]: wgt::AdapterInfo::name
    pub adapter: Option<&'static str>,

    /// Name of driver expected to fail, or `None` for any driver name.
    ///
    /// If this is `Some(s)` and `s` is a substring of
    /// [`AdapterInfo::driver`], then this `FailureCase` applies. If
    /// this is `None`, the driver name isn't considered.
    ///
    /// [`AdapterInfo::driver`]: wgt::AdapterInfo::driver
    pub driver: Option<&'static str>,

    /// Reason why the test is expected to fail.
    ///
    /// If this does not match, the failure will not match this case.
    ///
    /// If no reasons are pushed, will match any failure.
    pub reasons: Vec<FailureReason>,

    /// Behavior after this case matches a failure.
    pub behavior: FailureBehavior,
}

impl FailureCase {
    /// This case applies to all tests.
    pub fn always() -> Self {
        FailureCase::default()
    }

    /// This case applies to no tests.
    pub fn never() -> Self {
        FailureCase {
            backends: Some(wgpu::Backends::empty()),
            ..FailureCase::default()
        }
    }

    /// Tests running on any of the given backends.
    pub fn backend(backends: wgpu::Backends) -> Self {
        FailureCase {
            backends: Some(backends),
            ..FailureCase::default()
        }
    }

    /// Tests running on `adapter`.
    ///
    /// For this case to apply, the `adapter` string must appear as a substring
    /// of the adapter's [`AdapterInfo::name`]. The comparison is
    /// case-insensitive.
    ///
    /// [`AdapterInfo::name`]: wgt::AdapterInfo::name
    pub fn adapter(adapter: &'static str) -> Self {
        FailureCase {
            adapter: Some(adapter),
            ..FailureCase::default()
        }
    }

    /// Tests running on `backend` and `adapter`.
    ///
    /// For this case to apply, the test must be using an adapter for one of the
    /// given `backend` bits, and `adapter` string must appear as a substring of
    /// the adapter's [`AdapterInfo::name`]. The string comparison is
    /// case-insensitive.
    ///
    /// [`AdapterInfo::name`]: wgt::AdapterInfo::name
    pub fn backend_adapter(backends: wgpu::Backends, adapter: &'static str) -> Self {
        FailureCase {
            backends: Some(backends),
            adapter: Some(adapter),
            ..FailureCase::default()
        }
    }

    /// Tests running under WebGL.
    pub fn webgl2() -> Self {
        #[cfg(target_arch = "wasm32")]
        let case = FailureCase::backend(wgpu::Backends::GL);
        #[cfg(not(target_arch = "wasm32"))]
        let case = FailureCase::never();
        case
    }

    /// Tests running on the MoltenVK Vulkan driver on macOS.
    pub fn molten_vk() -> Self {
        FailureCase {
            backends: Some(wgpu::Backends::VULKAN),
            driver: Some("MoltenVK"),
            ..FailureCase::default()
        }
    }

    /// Return the reasons why this case should fail.
    pub fn reasons(&self) -> &[FailureReason] {
        if self.reasons.is_empty() {
            std::array::from_ref(&FailureReason::Any)
        } else {
            &self.reasons
        }
    }

    /// Matches this failure case against the given validation error substring.
    ///
    /// Substrings are matched case-insensitively.
    ///
    /// If multiple reasons are pushed, will match any of them.
    pub fn validation_error(mut self, msg: &'static str) -> Self {
        self.reasons.push(FailureReason::ValidationError(Some(msg)));
        self
    }

    /// Matches this failure case against the given panic substring.
    ///
    /// Substrings are matched case-insensitively.
    ///
    /// If multiple reasons are pushed, will match any of them.
    pub fn panic(mut self, msg: &'static str) -> Self {
        self.reasons.push(FailureReason::Panic(Some(msg)));
        self
    }

    /// Test is flaky with the given configuration. Do not assert failure.
    ///
    /// Use this _very_ sparyingly, and match as tightly as you can, including giving a specific failure message.
    pub fn flaky(self) -> Self {
        FailureCase {
            behavior: FailureBehavior::Ignore,
            ..self
        }
    }

    /// Test whether `self` applies to `info`.
    ///
    /// If it does, return a `FailureReasons` whose set bits indicate
    /// why. If it doesn't, return `None`.
    ///
    /// The caller is responsible for converting the string-valued
    /// fields of `info` to lower case, to ensure case-insensitive
    /// matching.
    pub(crate) fn applies_to_adapter(
        &self,
        info: &wgt::AdapterInfo,
    ) -> Option<FailureApplicationReasons> {
        let mut reasons = FailureApplicationReasons::empty();

        if let Some(backends) = self.backends {
            if !backends.contains(wgpu::Backends::from(info.backend)) {
                return None;
            }
            reasons.set(FailureApplicationReasons::BACKEND, true);
        }
        if let Some(vendor) = self.vendor {
            if vendor != info.vendor {
                return None;
            }
            reasons.set(FailureApplicationReasons::VENDOR, true);
        }
        if let Some(adapter) = self.adapter {
            let adapter = adapter.to_lowercase();
            if !info.name.contains(&adapter) {
                return None;
            }
            reasons.set(FailureApplicationReasons::ADAPTER, true);
        }
        if let Some(driver) = self.driver {
            let driver = driver.to_lowercase();
            if !info.driver.contains(&driver) {
                return None;
            }
            reasons.set(FailureApplicationReasons::DRIVER, true);
        }

        // If we got this far but no specific reasons were triggered, then this
        // must be a wildcard.
        if reasons.is_empty() {
            Some(FailureApplicationReasons::ALWAYS)
        } else {
            Some(reasons)
        }
    }

    pub(crate) fn matches_failure(&self, failure: &FailureResult) -> bool {
        for reason in self.reasons() {
            let result = match (reason, failure) {
                (FailureReason::Any, _) => {
                    log::error!("Matched failure case: Wildcard");
                    true
                }
                (FailureReason::ValidationError(None), FailureResult::ValidationError(_)) => {
                    log::error!("Matched failure case: Any Validation Error");
                    true
                }
                (
                    FailureReason::ValidationError(Some(expected)),
                    FailureResult::ValidationError(Some(actual)),
                ) => {
                    let result = actual.to_lowercase().contains(&expected.to_lowercase());
                    if result {
                        log::error!(
                            "Matched failure case: Validation Error containing \"{}\"",
                            expected
                        );
                    }
                    result
                }
                (FailureReason::Panic(None), FailureResult::Panic(_)) => {
                    log::error!("Matched failure case: Any Panic");
                    true
                }
                (FailureReason::Panic(Some(expected)), FailureResult::Panic(Some(actual))) => {
                    let result = actual.to_lowercase().contains(&expected.to_lowercase());
                    if result {
                        log::error!("Matched failure case: Panic containing \"{}\"", expected);
                    }
                    result
                }
                _ => false,
            };

            if result {
                return true;
            }
        }

        false
    }
}

bitflags::bitflags! {
    /// Reason why a test matches a given failure case.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct FailureApplicationReasons: u8 {
        const BACKEND = 1 << 0;
        const VENDOR = 1 << 1;
        const ADAPTER = 1 << 2;
        const DRIVER = 1 << 3;
        const ALWAYS = 1 << 4;
    }
}

/// Reason why a test is expected to fail.
///
/// If the test fails for a different reason, the given FailureCase will be ignored.
#[derive(Default, Debug, Clone, PartialEq)]
pub enum FailureReason {
    /// Matches any failure.
    #[default]
    Any,
    /// Matches validation errors raised from the backend validation.
    ///
    /// If a string is provided, matches only validation errors that contain the string.
    ValidationError(Option<&'static str>),
    /// A panic was raised.
    ///
    /// If a string is provided, matches only panics that contain the string.
    Panic(Option<&'static str>),
}

#[derive(Default, Clone)]
pub enum FailureBehavior {
    /// Assert that the test fails for the given reason.
    ///
    /// If the test passes, the test harness will panic.
    #[default]
    AssertFailure,
    /// Ignore the matching failure.
    ///
    /// This is useful for tests that flake in a very specific way,
    /// but sometimes succeed, so we can't assert that they always fail.
    Ignore,
}

#[derive(Debug)]
pub(crate) enum FailureResult {
    ValidationError(Option<String>),
    Panic(Option<String>),
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub(crate) enum ExpectationMatchResult {
    Panic,
    Complete,
}

/// Compares if the actual failures match the expected failures.
pub(crate) fn expectations_match_failures(
    expectations: &[FailureCase],
    mut actual: Vec<FailureResult>,
) -> ExpectationMatchResult {
    let mut result = ExpectationMatchResult::Complete;
    for expected_failure in expectations {
        let mut matched = false;
        for f_idx in (0..actual.len()).rev() {
            let failure = &actual[f_idx];
            if expected_failure.matches_failure(failure) {
                actual.swap_remove(f_idx);
                matched = true;
            }
        }

        if !matched && matches!(expected_failure.behavior, FailureBehavior::AssertFailure) {
            result = ExpectationMatchResult::Panic;
            log::error!(
                "Expected to fail due to {:?}, but did not fail",
                expected_failure.reasons()
            );
        }
    }

    if !actual.is_empty() {
        result = ExpectationMatchResult::Panic;
        for failure in actual {
            log::error!("Unexpected failure due to: {:?}", failure);
        }
    }

    result
}

#[cfg(test)]
mod test {
    use crate::{
        expectations::{ExpectationMatchResult, FailureResult},
        init::init_logger,
        FailureBehavior, FailureCase, FailureReason,
    };

    fn always_fail(reason: FailureReason) -> FailureCase {
        FailureCase {
            reasons: vec![reason],
            behavior: FailureBehavior::AssertFailure,
            ..FailureCase::default()
        }
    }

    fn flaky(reason: FailureReason) -> FailureCase {
        FailureCase {
            reasons: vec![reason],
            behavior: FailureBehavior::Ignore,
            ..FailureCase::default()
        }
    }

    fn expect_validation_err(msg: &'static str) -> FailureReason {
        FailureReason::ValidationError(Some(msg))
    }

    fn validation_err(msg: &'static str) -> FailureResult {
        FailureResult::ValidationError(Some(String::from(msg)))
    }

    fn panic(msg: &'static str) -> FailureResult {
        FailureResult::Panic(Some(String::from(msg)))
    }

    #[test]
    fn simple_match() {
        init_logger();

        // -- Unexpected failure --

        let expectation = vec![];
        let actual = vec![FailureResult::ValidationError(None)];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Panic
        );

        // -- Missing expected failure --

        let expectation = vec![always_fail(FailureReason::Any)];
        let actual = vec![];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Panic
        );

        // -- Expected failure (validation) --

        let expectation = vec![always_fail(FailureReason::Any)];
        let actual = vec![FailureResult::ValidationError(None)];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );

        // -- Expected failure (panic) --

        let expectation = vec![always_fail(FailureReason::Any)];
        let actual = vec![FailureResult::Panic(None)];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );
    }

    #[test]
    fn substring_match() {
        init_logger();

        // -- Matching Substring --

        let expectation = vec![always_fail(FailureReason::ValidationError(Some(
            "Some StrIng",
        )))];
        let actual = vec![FailureResult::ValidationError(Some(String::from(
            "a very long string that contains sOmE sTrInG",
        )))];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );

        // -- Non-Matching Substring --

        let expectation = vec![always_fail(expect_validation_err("Some String"))];
        let actual = vec![validation_err("a very long string that doesn't contain it")];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Panic
        );
    }

    #[test]
    fn ignore_flaky() {
        init_logger();

        let expectation = vec![flaky(expect_validation_err("blah"))];
        let actual = vec![validation_err("some blah")];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );

        let expectation = vec![flaky(expect_validation_err("blah"))];
        let actual = vec![];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );
    }

    #[test]
    fn matches_multiple_errors() {
        init_logger();

        // -- matches all matching errors --

        let expectation = vec![always_fail(expect_validation_err("blah"))];
        let actual = vec![
            validation_err("some blah"),
            validation_err("some other blah"),
        ];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );

        // -- but not all errors --

        let expectation = vec![always_fail(expect_validation_err("blah"))];
        let actual = vec![
            validation_err("some blah"),
            validation_err("some other blah"),
            validation_err("something else"),
        ];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Panic
        );
    }

    #[test]
    fn multi_reason_error() {
        init_logger();

        let expectation = vec![FailureCase::default()
            .validation_error("blah")
            .panic("panik")];
        let actual = vec![
            validation_err("my blah blah validation error"),
            panic("my panik"),
        ];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );
    }
}
