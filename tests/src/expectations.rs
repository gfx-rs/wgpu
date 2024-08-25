use core::fmt;

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
///     backends: Some(wgpu::Backends::DX12),
///     vendor: None,
///     adapter: Some("RTX"),
///     driver: None,
///     reasons: vec![FailureReason::validation_error().with_message("Some error substring")],
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

    /// Name of adapter expected to fail, or `None` for any adapter name.
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
    /// Create a new failure case.
    pub fn new() -> Self {
        Self::default()
    }

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
            std::array::from_ref(&FailureReason::ANY)
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
        self.reasons
            .push(FailureReason::validation_error().with_message(msg));
        self
    }

    /// Matches this failure case against the given panic substring.
    ///
    /// Substrings are matched case-insensitively.
    ///
    /// If multiple reasons are pushed, will match any of them.
    pub fn panic(mut self, msg: &'static str) -> Self {
        self.reasons.push(FailureReason::panic().with_message(msg));
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

    /// Returns true if the given failure "satisfies" this failure case.
    pub(crate) fn matches_failure(&self, failure: &FailureResult) -> bool {
        for reason in self.reasons() {
            let kind_matched = reason.kind.map_or(true, |kind| kind == failure.kind);

            let message_matched =
                reason
                    .message
                    .map_or(true, |message| matches!(&failure.message, Some(actual) if actual.to_lowercase().contains(&message.to_lowercase())));

            if kind_matched && message_matched {
                let message = failure.message.as_deref().unwrap_or("*no message*");
                log::error!("Matched {} {message}", failure.kind);
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
pub struct FailureReason {
    /// Match a particular kind of failure result.
    ///
    /// If `None`, match any result kind.
    kind: Option<FailureResultKind>,
    /// Match a particular message of a failure result.
    ///
    /// If `None`, matches any message. If `Some`, a case-insensitive sub-string
    /// test is performed. Allowing `"error occurred"` to match a message like
    /// `"An unexpected Error occurred!"`.
    message: Option<&'static str>,
}

impl FailureReason {
    /// Match any failure reason.
    const ANY: Self = Self {
        kind: None,
        message: None,
    };

    /// Match a validation error.
    #[allow(dead_code)] // Not constructed on wasm
    pub fn validation_error() -> Self {
        Self {
            kind: Some(FailureResultKind::ValidationError),
            message: None,
        }
    }

    /// Match a panic.
    pub fn panic() -> Self {
        Self {
            kind: Some(FailureResultKind::Panic),
            message: None,
        }
    }

    /// Match an error with a message.
    ///
    /// If specified, a case-insensitive sub-string test is performed. Allowing
    /// `"error occurred"` to match a message like `"An unexpected Error
    /// occurred!"`.
    pub fn with_message(self, message: &'static str) -> Self {
        Self {
            message: Some(message),
            ..self
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FailureResultKind {
    #[allow(dead_code)] // Not constructed on wasm
    ValidationError,
    Panic,
}

impl fmt::Display for FailureResultKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FailureResultKind::ValidationError => write!(f, "Validation Error"),
            FailureResultKind::Panic => write!(f, "Panic"),
        }
    }
}

#[derive(Debug)]
pub(crate) struct FailureResult {
    kind: FailureResultKind,
    message: Option<String>,
}

impl FailureResult {
    /// Failure result is a panic.
    pub(super) fn panic() -> Self {
        Self {
            kind: FailureResultKind::Panic,
            message: None,
        }
    }

    /// Failure result is a validation error.
    #[allow(dead_code)] // Not constructed on wasm
    pub(super) fn validation_error() -> Self {
        Self {
            kind: FailureResultKind::ValidationError,
            message: None,
        }
    }

    /// Message associated with a failure result.
    pub(super) fn with_message(self, message: impl fmt::Display) -> Self {
        Self {
            kind: self.kind,
            message: Some(message.to_string()),
        }
    }
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
    // Start with the assumption that we will pass.
    let mut result = ExpectationMatchResult::Complete;

    // Run through all expected failures.
    for expected_failure in expectations {
        // If any of the failures match.
        let mut matched = false;

        // Iterate through the failures.
        //
        // In reverse, to be able to use swap_remove.
        actual.retain(|failure| {
            // If the failure matches, remove it from the list of failures, as we expected it.
            let matches = expected_failure.matches_failure(failure);

            if matches {
                matched = true;
            }

            // Retain removes on false, so flip the bool so we remove on failure.
            !matches
        });

        // If we didn't match our expected failure against any of the actual failures,
        // and this failure is not flaky, then we need to panic, as we got an unexpected success.
        if !matched && matches!(expected_failure.behavior, FailureBehavior::AssertFailure) {
            result = ExpectationMatchResult::Panic;
            log::error!(
                "Expected to fail due to {:?}, but did not fail",
                expected_failure.reasons()
            );
        }
    }

    // If we have any failures left, then we got an unexpected failure
    // and we need to panic.
    if !actual.is_empty() {
        result = ExpectationMatchResult::Panic;
        for failure in actual {
            let message = failure.message.as_deref().unwrap_or("*no message*");
            log::error!("{}: {message}", failure.kind);
        }
    }

    result
}

#[cfg(test)]
mod test {
    use crate::{
        expectations::{ExpectationMatchResult, FailureResult},
        init::init_logger,
        FailureCase,
    };

    fn validation_err(msg: &'static str) -> FailureResult {
        FailureResult::validation_error().with_message(msg)
    }

    fn panic(msg: &'static str) -> FailureResult {
        FailureResult::panic().with_message(msg)
    }

    #[test]
    fn simple_match() {
        init_logger();

        // -- Unexpected failure --

        let expectation = vec![];
        let actual = vec![FailureResult::validation_error()];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Panic
        );

        // -- Missing expected failure --

        let expectation = vec![FailureCase::always()];
        let actual = vec![];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Panic
        );

        // -- Expected failure (validation) --

        let expectation = vec![FailureCase::always()];
        let actual = vec![FailureResult::validation_error()];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );

        // -- Expected failure (panic) --

        let expectation = vec![FailureCase::always()];
        let actual = vec![FailureResult::panic()];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );
    }

    #[test]
    fn substring_match() {
        init_logger();

        // -- Matching Substring --

        let expectation: Vec<FailureCase> =
            vec![FailureCase::always().validation_error("Some StrIng")];
        let actual = vec![FailureResult::validation_error().with_message(
            "a very long string that contains sOmE sTrInG of different capitalization",
        )];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );

        // -- Non-Matching Substring --

        let expectation = vec![FailureCase::always().validation_error("Some String")];
        let actual = vec![validation_err("a very long string that doesn't contain it")];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Panic
        );
    }

    #[test]
    fn ignore_flaky() {
        init_logger();

        let expectation = vec![FailureCase::always().validation_error("blah").flaky()];
        let actual = vec![validation_err("some blah")];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );

        let expectation = vec![FailureCase::always().validation_error("blah").flaky()];
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

        let expectation = vec![FailureCase::always().validation_error("blah")];
        let actual = vec![
            validation_err("some blah"),
            validation_err("some other blah"),
        ];

        assert_eq!(
            super::expectations_match_failures(&expectation, actual),
            ExpectationMatchResult::Complete
        );

        // -- but not all errors --

        let expectation = vec![FailureCase::always().validation_error("blah")];
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
