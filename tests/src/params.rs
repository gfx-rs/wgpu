use arrayvec::ArrayVec;
use wgt::{DownlevelCapabilities, Features, Limits, DownlevelFlags};

use crate::{GpuTestConfiguration, report::AdapterReport};

/// Conditions under which a test should fail or be skipped.
///
/// By passing a `FailureCase` to [`TestParameters::expect_fail`], you can
/// mark a test as expected to fail under the indicated conditions. By
/// passing it to [`TestParameters::skip`], you can request that the
/// test be skipped altogether.
///
/// If a field is `None`, then that field does not restrict matches. For
/// example:
///
/// ```
/// # use wgpu_test::FailureCase;
/// FailureCase {
///     backends: Some(wgpu::Backends::DX11 | wgpu::Backends::DX12),
///     vendor: None,
///     adapter: Some("RTX"),
///     driver: None,
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
    ///
    /// Because of wasm's limited ability to recover from errors, we
    /// usually need to skip the test altogether if it's not
    /// supported, so this should be usually used with
    /// [`TestParameters::skip`].
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

    /// Test whether `self` applies to `info`.
    ///
    /// If it does, return a `FailureReasons` whose set bits indicate
    /// why. If it doesn't, return `None`.
    ///
    /// The caller is responsible for converting the string-valued
    /// fields of `info` to lower case, to ensure case-insensitive
    /// matching.
    pub(crate) fn applies_to(&self, info: &wgt::AdapterInfo) -> Option<FailureReasons> {
        let mut reasons = FailureReasons::empty();

        if let Some(backends) = self.backends {
            if !backends.contains(wgpu::Backends::from(info.backend)) {
                return None;
            }
            reasons.set(FailureReasons::BACKEND, true);
        }
        if let Some(vendor) = self.vendor {
            if vendor != info.vendor {
                return None;
            }
            reasons.set(FailureReasons::VENDOR, true);
        }
        if let Some(adapter) = self.adapter {
            let adapter = adapter.to_lowercase();
            if !info.name.contains(&adapter) {
                return None;
            }
            reasons.set(FailureReasons::ADAPTER, true);
        }
        if let Some(driver) = self.driver {
            let driver = driver.to_lowercase();
            if !info.driver.contains(&driver) {
                return None;
            }
            reasons.set(FailureReasons::DRIVER, true);
        }

        // If we got this far but no specific reasons were triggered, then this
        // must be a wildcard.
        if reasons.is_empty() {
            Some(FailureReasons::ALWAYS)
        } else {
            Some(reasons)
        }
    }
}

const LOWEST_DOWNLEVEL_PROPERTIES: wgpu::DownlevelCapabilities = DownlevelCapabilities {
    flags: wgt::DownlevelFlags::empty(),
    limits: wgt::DownlevelLimits {},
    shader_model: wgt::ShaderModel::Sm2,
};

// This information determines if a test should run.
#[derive(Clone)]
pub struct TestParameters {
    pub required_features: Features,
    pub required_downlevel_caps: DownlevelCapabilities,
    pub required_limits: Limits,

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
            skips: Vec::new(),
            failures: Vec::new(),
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct FailureReasons: u8 {
        const BACKEND = 1 << 0;
        const VENDOR = 1 << 1;
        const ADAPTER = 1 << 2;
        const DRIVER = 1 << 3;
        const ALWAYS = 1 << 4;
    }
}

// Builder pattern to make it easier
impl TestParameters {
    /// Set of common features that most internal tests require for readback.
    pub fn test_features_limits(self) -> Self {
        self.features(Features::MAPPABLE_PRIMARY_BUFFERS | Features::VERTEX_WRITABLE_STORAGE)
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

pub struct TestInfo {
    pub skip: bool,
    pub expected_failure_reason: Option<FailureReasons>,
    pub running_msg: String,
}

impl TestInfo {
    pub(crate) fn from_configuration(test: &GpuTestConfiguration, adapter: &AdapterReport) -> Self {
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

        let expected_failure_reason = test
            .params
            .failures
            .iter()
            .find_map(|case| case.applies_to(&adapter_lowercase_info));

        let mut skip = false;
        let running_msg = if let Some(reasons) = skip_reason {
            skip = true;

            let names: ArrayVec<_, 4> = reasons.iter_names().map(|(name, _)| name).collect();
            let names_text = names.join(" | ");

            format!("Skipped Failure: {}", names_text)
        } else if !skipped_reasons.is_empty() {
            skip = true;
            format!("Skipped: {}", skipped_reasons.join(" | "))
        } else if let Some(failure_resasons) = expected_failure_reason {
            if cfg!(target_arch = "wasm32") {
                skip = true;
            }

            let names: ArrayVec<_, 4> = failure_resasons
                .iter_names()
                .map(|(name, _)| name)
                .collect();
            let names_text = names.join(" | ");

            format!("Executed Failure: {}", names_text)
        } else {
            String::from("Executed")
        };

        Self {
            skip,
            expected_failure_reason,
            running_msg,
        }
    }
}
