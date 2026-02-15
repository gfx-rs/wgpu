use serde::{Deserialize, Serialize};

/// Prefix emitted in stdout before JSON-serialized [`GpuHarnessEvent`] values.
///
/// `wgpu-test` writes lines in the form:
/// `WGPU_GPU_TEST_EVENT:{...json...}`
/// and `wgpu-xtask` scans test stdout for this marker.
pub const GPU_TEST_EVENT_PREFIX: &str = "WGPU_GPU_TEST_EVENT:";

/// A normalized failure descriptor used for expectation matching.
///
/// Both fields are optional so patterns can be as broad or as specific as needed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureSignature {
    /// Optional failure class, such as `"panic"` or `"test abort"`.
    pub kind: Option<String>,
    /// Optional human-readable error text used for substring matching.
    pub message: Option<String>,
}

/// Stable adapter identity for baseline/event matching.
///
/// This intentionally does not include transient adapter enumeration indices so
/// results remain comparable across runs when adapter ordering changes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdapterKey {
    /// Backend/API name, for example `"Vulkan"`, `"Dx12"`, `"Metal"`, `"Gl"`.
    pub backend: String,
    /// PCI vendor ID when available.
    pub vendor: u32,
    /// PCI device ID when available.
    pub device: u32,
    /// Human-readable adapter name.
    pub name: String,
    /// Driver identity/version string reported by `wgpu`.
    pub driver: String,
}

/// Identity for a single GPU test case execution target.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuTestKey {
    /// Canonical Rust test path, for example `wgpu_gpu::foo::bar`.
    pub test_path: String,
    /// Adapter the test was parameterized against.
    pub adapter: AdapterKey,
}

/// Event lifecycle stage emitted by the GPU harness.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventPhase {
    /// Emitted before running test body logic.
    Before,
    /// Emitted after the test body (if execution reaches completion reporting).
    After,
}

/// Structured event emitted by `wgpu-test` and consumed by `wgpu-xtask`.
///
/// A `Before` event communicates inline expectations and skip decisions.
/// An `After` event communicates observed outcome data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuHarnessEvent {
    /// Schema version for forward/backward compatibility.
    pub version: u32,
    /// Whether this record is a pre-run (`Before`) or post-run (`After`) event.
    pub phase: EventPhase,
    /// Stable test+adapter identity this event describes.
    pub key: GpuTestKey,
    /// True when inline test metadata says the test is expected to fail.
    pub inline_expect_fail: bool,
    /// True when inline test metadata says failure may be a crash/abort.
    pub inline_expect_crash: bool,
    /// True when the harness decides this case should be skipped.
    pub skip: bool,
    /// True when `skip` is specifically due to unsupported capability/feature.
    pub skip_due_to_unsupported: bool,
    /// True when `skip` is specifically due to inline expectation management.
    pub skip_due_to_expectation: bool,
    /// Expected failure signatures derived from inline expectation rules.
    pub expected_failure_signatures: Vec<FailureSignature>,
    /// Actual boolean outcome:
    /// - `Some(true)`: case completed as success
    /// - `Some(false)`: case completed as failure
    /// - `None`: no definitive pass/fail bit was recorded (commonly skipped)
    pub actual_success: Option<bool>,
    /// Observed failure signatures from the run (panic/abort/etc).
    pub actual_failure_signatures: Vec<FailureSignature>,
    /// Optional free-form verdict string from harness expectation reconciliation.
    pub expectation_verdict: Option<String>,
}
