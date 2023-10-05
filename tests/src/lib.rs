//! This module contains common test-only code that needs to be shared between the examples and the tests.

use std::panic::AssertUnwindSafe;

use futures_lite::FutureExt;
use wgpu::{Adapter, Device, DownlevelFlags, Instance, Queue};
use wgt::{Backends, DeviceDescriptor, DownlevelCapabilities, Features, Limits};

pub mod image;
pub mod infra;
mod isolation;

use crate::infra::{AdapterReport, GpuTestConfiguration, TestInfo};

pub use self::image::ComparisonType;
pub use ctor::ctor;
pub use wgpu_macros::gpu_test;

#[cfg(all(
    target_arch = "wasm32",
    any(target_os = "emscripten", feature = "webgl")
))]
const CANVAS_ID: &str = "test-canvas";

async fn initialize_device(
    adapter: &Adapter,
    features: Features,
    limits: Limits,
) -> (Device, Queue) {
    let bundle = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                features,
                limits,
            },
            None,
        )
        .await;

    match bundle {
        Ok(b) => b,
        Err(e) => panic!("Failed to initialize device: {e}"),
    }
}

pub struct TestingContext {
    pub adapter: Adapter,
    pub adapter_info: wgt::AdapterInfo,
    pub adapter_downlevel_capabilities: wgt::DownlevelCapabilities,
    pub device: Device,
    pub device_features: wgt::Features,
    pub device_limits: wgt::Limits,
    pub queue: Queue,
}

fn lowest_downlevel_properties() -> DownlevelCapabilities {
    DownlevelCapabilities {
        flags: wgt::DownlevelFlags::empty(),
        limits: wgt::DownlevelLimits {},
        shader_model: wgt::ShaderModel::Sm2,
    }
}

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
    fn applies_to(&self, info: &wgt::AdapterInfo) -> Option<FailureReasons> {
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
            required_downlevel_caps: lowest_downlevel_properties(),
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

pub async fn initialize_test(
    config: GpuTestConfiguration,
    test_info: Option<TestInfo>,
    adapter_index: usize,
) {
    // If we get information externally, skip based on that information before we do anything.
    if let Some(TestInfo { skip: true, .. }) = test_info {
        return;
    }

    // We don't actually care if it fails
    #[cfg(not(target_arch = "wasm32"))]
    let _ = env_logger::try_init();
    #[cfg(target_arch = "wasm32")]
    let _ = console_log::init_with_level(log::Level::Info);

    let _test_guard = isolation::OneTestPerProcessGuard::new();

    let (adapter, _surface_guard) = initialize_adapter(adapter_index);

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

    let (device, queue) = pollster::block_on(initialize_device(
        &adapter,
        config.params.required_features,
        config.params.required_limits.clone(),
    ));

    let context = TestingContext {
        adapter,
        adapter_info,
        adapter_downlevel_capabilities,
        device,
        device_features: config.params.required_features,
        device_limits: config.params.required_limits.clone(),
        queue,
    };

    // Run the test, and catch panics (possibly due to failed assertions).
    let panicked = AssertUnwindSafe((config.test.as_ref().unwrap())(context))
        .catch_unwind()
        .await
        .is_err();

    // Check whether any validation errors were reported during the test run.
    cfg_if::cfg_if!(
        if #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))] {
            let canary_set = wgpu::hal::VALIDATION_CANARY.get_and_reset();
        } else {
            let canary_set = _surface_guard.unwrap().check_for_unreported_errors();
        }
    );

    // Summarize reasons for actual failure, if any.
    let failure_cause = match (panicked, canary_set) {
        (true, true) => Some("PANIC AND VALIDATION ERROR"),
        (true, false) => Some("PANIC"),
        (false, true) => Some("VALIDATION ERROR"),
        (false, false) => None,
    };

    // Compare actual results against expectations.
    match (failure_cause, test_info.expected_failure_reason) {
        // The test passed, as expected.
        (None, None) => log::info!("TEST RESULT: PASSED"),
        // The test failed unexpectedly.
        (Some(cause), None) => {
            panic!("UNEXPECTED TEST FAILURE DUE TO {cause}")
        }
        // The test passed unexpectedly.
        (None, Some(reason)) => {
            panic!("UNEXPECTED TEST PASS: {reason:?}");
        }
        // The test failed, as expected.
        (Some(cause), Some(reason_expected)) => {
            log::info!(
                "TEST RESULT: EXPECTED FAILURE DUE TO {} (expected because of {:?})",
                cause,
                reason_expected
            );
        }
    }
}

pub fn initialize_adapter(adapter_index: usize) -> (Adapter, Option<SurfaceGuard>) {
    let instance = initialize_instance();
    #[allow(unused_variables)]
    let _surface: wgpu::Surface;
    let surface_guard: Option<SurfaceGuard>;

    // Create a canvas iff we need a WebGL2RenderingContext to have a working device.
    #[cfg(not(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    )))]
    {
        surface_guard = None;
    }
    #[cfg(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    ))]
    {
        // On wasm, append a canvas to the document body for initializing the adapter
        let canvas = create_html_canvas();

        // We use raw_window_handle here, as create_surface_from_canvas is not implemented on emscripten.
        struct WindowHandle;
        unsafe impl raw_window_handle::HasRawWindowHandle for WindowHandle {
            fn raw_window_handle(&self) -> raw_window_handle::RawWindowHandle {
                raw_window_handle::RawWindowHandle::Web({
                    let mut handle = raw_window_handle::WebWindowHandle::empty();
                    handle.id = 1;
                    handle
                })
            }
        }
        unsafe impl raw_window_handle::HasRawDisplayHandle for WindowHandle {
            fn raw_display_handle(&self) -> raw_window_handle::RawDisplayHandle {
                raw_window_handle::RawDisplayHandle::Web(
                    raw_window_handle::WebDisplayHandle::empty(),
                )
            }
        }

        _surface = unsafe {
            instance
                .create_surface(&WindowHandle)
                .expect("could not create surface from canvas")
        };

        surface_guard = Some(SurfaceGuard { canvas });
    }

    let adapter_iter = instance.enumerate_adapters(wgpu::Backends::all());
    let adapter_count = adapter_iter.len();
    let adapter = adapter_iter.into_iter()
        .nth(adapter_index)
        .unwrap_or_else(|| panic!("Tried to get index {adapter_index} adapter, but adapter list was only {adapter_count} long. Is .gpuconfig out of date?"));

    log::info!("Testing using adapter: {:#?}", adapter.get_info());

    (adapter, surface_guard)
}

pub fn initialize_instance() -> Instance {
    let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(Backends::all);
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
    let gles_minor_version = wgpu::util::gles_minor_version_from_env().unwrap_or_default();
    Instance::new(wgpu::InstanceDescriptor {
        backends,
        dx12_shader_compiler,
        gles_minor_version,
    })
}

// Public because it is used by tests of interacting with canvas
pub struct SurfaceGuard {
    #[cfg(target_arch = "wasm32")]
    pub canvas: web_sys::HtmlCanvasElement,
}

impl SurfaceGuard {
    #[cfg(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    ))]
    fn check_for_unreported_errors(&self) -> bool {
        use wasm_bindgen::JsCast;

        self.canvas
            .get_context("webgl2")
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::WebGl2RenderingContext>()
            .unwrap()
            .get_error()
            != web_sys::WebGl2RenderingContext::NO_ERROR
    }
}

#[cfg(all(
    target_arch = "wasm32",
    any(target_os = "emscripten", feature = "webgl")
))]
impl Drop for SurfaceGuard {
    fn drop(&mut self) {
        delete_html_canvas();
    }
}

#[cfg(target_arch = "wasm32")]
pub fn create_html_canvas() -> web_sys::HtmlCanvasElement {
    use wasm_bindgen::JsCast;

    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let body = doc.body().unwrap();
            let canvas = doc.create_element("Canvas").unwrap();
            canvas.set_attribute("data-raw-handle", "1").unwrap();
            canvas.set_id(CANVAS_ID);
            body.append_child(&canvas).unwrap();
            canvas.dyn_into::<web_sys::HtmlCanvasElement>().ok()
        })
        .expect("couldn't append canvas to document body")
}

#[cfg(all(
    target_arch = "wasm32",
    any(target_os = "emscripten", feature = "webgl")
))]
fn delete_html_canvas() {
    if let Some(document) = web_sys::window().and_then(|win| win.document()) {
        if let Some(element) = document.get_element_by_id(CANVAS_ID) {
            element.remove();
        }
    };
}

// Run some code in an error scope and assert that validation fails.
pub fn fail<T>(device: &wgpu::Device, callback: impl FnOnce() -> T) -> T {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = callback();
    assert!(pollster::block_on(device.pop_error_scope()).is_some());

    result
}

// Run some code in an error scope and assert that validation succeeds.
pub fn valid<T>(device: &wgpu::Device, callback: impl FnOnce() -> T) -> T {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = callback();
    assert!(pollster::block_on(device.pop_error_scope()).is_none());

    result
}

// Run some code in an error scope and assert that validation succeeds or fails depending on the
// provided `should_fail` boolean.
pub fn fail_if<T>(device: &wgpu::Device, should_fail: bool, callback: impl FnOnce() -> T) -> T {
    if should_fail {
        fail(device, callback)
    } else {
        valid(device, callback)
    }
}

#[macro_export]
macro_rules! gpu_test_main {
    () => {
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

        #[cfg(not(target_arch = "wasm32"))]
        fn main() -> $crate::infra::MainResult {
            $crate::infra::main()
        }
        #[cfg(target_arch = "wasm32")]
        fn main() {}
    };
}
