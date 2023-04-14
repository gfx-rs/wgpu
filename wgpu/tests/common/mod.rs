//! This module contains common test-only code that needs to be shared between the examples and the tests.
#![allow(dead_code)] // This module is used in a lot of contexts and only parts of it will be used

use std::panic::{catch_unwind, AssertUnwindSafe};

use wgpu::{Adapter, Device, DownlevelFlags, Instance, Queue, Surface};
use wgt::{Backends, DeviceDescriptor, DownlevelCapabilities, Features, Limits};

pub mod image;
mod isolation;

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

pub struct FailureCase {
    backends: Option<wgpu::Backends>,
    vendor: Option<usize>,
    adapter: Option<String>,
    skip: bool,
}

// This information determines if a test should run.
pub struct TestParameters {
    pub required_features: Features,
    pub required_downlevel_properties: DownlevelCapabilities,
    pub required_limits: Limits,
    // Backends where test should fail.
    pub failures: Vec<FailureCase>,
}

impl Default for TestParameters {
    fn default() -> Self {
        Self {
            required_features: Features::empty(),
            required_downlevel_properties: lowest_downlevel_properties(),
            required_limits: Limits::downlevel_webgl2_defaults(),
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
        const ALWAYS = 1 << 3;
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
        self.required_downlevel_properties.flags |= downlevel_flags;
        self
    }

    /// Set the limits needed for the test.
    pub fn limits(mut self, limits: Limits) -> Self {
        self.required_limits = limits;
        self
    }

    /// Mark the test as always failing, equivalent to specific_failure(None, None, None)
    pub fn failure(mut self) -> Self {
        self.failures.push(FailureCase {
            backends: None,
            vendor: None,
            adapter: None,
            skip: false,
        });
        self
    }

    /// Mark the test as always failing and needing to be skipped, equivalent to specific_failure(None, None, None)
    pub fn skip(mut self) -> Self {
        self.failures.push(FailureCase {
            backends: None,
            vendor: None,
            adapter: None,
            skip: true,
        });
        self
    }

    /// Mark the test as always failing on a specific backend, equivalent to specific_failure(backend, None, None)
    pub fn backend_failure(mut self, backends: wgpu::Backends) -> Self {
        self.failures.push(FailureCase {
            backends: Some(backends),
            vendor: None,
            adapter: None,
            skip: false,
        });
        self
    }

    /// Mark the test as always failing on WebGL. Because limited ability of wasm to recover from errors, we need to wholesale
    /// skip the test if it's not supported.
    pub fn webgl2_failure(mut self) -> Self {
        let _ = &mut self;
        #[cfg(target_arch = "wasm32")]
        self.failures.push(FailureCase {
            backends: Some(wgpu::Backends::GL),
            vendor: None,
            adapter: None,
            skip: true,
        });
        self
    }

    /// Determines if a test should fail under a particular set of conditions. If any of these are None, that means that it will match anything in that field.
    ///
    /// ex.
    /// `specific_failure(Some(wgpu::Backends::DX11 | wgpu::Backends::DX12), None, Some("RTX"), false)`
    /// means that this test will fail on all cards with RTX in their name on either D3D backend, no matter the vendor ID.
    ///
    /// If segfault is set to true, the test won't be run at all due to avoid segfaults.
    pub fn specific_failure(
        mut self,
        backends: Option<Backends>,
        vendor: Option<usize>,
        device: Option<&'static str>,
        skip: bool,
    ) -> Self {
        self.failures.push(FailureCase {
            backends,
            vendor,
            adapter: device.as_ref().map(AsRef::as_ref).map(str::to_lowercase),
            skip,
        });
        self
    }
}

pub fn initialize_test(parameters: TestParameters, test_function: impl FnOnce(TestingContext)) {
    // We don't actually care if it fails
    #[cfg(not(target_arch = "wasm32"))]
    let _ = env_logger::try_init();
    #[cfg(target_arch = "wasm32")]
    let _ = console_log::init_with_level(log::Level::Info);

    let _test_guard = isolation::OneTestPerProcessGuard::new();

    let (adapter, _surface_guard) = initialize_adapter();

    let adapter_info = adapter.get_info();
    let adapter_lowercase_name = adapter_info.name.to_lowercase();
    let adapter_features = adapter.features();
    let adapter_limits = adapter.limits();
    let adapter_downlevel_capabilities = adapter.get_downlevel_capabilities();

    let missing_features = parameters.required_features - adapter_features;
    if !missing_features.is_empty() {
        log::info!("TEST SKIPPED: MISSING FEATURES {:?}", missing_features);
        return;
    }

    if !parameters.required_limits.check_limits(&adapter_limits) {
        log::info!("TEST SKIPPED: LIMIT TOO LOW");
        return;
    }

    let missing_downlevel_flags =
        parameters.required_downlevel_properties.flags - adapter_downlevel_capabilities.flags;
    if !missing_downlevel_flags.is_empty() {
        log::info!(
            "TEST SKIPPED: MISSING DOWNLEVEL FLAGS {:?}",
            missing_downlevel_flags
        );
        return;
    }

    if adapter_downlevel_capabilities.shader_model
        < parameters.required_downlevel_properties.shader_model
    {
        log::info!(
            "TEST SKIPPED: LOW SHADER MODEL {:?}",
            adapter_downlevel_capabilities.shader_model
        );
        return;
    }

    let (device, queue) = pollster::block_on(initialize_device(
        &adapter,
        parameters.required_features,
        parameters.required_limits.clone(),
    ));

    let context = TestingContext {
        adapter,
        adapter_info: adapter_info.clone(),
        adapter_downlevel_capabilities,
        device,
        device_features: parameters.required_features,
        device_limits: parameters.required_limits,
        queue,
    };

    let expected_failure_reason = parameters.failures.iter().find_map(|failure| {
        let always =
            failure.backends.is_none() && failure.vendor.is_none() && failure.adapter.is_none();

        let expect_failure_backend = failure
            .backends
            .map(|f| f.contains(wgpu::Backends::from(adapter_info.backend)));
        let expect_failure_vendor = failure.vendor.map(|v| v == adapter_info.vendor);
        let expect_failure_adapter = failure
            .adapter
            .as_deref()
            .map(|f| adapter_lowercase_name.contains(f));

        if expect_failure_backend.unwrap_or(true)
            && expect_failure_vendor.unwrap_or(true)
            && expect_failure_adapter.unwrap_or(true)
        {
            if always {
                Some((FailureReasons::ALWAYS, failure.skip))
            } else {
                let mut reason = FailureReasons::empty();
                reason.set(
                    FailureReasons::BACKEND,
                    expect_failure_backend.unwrap_or(false),
                );
                reason.set(
                    FailureReasons::VENDOR,
                    expect_failure_vendor.unwrap_or(false),
                );
                reason.set(
                    FailureReasons::ADAPTER,
                    expect_failure_adapter.unwrap_or(false),
                );
                Some((reason, failure.skip))
            }
        } else {
            None
        }
    });

    if let Some((reason, true)) = expected_failure_reason {
        log::info!("EXPECTED TEST FAILURE SKIPPED: {:?}", reason);
        return;
    }

    let panicked = catch_unwind(AssertUnwindSafe(|| test_function(context))).is_err();
    cfg_if::cfg_if!(
        if #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))] {
            let canary_set = hal::VALIDATION_CANARY.get_and_reset();
        } else {
            let canary_set = _surface_guard.check_for_unreported_errors();
        }
    );

    let failed = panicked || canary_set;

    let failure_cause = match (panicked, canary_set) {
        (true, true) => "PANIC AND VALIDATION ERROR",
        (true, false) => "PANIC",
        (false, true) => "VALIDATION ERROR",
        (false, false) => "",
    };

    let expect_failure = expected_failure_reason.is_some();

    if failed == expect_failure {
        // We got the conditions we expected
        if let Some((expected_reason, _)) = expected_failure_reason {
            // Print out reason for the failure
            log::info!(
                "GOT EXPECTED TEST FAILURE DUE TO {}: {:?}",
                failure_cause,
                expected_reason
            );
        }
    } else if let Some((reason, _)) = expected_failure_reason {
        // We expected to fail, but things passed
        panic!("UNEXPECTED TEST PASS: {reason:?}");
    } else {
        panic!("UNEXPECTED TEST FAILURE DUE TO {failure_cause}")
    }
}

fn initialize_adapter() -> (Adapter, SurfaceGuard) {
    let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(Backends::all);
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
    let instance = Instance::new(wgpu::InstanceDescriptor {
        backends,
        dx12_shader_compiler,
    });
    let surface_guard;
    let compatible_surface;

    #[cfg(not(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    )))]
    {
        surface_guard = SurfaceGuard {};
        compatible_surface = None;
    }
    #[cfg(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    ))]
    {
        // On wasm, append a canvas to the document body for initializing the adapter
        let canvas = create_html_canvas();

        let surface = instance
            .create_surface_from_canvas(canvas.clone())
            .expect("could not create surface from canvas");

        surface_guard = SurfaceGuard { canvas };

        compatible_surface = Some(surface);
    }

    let compatible_surface: Option<&Surface> = compatible_surface.as_ref();
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance,
        backends,
        compatible_surface,
    ))
    .expect("could not find suitable adapter on the system");

    (adapter, surface_guard)
}

struct SurfaceGuard {
    #[cfg(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    ))]
    canvas: web_sys::HtmlCanvasElement,
}

impl SurfaceGuard {
    fn check_for_unreported_errors(&self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(all(target_arch = "wasm32", any(target_os = "emscripten", feature = "webgl")))] {
                use wasm_bindgen::JsCast;

                self.canvas
                    .get_context("webgl2")
                    .unwrap()
                    .unwrap()
                    .dyn_into::<web_sys::WebGl2RenderingContext>()
                    .unwrap()
                    .get_error()
                    != web_sys::WebGl2RenderingContext::NO_ERROR
            } else {
                false
            }
        }
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

#[cfg(all(
    target_arch = "wasm32",
    any(target_os = "emscripten", feature = "webgl")
))]
fn create_html_canvas() -> web_sys::HtmlCanvasElement {
    use wasm_bindgen::JsCast;

    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let body = doc.body().unwrap();
            let canvas = doc.create_element("Canvas").unwrap();
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
