//! This example demonstrates wgpu's error scopes.
//!
//! Error scopes let you check whether GPU work inside the scope produced any
//! errors. Push a scope with [`wgpu::Device::push_error_scope`], do some work,
//! then await [`wgpu::ErrorScopeGuard::pop`] to get the first error, if any.
//!
//! The example compiles a valid shader (no error), an invalid shader (a
//! validation error), and checks nested scopes pop in LIFO order. It does not
//! render anything; results are logged to the console. Runs on the webgl2,
//! webgpu and native backends (a vertex shader is used since WebGL2 has no
//! compute shaders; on wasm a fresh canvas is created for the surface, which
//! is never presented to).

fn error_description(err: &wgpu::Error) -> String {
    match err {
        wgpu::Error::Validation { description, .. } => format!("validation: {description}"),
        wgpu::Error::Internal { description, .. } => format!("internal: {description}"),
        wgpu::Error::OutOfMemory { .. } => "out of memory".to_string(),
    }
}

async fn run() {
    let instance = wgpu::Instance::default();

    // On wasm the WebGL2 backend needs a surface to enumerate an adapter;
    // WebGPU does not. The surface is never presented to.
    #[cfg(target_arch = "wasm32")]
    let compatible_surface = {
        use wasm_bindgen::JsCast;
        let canvas: web_sys::HtmlCanvasElement = web_sys::window()
            .and_then(|w| w.document())
            .expect("no document")
            .create_element("canvas")
            .expect("failed to create canvas")
            .dyn_into()
            .expect("created element is not a canvas");
        Some(
            instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                .unwrap(),
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let compatible_surface: Option<wgpu::Surface<'static>> = None;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: compatible_surface.as_ref(),
            ..Default::default()
        })
        .await
        .expect("no adapter");
    let (device, _queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            default_queue: wgpu::QueueDescriptor { label: None },
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("no device");

    const VALID_SHADER: &str =
        "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }";
    const INVALID_SHADER: &str = "this is not valid wgsl";

    fn create_module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    let mut failures = 0;

    // A valid shader inside a scope produces no error.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    create_module(&device, VALID_SHADER);
    match scope.pop().await {
        None => log::info!("PASS: valid shader produced no error"),
        Some(err) => {
            failures += 1;
            log::error!("FAIL: expected no error, got: {}", error_description(&err));
        }
    }

    // An invalid shader produces a validation error, captured by the scope.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    create_module(&device, INVALID_SHADER);
    match scope.pop().await {
        Some(wgpu::Error::Validation { description, .. }) => {
            log::info!("PASS: invalid shader produced a validation error: {description}");
        }
        other => {
            failures += 1;
            log::error!(
                "FAIL: expected a validation error, got: {}",
                other
                    .as_ref()
                    .map(error_description)
                    .unwrap_or_else(|| "None".to_string())
            );
        }
    }

    // Nested scopes pop in LIFO order; the inner scope captures the error.
    let outer = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let inner = device.push_error_scope(wgpu::ErrorFilter::Validation);
    create_module(&device, INVALID_SHADER);
    let inner_err = inner.pop().await;
    let outer_err = outer.pop().await;
    match (&inner_err, &outer_err) {
        (Some(wgpu::Error::Validation { description, .. }), None) => {
            log::info!("PASS: nested scopes attribute the error to the inner scope: {description}")
        }
        other => {
            failures += 1;
            let (inner, outer) = other;
            log::error!(
                "FAIL: expected (inner=Validation, outer=None), got (inner={}, outer={})",
                inner
                    .as_ref()
                    .map(error_description)
                    .unwrap_or_else(|| "None".to_string()),
                outer
                    .as_ref()
                    .map(error_description)
                    .unwrap_or_else(|| "None".to_string())
            );
        }
    }

    if failures > 0 {
        panic!("error_scope example failed: {failures} check(s) failed");
    }
    log::info!("error_scope: all checks passed");
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");

        web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("content"))
            .expect("could not get #content element")
            .set_inner_html(
                "<h1>This example tests wgpu error scopes. Open the console to see the results!</h1>",
            );

        wasm_bindgen_futures::spawn_local(run());
    }
}
