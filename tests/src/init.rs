use wgpu::{Adapter, Device, Instance, Queue};
use wgt::{Backends, Features, Limits};

/// Initialize the logger for the test runner.
pub fn init_logger() {
    // We don't actually care if it fails
    #[cfg(not(target_arch = "wasm32"))]
    let _ = env_logger::try_init();
    #[cfg(target_arch = "wasm32")]
    let _ = console_log::init_with_level(log::Level::Info);
}

/// Initialize a wgpu instance with the options from the environment.
pub fn initialize_instance(force_fxc: bool) -> Instance {
    // We ignore `WGPU_BACKEND` for now, merely using test filtering to only run a single backend's tests.
    //
    // We can potentially work support back into the test runner in the future, but as the adapters are matched up
    // based on adapter index, removing some backends messes up the indexes in annoying ways.
    //
    // WORKAROUND for https://github.com/rust-lang/cargo/issues/7160:
    // `--no-default-features` is not passed through correctly to the test runner.
    // We use it whenever we want to explicitly run with webgl instead of webgpu.
    // To "disable" webgpu regardless, we do this by removing the webgpu backend whenever we see
    // the webgl feature.
    let backends = if cfg!(feature = "webgl") {
        Backends::all() - Backends::BROWSER_WEBGPU
    } else {
        Backends::all()
    };
    // Some tests need to be able to force demote to FXC, to specifically test workarounds for FXC
    // behavior.
    let dx12_shader_compiler = if force_fxc {
        wgpu::Dx12Compiler::Fxc
    } else {
        wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default()
    };
    let gles_minor_version = wgpu::util::gles_minor_version_from_env().unwrap_or_default();
    Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::debugging().with_env(),
        dx12_shader_compiler,
        gles_minor_version,
    })
}

/// Initialize a wgpu adapter, taking the `n`th adapter from the instance.
pub async fn initialize_adapter(
    adapter_index: usize,
    force_fxc: bool,
) -> (Instance, Adapter, Option<SurfaceGuard>) {
    let instance = initialize_instance(force_fxc);
    #[allow(unused_variables)]
    let surface: Option<wgpu::Surface>;
    let surface_guard: Option<SurfaceGuard>;

    #[allow(unused_assignments)]
    // Create a canvas if we need a WebGL2RenderingContext to have a working device.
    #[cfg(not(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    )))]
    {
        surface = None;
        surface_guard = None;
    }
    #[cfg(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    ))]
    {
        // On wasm, append a canvas to the document body for initializing the adapter
        let canvas = initialize_html_canvas();

        surface = Some(
            instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
                .expect("could not create surface from canvas"),
        );

        surface_guard = Some(SurfaceGuard { canvas });
    }

    cfg_if::cfg_if! {
        if #[cfg(not(target_arch = "wasm32"))] {
            let adapter_iter = instance.enumerate_adapters(wgpu::Backends::all());
            let adapter_count = adapter_iter.len();
            let adapter = adapter_iter.into_iter()
                .nth(adapter_index)
                .unwrap_or_else(|| panic!("Tried to get index {adapter_index} adapter, but adapter list was only {adapter_count} long. Is .gpuconfig out of date?"));
        } else {
            assert_eq!(adapter_index, 0);
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: surface.as_ref(),
                ..Default::default()
            }).await.unwrap();
        }
    }

    log::info!("Testing using adapter: {:#?}", adapter.get_info());

    (instance, adapter, surface_guard)
}

/// Initialize a wgpu device from a given adapter.
pub async fn initialize_device(
    adapter: &Adapter,
    features: Features,
    limits: Limits,
) -> (Device, Queue) {
    let bundle = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await;

    match bundle {
        Ok(b) => b,
        Err(e) => panic!("Failed to initialize device: {e}"),
    }
}

/// Create a canvas for testing.
#[cfg(target_arch = "wasm32")]
pub fn initialize_html_canvas() -> web_sys::HtmlCanvasElement {
    use wasm_bindgen::JsCast;

    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let canvas = doc.create_element("Canvas").unwrap();
            canvas.dyn_into::<web_sys::HtmlCanvasElement>().ok()
        })
        .expect("couldn't create canvas")
}

pub struct SurfaceGuard {
    #[cfg(target_arch = "wasm32")]
    #[allow(unused)]
    canvas: web_sys::HtmlCanvasElement,
}

impl SurfaceGuard {
    #[cfg(all(
        target_arch = "wasm32",
        any(target_os = "emscripten", feature = "webgl")
    ))]
    pub(crate) fn check_for_unreported_errors(&self) -> bool {
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
