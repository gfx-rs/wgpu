use wgpu::{Adapter, Device, Instance, Queue};
use wgt::{Backends, Features, Limits};

pub fn initialize_instance() -> Instance {
    let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(Backends::all);
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
    let gles_minor_version = wgpu::util::gles_minor_version_from_env().unwrap_or_default();
    Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::debugging().with_env(),
        dx12_shader_compiler,
        gles_minor_version,
    })
}

pub async fn initialize_adapter(adapter_index: usize) -> (Adapter, Option<SurfaceGuard>) {
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
        let canvas = initialize_html_canvas();

        _surface = instance
            .create_surface_from_canvas(canvas.clone())
            .expect("could not create surface from canvas");

        surface_guard = Some(SurfaceGuard { canvas });
    }

    cfg_if::cfg_if! {
        if #[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))] {
            let adapter_iter = instance.enumerate_adapters(wgpu::Backends::all());
            let adapter_count = adapter_iter.len();
            let adapter = adapter_iter.into_iter()
                .nth(adapter_index)
                .unwrap_or_else(|| panic!("Tried to get index {adapter_index} adapter, but adapter list was only {adapter_count} long. Is .gpuconfig out of date?"));
        } else {
            assert_eq!(adapter_index, 0);
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
        }
    }

    log::info!("Testing using adapter: {:#?}", adapter.get_info());

    (adapter, surface_guard)
}

pub async fn initialize_device(
    adapter: &Adapter,
    features: Features,
    limits: Limits,
) -> (Device, Queue) {
    let bundle = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
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
