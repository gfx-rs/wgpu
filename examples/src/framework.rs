use std::sync::Arc;

use wgpu::{Instance, Surface};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

pub trait Example: 'static + Sized {
    const SRGB: bool = true;

    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::empty(),
            shader_model: wgpu::ShaderModel::Sm5,
            ..wgpu::DownlevelCapabilities::default()
        }
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_webgl2_defaults() // These downlevel limits will allow the code to run on all possible hardware
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self;

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );

    fn update(&mut self, event: WindowEvent);

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue);
}

// Initialize logging in platform dependant ways.
fn init_logger() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            // As we don't have an environment to pull logging level from, we use the query string.
            let query_string = web_sys::window().unwrap().location().search().unwrap();
            let query_level: Option<log::LevelFilter> = parse_url_query_string(&query_string, "RUST_LOG")
                .and_then(|x| x.parse().ok());

            // We keep wgpu at Error level, as it's very noisy.
            let base_level = query_level.unwrap_or(log::LevelFilter::Info);
            let wgpu_level = query_level.unwrap_or(log::LevelFilter::Error);

            // On web, we use fern, as console_log doesn't have filtering on a per-module level.
            fern::Dispatch::new()
                .level(base_level)
                .level_for("wgpu_core", wgpu_level)
                .level_for("wgpu_hal", wgpu_level)
                .level_for("naga", wgpu_level)
                .chain(fern::Output::call(console_log::log))
                .apply()
                .unwrap();
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        } else {
            // parse_default_env will read the RUST_LOG environment variable and apply it on top
            // of these default filters.
            env_logger::builder()
                .filter_level(log::LevelFilter::Info)
                // We keep wgpu at Error level, as it's very noisy.
                .filter_module("wgpu_core", log::LevelFilter::Info)
                .filter_module("wgpu_hal", log::LevelFilter::Error)
                .filter_module("naga", log::LevelFilter::Error)
                .parse_default_env()
                .init();
        }
    }
}

/// Wrapper type which manages the surface and surface configuration.
///
/// As surface usage varies per platform, wrapping this up cleans up the event loop code.
struct SurfaceWrapper {
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
}

impl SurfaceWrapper {
    /// Create a new surface wrapper with a surface.
    ///
    /// This should be called before the adapter is created.
    fn new(instance: &Instance, window: Arc<Window>) -> Self {
        Self {
            surface: Some(instance.create_surface(window).unwrap()),
            config: None,
        }
    }

    /// Configure the surface with the given adapter and device.
    ///
    /// This should be called after the device is created.
    fn configure(
        &mut self,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        window: &Window,
        srgb: bool,
    ) {
        // Window size is only actually valid after we enter the event loop.
        let window_size = window.inner_size();
        let width = window_size.width.max(1);
        let height = window_size.height.max(1);

        log::info!("Surface resume {window_size:?}");

        let surface = self.surface.as_ref().unwrap();

        // Get the default configuration,
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .expect("Surface isn't supported by the adapter.");
        if srgb {
            // Not all platforms (WebGPU) support sRGB swapchains, so we need to use view formats
            let view_format = config.format.add_srgb_suffix();
            config.view_formats.push(view_format);
        } else {
            // All platforms support non-sRGB swapchains, so we can just use the format directly.
            let format = config.format.remove_srgb_suffix();
            config.format = format;
            config.view_formats.push(format);
        };

        surface.configure(&device, &config);
        self.config = Some(config);
    }

    /// Resume event which recreates the surface, if it was dropped.
    fn resume(
        &mut self,
        instance: &Instance,
        window: &Arc<Window>,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        srgb: bool,
    ) {
        if self.surface.is_none() {
            self.surface = Some(instance.create_surface(window.clone()).unwrap());
            self.configure(adapter, device, window, srgb);
        }
    }

    /// Resize the surface, making sure to not resize to zero.
    fn resize(&mut self, device: &wgpu::Device, size: PhysicalSize<u32>) {
        log::info!("Surface resize {size:?}");

        let config = self.config.as_mut().unwrap();
        config.width = size.width.max(1);
        config.height = size.height.max(1);
        let surface = self.surface.as_ref().unwrap();
        surface.configure(&device, config);
    }

    /// Acquire the next surface texture.
    fn acquire(&mut self, device: &wgpu::Device) -> wgpu::SurfaceTexture {
        let surface = self.surface.as_ref().unwrap();

        match surface.get_current_texture() {
            Ok(frame) => frame,
            // If we timed out, just try again
            Err(wgpu::SurfaceError::Timeout) => surface
                .get_current_texture()
                .expect("Failed to acquire next surface texture!"),
            Err(
                // If the surface is outdated, or was lost, reconfigure it.
                wgpu::SurfaceError::Outdated
                | wgpu::SurfaceError::Lost
                // If OutOfMemory happens, reconfiguring may not help, but we might as well try
                | wgpu::SurfaceError::OutOfMemory,
            ) => {
                surface.configure(&device, self.config());
                surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            }
        }
    }

    /// On suspend on android, we drop the surface, as it's no longer valid.
    ///
    /// A suspend event is always followed by at least one resume event.
    fn suspend(&mut self) {
        self.surface = None;
    }

    fn get(&self) -> Option<&Surface> {
        self.surface.as_ref()
    }

    fn config(&self) -> &wgpu::SurfaceConfiguration {
        self.config.as_ref().unwrap()
    }
}

struct FrameCounter {
    // Instant of the last time we printed the frame time.
    last_printed_instant: web_time::Instant,
    // Number of frames since the last time we printed the frame time.
    frame_count: u32,
}

impl FrameCounter {
    fn new() -> Self {
        Self {
            last_printed_instant: web_time::Instant::now(),
            frame_count: 0,
        }
    }

    fn update(&mut self) {
        self.frame_count += 1;
        let new_instant = web_time::Instant::now();
        let elapsed_secs = (new_instant - self.last_printed_instant).as_secs_f32();
        if elapsed_secs > 1.0 {
            let elapsed_ms = elapsed_secs * 1000.0;
            let frame_time = elapsed_ms / self.frame_count as f32;
            let fps = self.frame_count as f32 / elapsed_secs;
            log::info!("Frame time {:.2}ms ({:.1} FPS)", frame_time, fps);

            self.last_printed_instant = new_instant;
            self.frame_count = 0;
        }
    }
}

/// Stores all of the state of the example after initialization.
struct InitializedExample<E: Example> {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    window: Arc<Window>,
    surface: SurfaceWrapper,

    frame_counter: FrameCounter,
    example: E,
}

impl<E: Example> InitializedExample<E> {
    /// Asynchronously initialize the all the example state.
    async fn new(window: Arc<Window>) -> InitializedExample<E> {
        log::info!("Initializing wgpu...");

        let backends = wgpu::util::backend_bits_from_env().unwrap_or_default();
        let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
        let gles_minor_version = wgpu::util::gles_minor_version_from_env().unwrap_or_default();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
            dx12_shader_compiler,
            gles_minor_version,
        });

        let mut surface = SurfaceWrapper::new(&instance, window.clone());

        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, surface.get())
            .await
            .expect("No suitable GPU adapters found on the system!");

        let adapter_info = adapter.get_info();
        log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

        let optional_features = E::optional_features();
        let required_features = E::required_features();
        let adapter_features = adapter.features();
        assert!(
            adapter_features.contains(required_features),
            "Adapter does not support required features for this example: {:?}",
            required_features - adapter_features
        );

        let required_downlevel_capabilities = E::required_downlevel_capabilities();
        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        assert!(
            downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
            "Adapter does not support the minimum shader model required to run this example: {:?}",
            required_downlevel_capabilities.shader_model
        );
        assert!(
            downlevel_capabilities
                .flags
                .contains(required_downlevel_capabilities.flags),
            "Adapter does not support the downlevel capabilities required to run this example: {:?}",
            required_downlevel_capabilities.flags - downlevel_capabilities.flags
        );

        // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
        let needed_limits = E::required_limits().using_resolution(adapter.limits());

        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: (optional_features & adapter_features) | required_features,
                    required_limits: needed_limits,
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        surface.configure(&adapter, &device, &window, E::SRGB);

        let frame_counter = FrameCounter::new();

        let example = E::init(surface.config(), &adapter, &device, &queue);

        log::info!("Initialization complete.");

        InitializedExample {
            window,
            frame_counter,
            example,

            instance,
            adapter,
            device,
            queue,
            surface,
        }
    }
}

/// Event used to initialize the example state after initialization finishes.
struct StateInitEvent<E: Example>(InitializedExample<E>);

/// Initialization state for the example.
///
/// State should only ever move down the chain, from `Uninitialized` -> `Initializing` -> `Initialized`.
enum InitState<E: Example> {
    /// The example has not been initialized yet.
    Uninitialized,
    /// The example is being initialized asynchronously.
    Initializing,
    /// The example has been initialized.
    Initialized(InitializedExample<E>),
}

struct LoopState<E: Example> {
    state: InitState<E>,
    title: &'static str,
    event_loop_proxy: EventLoopProxy<StateInitEvent<E>>,
}

impl<E: Example> LoopState<E> {
    fn new(title: &'static str, event_loop: &EventLoop<StateInitEvent<E>>) -> LoopState<E> {
        LoopState {
            state: InitState::Uninitialized,
            title,
            event_loop_proxy: event_loop.create_proxy(),
        }
    }
}

impl<E: Example> ApplicationHandler<StateInitEvent<E>> for LoopState<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match self.state {
            InitState::Uninitialized => {
                // If we're uninitialized, start initializing and
                // continue to the main body.
                self.state = InitState::Initializing;
            }
            InitState::Initializing => {
                // If we're already initializing, just ignore the resume event.
                return;
            }
            InitState::Initialized(ref mut state) => {
                // If we're already initialized, resume the surface.
                state.surface.resume(
                    &state.instance,
                    &state.window,
                    &state.adapter,
                    &state.device,
                    E::SRGB,
                );
            }
        }

        log::info!("Initializing example...");

        // Configure and create the window.
        let mut window_attributes = Window::default_attributes().with_title(self.title);
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            window_attributes = window_attributes.with_canvas(Some(canvas));
        }
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        // Create the example future.
        let future = InitializedExample::new(window);

        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                // WASM we can't use block_on, so we spawn the future and send the result back to the event loop.

                let event_loop_proxy = self.event_loop_proxy.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let state = future.await;
                    event_loop_proxy.send_event(StateInitEvent(state)).unwrap_or_else(|_| {
                        panic!("Failed to send StateInitEvent");
                    });
                });
            } else {
                // On native, we can block on the future and send the result back directly.
                let state = pollster::block_on(future);
                self.event_loop_proxy.send_event(StateInitEvent(state)).unwrap_or_else(|_| {
                    panic!("Failed to send StateInitEvent");
                });
            }
        };
    }

    /// Handle the event where the example has been initialized.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: StateInitEvent<E>) {
        log::info!("Received initialized event.");

        let state = event.0;
        state.window.request_redraw();

        self.state = InitState::Initialized(state);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        let InitState::Initialized(ref mut state) = self.state else {
            return;
        };
        state.surface.suspend();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let InitState::Initialized(ref mut state) = self.state else {
            return;
        };

        match event {
            WindowEvent::Resized(size) => {
                state.surface.resize(&state.device, size);
                state
                    .example
                    .resize(state.surface.config(), &state.device, &state.queue);

                state.window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            }
            | WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(s),
                        ..
                    },
                ..
            } if s == "r" => {
                println!("{:#?}", state.instance.generate_report());
            }
            WindowEvent::RedrawRequested => {
                state.frame_counter.update();

                let frame = state.surface.acquire(&state.device);

                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(state.surface.config().view_formats[0]),
                    ..wgpu::TextureViewDescriptor::default()
                });

                state.example.render(&view, &state.device, &state.queue);

                frame.present();

                state.window.request_redraw();
            }
            _ => state.example.update(event),
        }
    }
}

pub fn run<E: Example>(title: &'static str) {
    init_logger();

    let event_loop = EventLoop::<StateInitEvent<E>>::with_user_event()
        .build()
        .unwrap();
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut loop_state: LoopState<E> = LoopState::new(title, &event_loop);

    log::info!("Entering event loop...");
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::EventLoopExtWebSys;

            event_loop.spawn_app(loop_state);
        } else {
            event_loop.run_app(&mut loop_state).unwrap();
        }
    }
}

#[cfg(target_arch = "wasm32")]
/// Parse the query string as returned by `web_sys::window()?.location().search()?` and get a
/// specific key out of it.
pub fn parse_url_query_string<'a>(query: &'a str, search_key: &str) -> Option<&'a str> {
    let query_string = query.strip_prefix('?')?;

    for pair in query_string.split('&') {
        let mut pair = pair.split('=');
        let key = pair.next()?;
        let value = pair.next()?;

        if key == search_key {
            return Some(value);
        }
    }

    None
}

#[cfg(test)]
pub use wgpu_test::image::ComparisonType;

#[cfg(test)]
#[derive(Clone)]
pub struct ExampleTestParams<E> {
    pub name: &'static str,
    // Path to the reference image, relative to the root of the repo.
    pub image_path: &'static str,
    pub width: u32,
    pub height: u32,
    pub optional_features: wgpu::Features,
    pub base_test_parameters: wgpu_test::TestParameters,
    /// Comparisons against FLIP statistics that determine if the test passes or fails.
    pub comparisons: &'static [ComparisonType],
    pub _phantom: std::marker::PhantomData<E>,
}

#[cfg(test)]
impl<E: Example + wgpu::WasmNotSendSync> From<ExampleTestParams<E>>
    for wgpu_test::GpuTestConfiguration
{
    fn from(params: ExampleTestParams<E>) -> Self {
        wgpu_test::GpuTestConfiguration::new()
            .name(params.name)
            .parameters({
                assert_eq!(params.width % 64, 0, "width needs to be aligned 64");

                let features = E::required_features() | params.optional_features;

                params.base_test_parameters.clone().features(features)
            })
            .run_async(move |ctx| async move {
                let format = if E::SRGB {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                };
                let dst_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("destination"),
                    size: wgpu::Extent3d {
                        width: params.width,
                        height: params.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });

                let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());

                let dst_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("image map buffer"),
                    size: params.width as u64 * params.height as u64 * 4,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut example = E::init(
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format,
                        width: params.width,
                        height: params.height,
                        desired_maximum_frame_latency: 2,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![format],
                    },
                    &ctx.adapter,
                    &ctx.device,
                    &ctx.queue,
                );

                example.render(&dst_view, &ctx.device, &ctx.queue);

                let mut cmd_buf = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                cmd_buf.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture: &dst_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &dst_buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(params.width * 4),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: params.width,
                        height: params.height,
                        depth_or_array_layers: 1,
                    },
                );

                ctx.queue.submit(Some(cmd_buf.finish()));

                let dst_buffer_slice = dst_buffer.slice(..);
                dst_buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
                ctx.async_poll(wgpu::Maintain::wait())
                    .await
                    .panic_on_timeout();
                let bytes = dst_buffer_slice.get_mapped_range().to_vec();

                wgpu_test::image::compare_image_output(
                    dbg!(env!("CARGO_MANIFEST_DIR").to_string() + "/../" + params.image_path),
                    &ctx.adapter_info,
                    params.width,
                    params.height,
                    &bytes,
                    params.comparisons,
                )
                .await;
            })
    }
}
