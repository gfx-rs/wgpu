use std::future::Future;
use std::sync::Arc;

use wgpu::{Instance, Surface};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{Key, NamedKey},
    window::Window,
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

            let base_level = query_level.unwrap_or(log::LevelFilter::Info);

            // On web, we use fern, as console_log doesn't have filtering on a per-module level.
            fern::Dispatch::new()
                .level(base_level)
                .chain(fern::Output::call(console_log::log))
                .apply()
                .unwrap();
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        } else {
            // parse_default_env will read the RUST_LOG environment variable and apply it on top
            // of these default filters.
            env_logger::builder()
                .filter_level(log::LevelFilter::Info)
                .parse_default_env()
                .init();
        }
    }
}

/// Runs a future to completion. On native this blocks via pollster, on wasm this spawns
/// a local task. This allows the same async wgpu initialization code to work on both platforms.
#[cfg(not(target_arch = "wasm32"))]
fn spawn(f: impl Future<Output = ()> + 'static) {
    pollster::block_on(f);
}

/// Runs a future to completion. On native this blocks via pollster, on wasm this spawns
/// a local task. This allows the same async wgpu initialization code to work on both platforms.
#[cfg(target_arch = "wasm32")]
fn spawn(f: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(f);
}

/// Wrapper type which manages the surface and surface configuration.
///
/// As surface usage varies per platform, wrapping this up cleans up the event loop code.
struct SurfaceWrapper {
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
}

impl SurfaceWrapper {
    /// Create a new surface wrapper with no surface or configuration.
    fn new() -> Self {
        Self {
            surface: None,
            config: None,
        }
    }

    /// Called after the instance is created, but before we request an adapter.
    ///
    /// On wasm, we need to create the surface here, as the WebGL backend needs
    /// a surface (and hence a canvas) to be present to create the adapter.
    ///
    /// We cannot unconditionally create a surface here, as Android requires
    /// us to wait until we receive the `Resumed` event to do so.
    fn pre_adapter(&mut self, instance: &Instance, window: Arc<Window>) {
        if cfg!(target_arch = "wasm32") {
            self.surface = Some(instance.create_surface(window).unwrap());
        }
    }

    /// Called on resume to create (on native) and configure the surface.
    ///
    /// On all native platforms, this is where we create the surface.
    /// On wasm, the surface was already created in [`Self::pre_adapter`].
    ///
    /// Additionally, we configure the surface based on the (now valid) window size.
    fn resume(&mut self, context: &ExampleContext, window: Arc<Window>, srgb: bool) {
        // Window size is only actually valid after we enter the event loop.
        let window_size = window.inner_size();
        let width = window_size.width.max(1);
        let height = window_size.height.max(1);

        log::info!("Surface resume {window_size:?}");

        // We didn't create the surface in pre_adapter, so we need to do so now.
        if !cfg!(target_arch = "wasm32") {
            self.surface = Some(context.instance.create_surface(window).unwrap());
        }

        // From here on, self.surface should be Some.

        let surface = self.surface.as_ref().unwrap();

        // Get the default configuration,
        let mut config = surface
            .get_default_config(&context.adapter, width, height)
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
        config.desired_maximum_frame_latency = 3;

        surface.configure(&context.device, &config);
        self.config = Some(config);
    }

    /// Resize the surface, making sure to not resize to zero.
    fn resize(&mut self, context: &ExampleContext, size: PhysicalSize<u32>) {
        log::info!("Surface resize {size:?}");

        let config = self.config.as_mut().unwrap();
        config.width = size.width.max(1);
        config.height = size.height.max(1);
        let surface = self.surface.as_ref().unwrap();
        surface.configure(&context.device, config);
    }

    /// Acquire the next surface texture.
    ///
    /// Returns `None` on failure.
    fn acquire(&mut self, context: &ExampleContext) -> Option<wgpu::SurfaceTexture> {
        let surface = self.surface.as_ref().unwrap();

        match surface.get_current_texture() {
            Ok(frame) => Some(frame),
            // If we timed out or the window is occluded, skip this frame:
            Err(wgpu::SurfaceError::Timeout | wgpu::SurfaceError::Occluded) => None,
            Err(
                // If the surface is outdated, or was lost, reconfigure it.
                wgpu::SurfaceError::Outdated
                | wgpu::SurfaceError::Lost
                | wgpu::SurfaceError::Other
                // If OutOfMemory happens, reconfiguring may not help, but we might as well try
                | wgpu::SurfaceError::OutOfMemory,
            ) => {
                surface.configure(&context.device, self.config());
                Some(surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!"))
            }
        }
    }

    /// On suspend on android, we drop the surface, as it's no longer valid.
    ///
    /// A suspend event is always followed by at least one resume event.
    fn suspend(&mut self) {
        if cfg!(target_os = "android") {
            self.surface = None;
        }
    }

    fn get(&self) -> Option<&'_ Surface<'static>> {
        self.surface.as_ref()
    }

    fn config(&self) -> &wgpu::SurfaceConfiguration {
        self.config.as_ref().unwrap()
    }
}

/// Context containing global wgpu resources.
struct ExampleContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl ExampleContext {
    /// Initializes the example context.
    async fn init_async<E: Example>(
        surface: &mut SurfaceWrapper,
        window: Arc<Window>,
        display_handle: winit::event_loop::OwnedDisplayHandle,
    ) -> Self {
        log::info!("Initializing wgpu...");

        let instance_descriptor =
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display_handle));
        let instance = wgpu::Instance::new(instance_descriptor);
        surface.pre_adapter(&instance, window);
        let adapter = get_adapter_with_capabilities_or_from_env(
            &instance,
            &E::required_features(),
            &E::required_downlevel_capabilities(),
            &surface.get(),
        )
        .await;
        // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
        let needed_limits = E::required_limits().using_resolution(adapter.limits());

        let info = adapter.get_info();
        log::info!("Selected adapter: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: (E::optional_features() & adapter.features())
                    | E::required_features(),
                required_limits: needed_limits,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: match std::env::var_os("WGPU_TRACE") {
                    Some(path) => wgpu::Trace::Directory(path.into()),
                    None => wgpu::Trace::Off,
                },
            })
            .await
            .expect("Unable to find a suitable GPU adapter!");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
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
            log::info!("Frame time {frame_time:.2}ms ({fps:.1} FPS)");

            self.last_printed_instant = new_instant;
            self.frame_count = 0;
        }
    }
}

/// User event sent via [`EventLoopProxy`] to deliver async initialization results
/// back to the main event loop.
enum AppAction {
    /// The async wgpu initialization has completed.
    WgpuInitialized {
        context: ExampleContext,
        surface: SurfaceWrapper,
    },
}

#[expect(clippy::large_enum_variant)]
enum AppState<E> {
    /// Waiting for the first `resumed()` call.
    Uninitialized,
    /// Window created, async wgpu initialization in progress.
    Loading,
    /// Fully initialized and rendering.
    Running {
        context: ExampleContext,
        surface: SurfaceWrapper,
        example: E,
    },
}

/// The main application struct, implementing winit's [`ApplicationHandler`].
///
/// Winit 0.30 requires that windows are not created until the `resumed()` callback,
/// and that all wgpu resources (instance, adapter, device) are initialized after the
/// window exists. On native, this init happens synchronously via `pollster::block_on`.
/// On wasm, it happens asynchronously via `wasm_bindgen_futures::spawn_local`, with
/// the results delivered back through an [`EventLoopProxy`] user event.
struct App<E: Example> {
    title: &'static str,
    proxy: EventLoopProxy<AppAction>,
    window: Option<Arc<Window>>,
    frame_counter: FrameCounter,
    occluded: bool,
    state: AppState<E>,
}

impl<E: Example> App<E> {
    fn new(title: &'static str, event_loop: &EventLoop<AppAction>) -> Self {
        Self {
            title,
            proxy: event_loop.create_proxy(),
            window: None,
            frame_counter: FrameCounter::new(),
            occluded: false,
            state: AppState::Uninitialized,
        }
    }
}

impl<E: Example> ApplicationHandler<AppAction> for App<E> {
    /// Called when the application is (re)started. On the first call, the window and wgpu
    /// resources are created. On Android, this may be called again after each suspend —
    /// in that case we only need to re-create the surface.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // On Android, re-create the surface after a suspend/resume cycle.
        if let AppState::Running {
            ref context,
            ref mut surface,
            ..
        } = self.state
        {
            if let Some(window) = &self.window {
                surface.resume(context, window.clone(), E::SRGB);
                window.request_redraw();
            }
            return;
        }

        if !matches!(self.state, AppState::Uninitialized) {
            return;
        }
        self.state = AppState::Loading;

        #[cfg_attr(
            not(target_arch = "wasm32"),
            expect(unused_mut, reason = "wasm32 re-assigns to specify canvas")
        )]
        let mut attributes = Window::default_attributes().with_title(self.title);

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
            attributes = attributes.with_canvas(Some(canvas));
        }

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("Failed to create window"),
        );
        self.window = Some(window.clone());

        let display_handle = event_loop.owned_display_handle();
        let proxy = self.proxy.clone();

        // Spawn the async wgpu initialization. On native, `spawn` uses `pollster::block_on`
        // so this completes synchronously before `resumed()` returns. On wasm, `spawn` uses
        // `wasm_bindgen_futures::spawn_local` so the result arrives later via `user_event()`.
        spawn(async move {
            let mut surface = SurfaceWrapper::new();
            let context =
                ExampleContext::init_async::<E>(&mut surface, window.clone(), display_handle).await;
            surface.resume(&context, window, E::SRGB);
            let _ = proxy.send_event(AppAction::WgpuInitialized { context, surface });
        });
    }

    /// Receives the result of the async wgpu initialization. Creates the [`Example`] and
    /// transitions to the running state.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppAction) {
        match event {
            AppAction::WgpuInitialized { context, surface } => {
                let example = E::init(
                    surface.config(),
                    &context.adapter,
                    &context.device,
                    &context.queue,
                );

                self.state = AppState::Running {
                    context,
                    surface,
                    example,
                };

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let AppState::Running { surface, .. } = &mut self.state {
            surface.suspend();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let AppState::Running {
            ref mut context,
            ref mut surface,
            ref mut example,
        } = self.state
        else {
            return;
        };

        match event {
            WindowEvent::Resized(size) => {
                surface.resize(context, size);
                example.resize(surface.config(), &context.device, &context.queue);

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
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
                println!("{:#?}", context.instance.generate_report());
            }
            WindowEvent::RedrawRequested => {
                // Don't render while occluded, this may leak on apple platforms.
                if self.occluded {
                    return;
                }

                self.frame_counter.update();

                if let Some(frame) = surface.acquire(context) {
                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(surface.config().view_formats[0]),
                        ..wgpu::TextureViewDescriptor::default()
                    });

                    example.render(&view, &context.device, &context.queue);

                    if let Some(window) = &self.window {
                        window.pre_present_notify();
                    }
                    frame.present();
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Occluded(is_occluded) => {
                self.occluded = is_occluded;
                // Resume rendering when un-occluded.
                if !is_occluded {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            _ => example.update(event),
        }
    }
}

fn start<E: Example>(title: &'static str) {
    init_logger();

    log::debug!(
        "Enabled backends: {:?}",
        wgpu::Instance::enabled_backend_features()
    );

    let event_loop = EventLoop::with_user_event().build().unwrap();

    #[cfg_attr(target_arch = "wasm32", expect(unused_mut))]
    let mut app = App::<E>::new(title, &event_loop);

    log::info!("Entering event loop...");
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::EventLoopExtWebSys;
            event_loop.spawn_app(app);
        } else {
            event_loop.run_app(&mut app).unwrap();
        }
    }
}

pub fn run<E: Example>(title: &'static str) {
    start::<E>(title);
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

use crate::utils::get_adapter_with_capabilities_or_from_env;

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

                params
                    .base_test_parameters
                    .clone()
                    .features(features)
                    .limits(E::required_limits())
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
                    wgpu::TexelCopyTextureInfo {
                        texture: &dst_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &dst_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
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
                ctx.async_poll(wgpu::PollType::wait_indefinitely())
                    .await
                    .unwrap();
                let bytes = dst_buffer_slice.get_mapped_range().to_vec();

                wgpu_test::image::compare_image_output(
                    dbg!(env!("CARGO_MANIFEST_DIR").to_string() + "/../../" + params.image_path),
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
