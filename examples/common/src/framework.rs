use web_time::Instant;
use wgpu::{Instance, Surface, WasmNotSend, WasmNotSync};
use wgpu_test::GpuTestConfiguration;
use winit::{
    dpi::PhysicalSize,
    event::{Event, KeyEvent, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    keyboard::{Key, NamedKey},
    window::Window,
};

#[allow(dead_code)]
pub fn cast_slice<T>(data: &[T]) -> &[u8] {
    use std::{mem::size_of_val, slice::from_raw_parts};

    unsafe { from_raw_parts(data.as_ptr() as *const u8, size_of_val(data)) }
}

#[allow(dead_code)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

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

fn init_logger() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let query_string = web_sys::window().unwrap().location().search().unwrap();
            let level: log::Level = parse_url_query_string(&query_string, "RUST_LOG")
                .and_then(|x| x.parse().ok())
                .unwrap_or(log::Level::Error);
            console_log::init_with_level(level).expect("could not initialize logger");
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        } else {
            env_logger::init();
        }
    }
}

struct WindowLoop {
    event_loop: EventLoop<()>,
    window: Window,
}

fn init_event_loop(title: &str) -> WindowLoop {
    let event_loop = EventLoop::new().unwrap();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title(title);
    let window = builder.build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let canvas = window.canvas().expect("Couldn't get canvas");
        canvas.style().set_css_text("height: 500px; width: 500px;");
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| body.append_child(&canvas).ok())
            .expect("couldn't append canvas to document body");
    }

    WindowLoop { event_loop, window }
}

struct SurfaceContainer {
    surface: Option<wgpu::Surface>,
    config: Option<wgpu::SurfaceConfiguration>,
}

impl SurfaceContainer {
    fn new() -> Self {
        Self {
            surface: None,
            config: None,
        }
    }

    fn pre_adapter(&mut self, instance: &Instance, window: &Window) {
        if cfg!(target_arch = "wasm32") {
            self.surface = Some(unsafe { instance.create_surface(&window).unwrap() });
        }
    }

    fn resume(&mut self, context: &ExampleContext, window: &Window, srgb: bool) {
        if !cfg!(target_arch = "wasm32") {
            self.surface = Some(unsafe { context.instance.create_surface(&window).unwrap() });
        }

        let surface = self.surface.as_ref().unwrap();

        let window_size = window.inner_size();

        let config = self.config.insert(
            surface
                .get_default_config(&context.adapter, window_size.width, window_size.height)
                .expect("Surface isn't supported by the adapter."),
        );
        let surface_view_format = if srgb {
            config.format.add_srgb_suffix()
        } else {
            config.format.remove_srgb_suffix()
        };
        config.view_formats.push(surface_view_format);

        surface.configure(&context.device, &config);
    }

    fn resize(&mut self, context: &ExampleContext, size: PhysicalSize<u32>) {
        let config = self.config.as_mut().unwrap();
        config.width = size.width.max(1);
        config.height = size.height.max(1);
        let surface = self.surface.as_ref().unwrap();
        surface.configure(&context.device, config);
    }

    fn acquire(&mut self, context: &ExampleContext) -> wgpu::SurfaceTexture {
        let surface = self.get().unwrap();

        match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                surface.configure(&context.device, self.config());
                surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            }
        }
    }

    fn suspend(&mut self) {
        if !cfg!(target_arch = "wasm32") {
            self.surface = None;
        }
    }

    fn get(&self) -> Option<&Surface> {
        self.surface.as_ref()
    }

    fn config(&self) -> &wgpu::SurfaceConfiguration {
        self.config.as_ref().unwrap()
    }
}

struct ExampleContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl ExampleContext {
    async fn init_async<E: Example>(surface: &mut SurfaceContainer, window: &Window) -> Self {
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
        surface.pre_adapter(&instance, window);
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
                    features: (optional_features & adapter_features) | required_features,
                    limits: needed_limits,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
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

async fn start<E: Example>(title: &str) {
    init_logger();
    let window_loop = init_event_loop(title);
    let mut surface = SurfaceContainer::new();
    let context = ExampleContext::init_async::<E>(&mut surface, &window_loop.window).await;

    let mut last_frame_inst = web_time::Instant::now();
    let (mut frame_count, mut accum_time) = (0, 0.0);

    let mut example = None;

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::EventLoopExtWebSys;
            let event_loop_function = EventLoop::spawn;
        } else {
            let event_loop_function = EventLoop::run;
        }
    }

    log::info!("Entering event loop...");
    let _ = (event_loop_function)(
        window_loop.event_loop,
        move |event: Event<()>, target: &EventLoopWindowTarget<()>| {
            target.set_control_flow(ControlFlow::Poll);

            fn start_condition(e: &Event<()>) -> bool {
                match e {
                    Event::NewEvents(StartCause::Init) => !cfg!(target_os = "android"),
                    Event::Resumed => cfg!(target_os = "android"),
                    _ => false,
                }
            }

            match event {
                ref e if start_condition(e) => {
                    log::error!("Surface resume");
                    surface.resume(&context, &window_loop.window, E::SRGB);

                    example = Some(E::init(
                        surface.config(),
                        &context.adapter,
                        &context.device,
                        &context.queue,
                    ));
                }
                Event::Suspended => {
                    log::error!("Surface suspend");
                    surface.suspend();
                    example = None;
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(size) => {
                        log::error!("Surface resize {size:?}");
                        surface.resize(&context, size);
                        example.as_mut().unwrap().resize(
                            &surface.config(),
                            &context.device,
                            &context.queue,
                        );

                        window_loop.window.request_redraw();
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
                        target.exit();
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
                        accum_time += last_frame_inst.elapsed().as_secs_f32();
                        last_frame_inst = Instant::now();
                        frame_count += 1;
                        if frame_count == 100 {
                            println!(
                                "Avg frame time {}ms",
                                accum_time * 1000.0 / frame_count as f32
                            );
                            accum_time = 0.0;
                            frame_count = 0;
                        }

                        let frame = surface.acquire(&context);
                        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                            format: Some(surface.config().view_formats[0]),
                            ..wgpu::TextureViewDescriptor::default()
                        });

                        example
                            .as_mut()
                            .unwrap()
                            .render(&view, &context.device, &context.queue);

                        frame.present();

                        window_loop.window.request_redraw();
                    }
                    _ => example.as_mut().unwrap().update(event),
                },
                _ => {}
            }
        },
    );
}

pub fn run<E: Example>(title: &'static str) {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            wasm_bindgen_futures::spawn_local(async move { start::<E>(title).await })
        } else {
            pollster::block_on(start::<E>(title));
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

pub use wgpu_test::image::ComparisonType;

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

impl<E: Example + WasmNotSend + WasmNotSync> From<ExampleTestParams<E>> for GpuTestConfiguration {
    fn from(params: ExampleTestParams<E>) -> Self {
        GpuTestConfiguration::new()
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
                ctx.device.poll(wgpu::Maintain::Wait);
                let bytes = dst_buffer_slice.get_mapped_range().to_vec();

                wgpu_test::image::compare_image_output(
                    env!("CARGO_MANIFEST_DIR").to_string() + "/../../" + params.image_path,
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
