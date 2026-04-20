use std::{borrow::Cow, future::Future, sync::Arc};
use wgpu::CurrentSurfaceTexture;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    window::Window,
};

/// Runs a future to completion. On native this blocks synchronously via pollster.
/// On wasm this spawns a local task so control returns to the browser immediately.
#[cfg(not(target_arch = "wasm32"))]
fn spawn(f: impl Future<Output = ()> + 'static) {
    pollster::block_on(f);
}

/// Runs a future to completion. On native this blocks synchronously via pollster.
/// On wasm this spawns a local task so control returns to the browser immediately.
#[cfg(target_arch = "wasm32")]
fn spawn(f: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(f);
}

struct WgpuState {
    instance: wgpu::Instance,
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
}

enum TriangleAction {
    Initialized(WgpuState),
}

#[expect(clippy::large_enum_variant)]
enum AppState {
    Uninitialized,
    Loading,
    Running(WgpuState),
}

struct App {
    proxy: EventLoopProxy<TriangleAction>,
    window: Option<Arc<Window>>,
    state: AppState,
}

impl App {
    fn new(event_loop: &EventLoop<TriangleAction>) -> Self {
        Self {
            proxy: event_loop.create_proxy(),
            window: None,
            state: AppState::Uninitialized,
        }
    }
}

impl ApplicationHandler<TriangleAction> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.state, AppState::Uninitialized) {
            return;
        }
        self.state = AppState::Loading;

        #[cfg_attr(
            not(target_arch = "wasm32"),
            expect(unused_mut, reason = "wasm32 re-assigns to specify canvas")
        )]
        let mut attributes = Window::default_attributes();

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

        spawn(async move {
            let mut size = window.inner_size();
            size.width = size.width.max(1);
            size.height = size.height.max(1);

            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(
                    Box::new(display_handle),
                ));

            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    // Request an adapter which can render to our surface
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .expect("Failed to find an appropriate adapter");

            // Create the logical device and command queue
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    // Make sure we use the texture resolution limits from the adapter,
                    // so we can support images the size of the swapchain.
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("Failed to create device");

            // Load the shaders from disk
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                immediate_size: 0,
            });

            let swapchain_capabilities = surface.get_capabilities(&adapter);
            let swapchain_format = swapchain_capabilities.formats[0];

            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(swapchain_format.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            let config = surface
                .get_default_config(&adapter, size.width, size.height)
                .unwrap();
            surface.configure(&device, &config);

            let _ = proxy.send_event(TriangleAction::Initialized(WgpuState {
                instance,
                window,
                device,
                queue,
                surface,
                config,
                render_pipeline,
            }));
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: TriangleAction) {
        match event {
            TriangleAction::Initialized(wgpu_state) => {
                self.state = AppState::Running(wgpu_state);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let AppState::Running(wgpu_state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::Resized(new_size) => {
                // Reconfigure the surface with the new size
                wgpu_state.config.width = new_size.width.max(1);
                wgpu_state.config.height = new_size.height.max(1);
                wgpu_state
                    .surface
                    .configure(&wgpu_state.device, &wgpu_state.config);
                // On macos the window needs to be redrawn manually after resizing
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let frame = match wgpu_state.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(frame) => frame,
                    CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => {
                        // Try again later
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                    CurrentSurfaceTexture::Suboptimal(texture) => {
                        drop(texture);

                        wgpu_state
                            .surface
                            .configure(&wgpu_state.device, &wgpu_state.config);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                    CurrentSurfaceTexture::Outdated => {
                        wgpu_state
                            .surface
                            .configure(&wgpu_state.device, &wgpu_state.config);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                    CurrentSurfaceTexture::Validation => {
                        unreachable!("No error scope registered, so validation errors will panic")
                    }
                    CurrentSurfaceTexture::Lost => {
                        wgpu_state.surface = wgpu_state
                            .instance
                            .create_surface(wgpu_state.window.clone())
                            .unwrap();
                        wgpu_state
                            .surface
                            .configure(&wgpu_state.device, &wgpu_state.config);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = wgpu_state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    rpass.set_pipeline(&wgpu_state.render_pipeline);
                    rpass.draw(0..3, 0..1);
                }

                wgpu_state.queue.submit(Some(encoder.finish()));
                if let Some(window) = &self.window {
                    window.pre_present_notify();
                }
                wgpu_state.queue.present(frame);
            }
            WindowEvent::Occluded(is_occluded) => {
                if !is_occluded {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}

pub fn main() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("could not initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::with_user_event().build().unwrap();

    #[cfg_attr(target_arch = "wasm32", expect(unused_mut))]
    let mut app = App::new(&event_loop);

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::EventLoopExtWebSys;
            event_loop.spawn_app(app);
        } else {
            event_loop.run_app(&mut app).unwrap();
        }
    }
}
