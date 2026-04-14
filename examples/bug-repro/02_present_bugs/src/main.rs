//! Repro for various issues with queue presentation
//!
//! The 2 current bugs being tested are presentation after no usage of surface texture
//! and queue destruction immediately after present

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

struct State {
    window: Arc<Window>,
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: Option<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Presentation bugs"))
                .unwrap(),
        );
        self.state = Some(State::new(window));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.surface_config.width = size.width;
                state.surface_config.height = size.height;
                state
                    .surface
                    .configure(&state.device, &state.surface_config);
            }
            WindowEvent::RedrawRequested => {
                let frame = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f) => f,
                    wgpu::CurrentSurfaceTexture::Suboptimal(_)
                    | wgpu::CurrentSurfaceTexture::Outdated => {
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Lost => {
                        state.surface =
                            state.instance.create_surface(state.window.clone()).unwrap();
                        state
                            .surface
                            .configure(&state.device, &state.surface_config);
                        return;
                    }
                    _ => return,
                };
                let Some(queue) = state.queue.take() else {
                    return;
                };
                // Immediately present the surface texture (with nothing on it) and then drop the queue, which should cause a full wait.
                queue.present(frame);
                event_loop.exit();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

impl State {
    fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        instance_desc.flags |= wgpu::InstanceFlags::advanced_debugging();
        let instance = wgpu::Instance::new(instance_desc);
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No adapter");

        println!("Adapter: {:?}", adapter.get_info().name);

        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();

        let surface_format = surface.get_capabilities(&adapter).formats[0];
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        State {
            window,
            instance,
            device,
            queue: Some(queue),
            surface,
            surface_config,
        }
    }
}
