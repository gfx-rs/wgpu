#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use std::{collections::HashMap, sync::Arc};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

struct ViewportDesc {
    window: Arc<Window>,
    background: wgpu::Color,
    surface: wgpu::Surface<'static>,
}

struct Viewport {
    desc: ViewportDesc,
    config: wgpu::SurfaceConfiguration,
}

impl ViewportDesc {
    fn new(window: Arc<Window>, background: wgpu::Color, instance: &wgpu::Instance) -> Self {
        let surface = instance.create_surface(window.clone()).unwrap();
        Self {
            window,
            background,
            surface,
        }
    }

    fn build(self, adapter: &wgpu::Adapter, device: &wgpu::Device) -> Viewport {
        let size = self.window.inner_size();
        let config = self
            .surface
            .get_default_config(adapter, size.width, size.height)
            .unwrap();
        self.surface.configure(device, &config);
        Viewport { desc: self, config }
    }
}

impl Viewport {
    fn resize(&mut self, device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = size.width;
        self.config.height = size.height;
        self.desc.surface.configure(device, &self.config);
    }
    fn get_current_texture(&mut self) -> wgpu::SurfaceTexture {
        self.desc
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture")
    }
}

struct LoopState {
    // See https://docs.rs/winit/latest/winit/changelog/v0_30/index.html#removed
    // for the recommended practice regarding Window creation (from which everything depends)
    // in winit >= 0.30.0.
    // The actual state is in an Option because its initialization is now delayed to after
    // the even loop starts running.
    state: Option<InitializedLoopState>,
}

impl LoopState {
    fn new() -> LoopState {
        LoopState { state: None }
    }
}

struct InitializedLoopState {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    viewports: HashMap<WindowId, Viewport>,
}

impl InitializedLoopState {
    async fn new(event_loop: &ActiveEventLoop) -> InitializedLoopState {
        const WINDOW_SIZE: u32 = 128;
        const WINDOW_PADDING: u32 = 16;
        const WINDOW_TITLEBAR: u32 = 32;
        const WINDOW_OFFSET: u32 = WINDOW_SIZE + WINDOW_PADDING;
        const ROWS: u32 = 4;
        const COLUMNS: u32 = 4;

        let mut viewports = Vec::with_capacity((ROWS * COLUMNS) as usize);
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let window = event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title(format!("x{column}y{row}"))
                            .with_inner_size(winit::dpi::PhysicalSize::new(
                                WINDOW_SIZE,
                                WINDOW_SIZE,
                            )),
                    )
                    .unwrap();
                let window = Arc::new(window);
                window.set_outer_position(winit::dpi::PhysicalPosition::new(
                    WINDOW_PADDING + column * WINDOW_OFFSET,
                    WINDOW_PADDING + row * (WINDOW_OFFSET + WINDOW_TITLEBAR),
                ));
                fn frac(index: u32, max: u32) -> f64 {
                    index as f64 / max as f64
                }
                viewports.push((
                    window,
                    wgpu::Color {
                        r: frac(row, ROWS),
                        g: 0.5 - frac(row * column, ROWS * COLUMNS) * 0.5,
                        b: frac(column, COLUMNS),
                        a: 1.0,
                    },
                ))
            }
        }

        let instance = wgpu::Instance::default();
        let viewports: Vec<_> = viewports
            .into_iter()
            .map(|(window, color)| ViewportDesc::new(window, color, &instance))
            .collect();
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(
            &instance,
            viewports.first().map(|desc| &desc.surface),
        )
        .await
        .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let viewports: HashMap<WindowId, Viewport> = viewports
            .into_iter()
            .map(|desc| (desc.window.id(), desc.build(&adapter, &device)))
            .collect();

        InitializedLoopState {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            viewports,
        }
    }
}

impl ApplicationHandler for LoopState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            self.state = Some(pollster::block_on(InitializedLoopState::new(event_loop)));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            match event {
                WindowEvent::Resized(new_size) => {
                    // Recreate the swap chain with the new size
                    if let Some(viewport) = state.viewports.get_mut(&window_id) {
                        viewport.resize(&state.device, new_size);
                        // On macos the window needs to be redrawn manually after resizing
                        viewport.desc.window.request_redraw();
                    }
                }
                WindowEvent::RedrawRequested => {
                    if let Some(viewport) = state.viewports.get_mut(&window_id) {
                        let frame = viewport.get_current_texture();
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder =
                            state
                                .device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });
                        {
                            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(viewport.desc.background),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                        }

                        state.queue.submit(Some(encoder.finish()));
                        frame.present();
                    }
                }
                WindowEvent::CloseRequested => {
                    state.viewports.remove(&window_id);
                    if state.viewports.is_empty() {
                        event_loop.exit();
                    }
                }
                _ => {}
            }
        }
    }
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();

        let mut loop_state = LoopState::new();
        let event_loop = EventLoop::new().unwrap();

        log::info!("Entering event loop...");
        event_loop.run_app(&mut loop_state).unwrap();
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        panic!("wasm32 is not supported")
    }
}
