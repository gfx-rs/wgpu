#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports))]

use std::{collections::HashMap, sync::Arc};
use wgpu::CurrentSurfaceTexture;
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

    fn get_current_texture(&mut self) -> CurrentSurfaceTexture {
        self.desc.surface.get_current_texture()
    }
}

const WINDOW_SIZE: u32 = 128;
const WINDOW_PADDING: u32 = 16;
const WINDOW_TITLEBAR: u32 = 32;
const WINDOW_OFFSET: u32 = WINDOW_SIZE + WINDOW_PADDING;
const ROWS: u32 = 4;
const COLUMNS: u32 = 4;

enum AppState {
    Uninitialized,
    Running {
        instance: wgpu::Instance,
        device: wgpu::Device,
        queue: wgpu::Queue,
        viewports: HashMap<WindowId, Viewport>,
    },
}

struct App {
    state: AppState,
}

impl App {
    fn new() -> Self {
        Self {
            state: AppState::Uninitialized,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.state, AppState::Uninitialized) {
            return;
        }

        // Create all 16 windows.
        let mut windows: Vec<(Arc<Window>, wgpu::Color)> =
            Vec::with_capacity((ROWS * COLUMNS) as usize);
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let window = Arc::new(
                    event_loop
                        .create_window(
                            Window::default_attributes()
                                .with_title(format!("x{column}y{row}"))
                                .with_inner_size(winit::dpi::PhysicalSize::new(
                                    WINDOW_SIZE,
                                    WINDOW_SIZE,
                                )),
                        )
                        .unwrap(),
                );
                window.set_outer_position(winit::dpi::PhysicalPosition::new(
                    WINDOW_PADDING + column * WINDOW_OFFSET,
                    WINDOW_PADDING + row * (WINDOW_OFFSET + WINDOW_TITLEBAR),
                ));
                fn frac(index: u32, max: u32) -> f64 {
                    index as f64 / max as f64
                }
                windows.push((
                    window,
                    wgpu::Color {
                        r: frac(row, ROWS),
                        g: 0.5 - frac(row * column, ROWS * COLUMNS) * 0.5,
                        b: frac(column, COLUMNS),
                        a: 1.0,
                    },
                ));
            }
        }

        // Initialize wgpu synchronously (native-only).
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(
                Box::new(event_loop.owned_display_handle()),
            ));
        let viewport_descs: Vec<_> = windows
            .into_iter()
            .map(|(window, color)| ViewportDesc::new(window, color, &instance))
            .collect();
        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: viewport_descs.first().map(|desc| &desc.surface),
                    ..Default::default()
                })
                .await
                .expect("Failed to find an appropriate adapter");

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("Failed to create device");

            (adapter, device, queue)
        });

        let viewports: HashMap<WindowId, Viewport> = viewport_descs
            .into_iter()
            .map(|desc| (desc.window.id(), desc.build(&adapter, &device)))
            .collect();

        self.state = AppState::Running {
            instance,
            device,
            queue,
            viewports,
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let AppState::Running {
            instance,
            device,
            queue,
            viewports,
        } = &mut self.state
        else {
            return;
        };

        match event {
            WindowEvent::Resized(new_size) => {
                // Recreate the swap chain with the new size
                if let Some(viewport) = viewports.get_mut(&window_id) {
                    viewport.resize(device, new_size);
                    // On macos the window needs to be redrawn manually after resizing
                    viewport.desc.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(viewport) = viewports.get_mut(&window_id) {
                    let frame = match viewport.get_current_texture() {
                        CurrentSurfaceTexture::Success(frame) => frame,
                        CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => {
                            viewport.desc.window.request_redraw();
                            return;
                        }
                        CurrentSurfaceTexture::Suboptimal(texture) => {
                            drop(texture);
                            viewport.desc.surface.configure(device, &viewport.config);
                            viewport.desc.window.request_redraw();
                            return;
                        }
                        CurrentSurfaceTexture::Outdated => {
                            viewport.desc.surface.configure(device, &viewport.config);
                            viewport.desc.window.request_redraw();
                            return;
                        }
                        CurrentSurfaceTexture::Validation => {
                            unreachable!(
                                "No error scope registered, so validation errors will panic"
                            )
                        }
                        CurrentSurfaceTexture::Lost => {
                            viewport.desc.surface = instance
                                .create_surface(viewport.desc.window.clone())
                                .unwrap();
                            viewport.desc.surface.configure(device, &viewport.config);
                            viewport.desc.window.request_redraw();
                            return;
                        }
                    };

                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(viewport.desc.background),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                    }

                    queue.submit(Some(encoder.finish()));
                    viewport.desc.window.pre_present_notify();
                    queue.present(frame);
                }
            }
            WindowEvent::Occluded(is_occluded) => {
                if !is_occluded {
                    if let Some(viewport) = viewports.get(&window_id) {
                        viewport.desc.window.request_redraw();
                    }
                }
            }
            WindowEvent::CloseRequested => {
                viewports.remove(&window_id);
                if viewports.is_empty() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let event_loop = winit::event_loop::EventLoop::new().unwrap();
        let mut app = App::new();
        event_loop.run_app(&mut app).unwrap();
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        panic!("wasm32 is not supported")
    }
}
