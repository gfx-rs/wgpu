use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

const ZOOM_INCREMENT_FACTOR: f32 = 1.1;
const CAMERA_POS_INCREMENT_FACTOR: f32 = 0.1;

#[derive(Debug)]
struct AppState {
    pub cursor_pos: [f32; 2],
    pub zoom: f32,
    pub max_iterations: f32,
}

impl AppState {
    fn translate_view(&mut self, increments: i32, axis: usize) {
        self.cursor_pos[axis] += CAMERA_POS_INCREMENT_FACTOR * increments as f32 / self.zoom;
    }

    fn zoom(&mut self, amount: f32) {
        self.zoom += ZOOM_INCREMENT_FACTOR * amount * self.zoom.powf(1.02);
        self.zoom = self.zoom.max(1.1);
    }
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            cursor_pos: [0.0, 0.0],
            zoom: 1.0,
            max_iterations: 50.0,
        }
    }
}

struct WgpuContext {
    pub window: Window,
    pub surface: wgpu::Surface,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
}

impl WgpuContext {
    async fn new(window: Window) -> WgpuContext {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shader.wgsl"
            ))),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<AppState>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        WgpuContext {
            window,
            surface,
            surface_config,
            device,
            queue,
            pipeline,
            bind_group,
            uniform_buffer,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.window.request_redraw();
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut wgpu_context = Some(WgpuContext::new(window).await);
    let mut state = Some(AppState::default());
    let main_window_id = wgpu_context.as_ref().unwrap().window.id();
    event_loop.run(move |event, _, control_flow| match event {
        Event::LoopDestroyed => {
            wgpu_context = None;
            state = None;
        }
        Event::WindowEvent { window_id, event } if window_id == main_window_id => match event {
            WindowEvent::CloseRequested => {
                control_flow.set_exit();
            }
            WindowEvent::KeyboardInput { input, .. } => {
                let state_mut = state.as_mut().unwrap();
                let wgpu_context_ref = wgpu_context.as_ref().unwrap();
                if let Some(virtual_keycode) = input.virtual_keycode {
                    match virtual_keycode {
                        VirtualKeyCode::Escape => control_flow.set_exit(),
                        VirtualKeyCode::Up => state_mut.translate_view(1, 1),
                        VirtualKeyCode::Down => state_mut.translate_view(-1, 1),
                        VirtualKeyCode::Left => state_mut.translate_view(-1, 0),
                        VirtualKeyCode::Right => state_mut.translate_view(1, 0),
                        VirtualKeyCode::U => state_mut.max_iterations += 3.0,
                        VirtualKeyCode::D => state_mut.max_iterations -= 3.0,
                        _ => {}
                    }
                    wgpu_context_ref.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let change = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, vertical) => vertical,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                };
                let state_mut = state.as_mut().unwrap();
                let wgpu_context_ref = wgpu_context.as_ref().unwrap();
                state_mut.zoom(change);
                wgpu_context_ref.window.request_redraw();
            }
            WindowEvent::Resized(new_size) => {
                let wgpu_context_mut = wgpu_context.as_mut().unwrap();
                wgpu_context_mut.resize(new_size);
                wgpu_context_mut.window.request_redraw();
            }
            _ => {}
        },
        Event::RedrawRequested(id) if id == main_window_id => {
            let wgpu_context_ref = wgpu_context.as_ref().unwrap();
            let state_ref = state.as_ref().unwrap();
            let frame = wgpu_context_ref.surface.get_current_texture().unwrap();
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            wgpu_context_ref.queue.write_buffer(
                &wgpu_context_ref.uniform_buffer,
                0,
                /* Ew gross I know. We have to do this to circumvent bytemuck's requirement
                that AppState implement POD (Plain Old Data). */
                bytemuck::bytes_of(&[
                    state_ref.cursor_pos[0],
                    state_ref.cursor_pos[1],
                    state_ref.zoom,
                    state_ref.max_iterations,
                ]),
            );
            let mut encoder = wgpu_context_ref
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });
                render_pass.set_pipeline(&wgpu_context_ref.pipeline);
                render_pass.set_bind_group(0, &wgpu_context_ref.bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
            wgpu_context_ref.queue.submit(Some(encoder.finish()));
            frame.present();
        }
        _ => {}
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
