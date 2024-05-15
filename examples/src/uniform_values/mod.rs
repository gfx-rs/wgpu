//! Points of interest for seeing uniforms in action:
//!
//! 1. the struct for the data stored in the uniform buffer is defined.
//! 2. the uniform buffer itself is created.
//! 3. the bind group that will bind the uniform buffer and it's layout are created.
//! 4. the bind group layout is attached to the pipeline layout.
//! 5. the uniform buffer and the bind group are stored alongside the pipeline.
//! 6. an instance of `AppState` is created. This variable will be modified
//! to change parameters in the shader and modified by app events to preform and save
//! those changes.
//! 7. (7a and 7b) the `state` variable created at (6) is modified by commands such
//! as pressing the arrow keys or zooming in or out.
//! 8. the contents of the `AppState` are loaded into the uniform buffer in preparation.
//! 9. the bind group with the uniform buffer is attached to the render pass.
//!
//! The usage of the uniform buffer within the shader itself is pretty self-explanatory given
//! some understanding of WGSL.

use std::sync::Arc;
// We won't bring StorageBuffer into scope as that might be too easy to confuse
// with actual GPU-allocated WGPU storage buffers.
use encase::ShaderType;
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const ZOOM_INCREMENT_FACTOR: f32 = 1.1;
const CAMERA_POS_INCREMENT_FACTOR: f32 = 0.1;

// (1)
#[derive(Debug, ShaderType)]
struct AppState {
    pub cursor_pos: glam::Vec2,
    pub zoom: f32,
    pub max_iterations: u32,
}

impl AppState {
    // Translating Rust structures to WGSL is always tricky and can prove
    // incredibly difficult to remember all the rules by which WGSL
    // lays out and formats structs in memory. It is also often extremely
    // frustrating to debug when things don't go right.
    //
    // You may sometimes see structs translated to bytes through
    // using `#[repr(C)]` on the struct so that the struct has a defined,
    // guaranteed internal layout and then implementing bytemuck's POD
    // trait so that one can preform a bitwise cast. There are issues with
    // this approach though as C's struct layouts aren't always compatible
    // with WGSL, such as when special WGSL types like vec's and mat's
    // get involved that have special alignment rules and especially
    // when the target buffer is going to be used in the uniform memory
    // space.
    //
    // Here though, we use the encase crate which makes translating potentially
    // complex Rust structs easy through combined use of the [`ShaderType`] trait
    // / derive macro and the buffer structs which hold data formatted for WGSL
    // in either the storage or uniform spaces.
    fn as_wgsl_bytes(&self) -> encase::internal::Result<Vec<u8>> {
        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(self)?;
        Ok(buffer.into_inner())
    }

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
            cursor_pos: glam::Vec2::ZERO,
            zoom: 1.0,
            max_iterations: 50,
        }
    }
}

struct WgpuContext {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
}

impl WgpuContext {
    async fn new(window: Arc<Window>) -> WgpuContext {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
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
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
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

        // (2)
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<AppState>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // (3)
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
            // (4)
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
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let surface_config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &surface_config);

        // (5)
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

struct LoopState {
    // See https://docs.rs/winit/latest/winit/changelog/v0_30/index.html#removed
    // for the recommended pratcice regarding Window creation (from which everything depends)
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
    wgpu_context: WgpuContext,
    app_state: AppState,
    main_window_id: WindowId,
}

impl InitializedLoopState {
    async fn new(event_loop: &ActiveEventLoop) -> InitializedLoopState {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes()
            .with_title("Remember: Use U/D to change sample count!")
            .with_inner_size(winit::dpi::LogicalSize::new(900, 900));
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

        let wgpu_context = WgpuContext::new(window).await;
        // (6)
        let app_state = AppState::default();
        let main_window_id = wgpu_context.window.id();

        InitializedLoopState {
            wgpu_context,
            app_state,
            main_window_id,
        }
    }
}

impl ApplicationHandler for LoopState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.state = Some(pollster::block_on(InitializedLoopState::new(event_loop)));
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.state = Some(wasm_bindgen_futures::spawn_local(async move {
                    InitializedLoopState::new(event_loop).await
                }));
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            if window_id != state.main_window_id {
                return;
            }
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            logical_key, text, ..
                        },
                    ..
                } => {
                    if let Key::Named(key) = logical_key {
                        match key {
                            NamedKey::Escape => event_loop.exit(),
                            NamedKey::ArrowUp => state.app_state.translate_view(1, 1),
                            NamedKey::ArrowDown => state.app_state.translate_view(-1, 1),
                            NamedKey::ArrowLeft => state.app_state.translate_view(-1, 0),
                            NamedKey::ArrowRight => state.app_state.translate_view(1, 0),
                            _ => {}
                        }
                    }

                    if let Some(text) = text {
                        if text == "u" {
                            state.app_state.max_iterations += 3;
                        } else if text == "d" {
                            state.app_state.max_iterations -= 3;
                        }
                    };

                    state.wgpu_context.window.request_redraw();
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let change = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, vertical) => vertical,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                    };
                    // (7b)
                    state.app_state.zoom(change);
                    state.wgpu_context.window.request_redraw();
                }
                WindowEvent::Resized(new_size) => {
                    state.wgpu_context.resize(new_size);
                    state.wgpu_context.window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    let frame = state.wgpu_context.surface.get_current_texture().unwrap();
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // (8)
                    state.wgpu_context.queue.write_buffer(
                        &state.wgpu_context.uniform_buffer,
                        0,
                        &state.app_state.as_wgsl_bytes().expect(
                            "Error in encase translating AppState \
				struct to WGSL bytes.",
                        ),
                    );
                    let mut encoder = state
                        .wgpu_context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                occlusion_query_set: None,
                                timestamp_writes: None,
                            });
                        render_pass.set_pipeline(&state.wgpu_context.pipeline);
                        // (9)
                        render_pass.set_bind_group(0, &state.wgpu_context.bind_group, &[]);
                        render_pass.draw(0..3, 0..1);
                    }
                    state.wgpu_context.queue.submit(Some(encoder.finish()));
                    frame.present();
                }
                _ => {}
            }
        }
    }
}

async fn start() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder().format_timestamp_nanos().init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");

        let document = web_sys::window()
            .and_then(|win| win.document())
            .expect("Failed to get document.");
        let body = document.body().unwrap();
        let controls_text = document
            .create_element("p")
            .expect("Failed to create controls text as element.");
        controls_text.set_inner_html(
            "Controls: <br/>
Up, Down, Left, Right: Move view, <br/>
Scroll: Zoom, <br/>
U, D: Increase / decrease sample count.",
        );
        body.append_child(&controls_text)
            .expect("Failed to append controls text to body.");
    }

    let mut loop_state = LoopState::new();
    let event_loop = EventLoop::new().unwrap();

    log::info!("Entering event loop...");
    event_loop.run_app(&mut loop_state).unwrap();
}

pub fn main() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            wasm_bindgen_futures::spawn_local(async move { start::<E>(title).await })
        } else {
            pollster::block_on(start());
        }
    }
}
