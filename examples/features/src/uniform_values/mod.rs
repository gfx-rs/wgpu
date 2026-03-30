//! Points of interest for seeing uniforms in action:
//!
//! 1. the struct for the data stored in the uniform buffer is defined.
//! 2. the uniform buffer itself is created.
//! 3. the bind group that will bind the uniform buffer and it's layout are created.
//! 4. the bind group layout is attached to the pipeline layout.
//! 5. the uniform buffer and the bind group are stored alongside the pipeline.
//! 6. an instance of `AppState` is created. This variable will be modified
//!    to change parameters in the shader and modified by app events to preform and save
//!    those changes.
//! 7. (7a and 7b) the `state` variable created at (6) is modified by commands such
//!    as pressing the arrow keys or zooming in or out.
//! 8. the contents of the `AppState` are loaded into the uniform buffer in preparation.
//! 9. the bind group with the uniform buffer is attached to the render pass.
//!
//! The usage of the uniform buffer within the shader itself is pretty self-explanatory given
//! some understanding of WGSL.

use std::{future::Future, sync::Arc};
// We won't bring StorageBuffer into scope as that might be too easy to confuse
// with actual GPU-allocated wgpu storage buffers.
use encase::ShaderType;
use wgpu::CurrentSurfaceTexture;
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{Key, NamedKey},
    window::Window,
};

const ZOOM_INCREMENT_FACTOR: f32 = 1.1;
const CAMERA_POS_INCREMENT_FACTOR: f32 = 0.1;

#[cfg(not(target_arch = "wasm32"))]
fn spawn(f: impl Future<Output = ()> + 'static) {
    pollster::block_on(f);
}

#[cfg(target_arch = "wasm32")]
fn spawn(f: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(f);
}

// (1)
#[derive(Debug, ShaderType)]
struct ShaderState {
    pub cursor_pos: glam::Vec2,
    pub zoom: f32,
    pub max_iterations: u32,
}

impl ShaderState {
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

impl Default for ShaderState {
    fn default() -> Self {
        ShaderState {
            cursor_pos: glam::Vec2::ZERO,
            zoom: 1.0,
            max_iterations: 50,
        }
    }
}

struct WgpuContext {
    pub instance: wgpu::Instance,
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
    async fn new(
        window: Arc<Window>,
        display_handle: winit::event_loop::OwnedDisplayHandle,
    ) -> WgpuContext {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display_handle)),
        );
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
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
            .unwrap();

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // (2)
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<ShaderState>() as u64,
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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
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
        let surface_config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &surface_config);

        // (5)
        WgpuContext {
            instance,
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
    }
}

enum UniformAction {
    Initialized(WgpuContext),
}

#[expect(clippy::large_enum_variant)]
enum RunState {
    Uninitialized,
    Loading,
    Running {
        wgpu_ctx: WgpuContext,
        // (6)
        shader_state: ShaderState,
    },
}

struct App {
    proxy: EventLoopProxy<UniformAction>,
    window: Option<Arc<Window>>,
    state: RunState,
}

impl App {
    fn new(event_loop: &EventLoop<UniformAction>) -> Self {
        Self {
            proxy: event_loop.create_proxy(),
            window: None,
            state: RunState::Uninitialized,
        }
    }
}

impl ApplicationHandler<UniformAction> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.state, RunState::Uninitialized) {
            return;
        }
        self.state = RunState::Loading;

        #[cfg_attr(
            not(target_arch = "wasm32"),
            expect(unused_mut, reason = "wasm32 re-assigns to specify canvas")
        )]
        let mut attributes = Window::default_attributes()
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
            let wgpu_ctx = WgpuContext::new(window, display_handle).await;
            let _ = proxy.send_event(UniformAction::Initialized(wgpu_ctx));
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UniformAction) {
        match event {
            UniformAction::Initialized(wgpu_ctx) => {
                self.state = RunState::Running {
                    wgpu_ctx,
                    shader_state: ShaderState::default(),
                };
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.state = RunState::Uninitialized;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let RunState::Running {
            wgpu_ctx,
            shader_state,
        } = &mut self.state
        else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    logical_key, text, ..
                },
                ..
            } => {
                if let Key::Named(key) = logical_key {
                    match key {
                        NamedKey::Escape => event_loop.exit(),
                        NamedKey::ArrowUp => shader_state.translate_view(1, 1),
                        NamedKey::ArrowDown => shader_state.translate_view(-1, 1),
                        NamedKey::ArrowLeft => shader_state.translate_view(-1, 0),
                        NamedKey::ArrowRight => shader_state.translate_view(1, 0),
                        _ => {}
                    }
                }

                if let Some(text) = text {
                    if text == "u" {
                        shader_state.max_iterations += 3;
                    } else if text == "d" {
                        shader_state.max_iterations = shader_state.max_iterations.saturating_sub(3);
                    }
                };

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let change = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, vertical) => vertical,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                };
                // (7b)
                shader_state.zoom(change);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(new_size) => {
                wgpu_ctx.resize(new_size);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let frame = match wgpu_ctx.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(frame) => frame,
                    CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => {
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                    CurrentSurfaceTexture::Suboptimal(_) | CurrentSurfaceTexture::Outdated => {
                        wgpu_ctx
                            .surface
                            .configure(&wgpu_ctx.device, &wgpu_ctx.surface_config);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                    CurrentSurfaceTexture::Validation => {
                        unreachable!("No error scope registered, so validation errors will panic")
                    }
                    CurrentSurfaceTexture::Lost => {
                        wgpu_ctx.surface = wgpu_ctx
                            .instance
                            .create_surface(wgpu_ctx.window.clone())
                            .unwrap();
                        wgpu_ctx
                            .surface
                            .configure(&wgpu_ctx.device, &wgpu_ctx.surface_config);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // (8)
                wgpu_ctx.queue.write_buffer(
                    &wgpu_ctx.uniform_buffer,
                    0,
                    &shader_state.as_wgsl_bytes().expect(
                        "Error in encase translating ShaderState \
                    struct to WGSL bytes.",
                    ),
                );
                let mut encoder = wgpu_ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                        occlusion_query_set: None,
                        timestamp_writes: None,
                        multiview_mask: None,
                    });
                    render_pass.set_pipeline(&wgpu_ctx.pipeline);
                    // (9)
                    render_pass.set_bind_group(0, Some(&wgpu_ctx.bind_group), &[]);
                    render_pass.draw(0..3, 0..1);
                }
                wgpu_ctx.queue.submit(Some(encoder.finish()));
                if let Some(window) = &self.window {
                    window.pre_present_notify();
                }
                frame.present();
            }
            WindowEvent::Occluded(is_occluded) => {
                if !is_occluded {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
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
            env_logger::builder().format_timestamp_nanos().init();
        }
    }

    let event_loop = EventLoop::with_user_event().build().unwrap();

    #[cfg_attr(target_arch = "wasm32", expect(unused_mut))]
    let mut app = App::new(&event_loop);

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::EventLoopExtWebSys;

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

            event_loop.spawn_app(app);
        } else {
            event_loop.run_app(&mut app).unwrap();
        }
    }
}
