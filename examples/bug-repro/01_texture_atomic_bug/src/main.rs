//! Repro for Metal driver bug where fragment shader texture atomic writes randomly drop unless
//! a compute pass with atomic access to the texture is inserted between the write and the read.
//! Both 32-bit and 64-bit atomic textures are affected.
//! The bug does not reproduce with `MTL_SHADER_VALIDATION=1`.
//! Known to reproduce on Apple M4 Max, macOS 26.3 (Tahoe).
//! Dropped writes appear as various tile-shaped black holes that flicker around each frame.

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
    instance: wgpu::Instance,
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    width: u32,
    height: u32,
    _storage_texture: wgpu::Texture,
    _dummy_texture: wgpu::Texture,
    clear_pipeline: wgpu::ComputePipeline,
    clear_bg: wgpu::BindGroup,
    raster_pipeline: wgpu::RenderPipeline,
    raster_bg: wgpu::BindGroup,
    dummy_view: wgpu::TextureView,
    vis_pipeline: wgpu::RenderPipeline,
    vis_bg: wgpu::BindGroup,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Metal Texture Atomic Bug")
                        .with_inner_size(winit::dpi::LogicalSize::new(2560, 1440)),
                )
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
            WindowEvent::RedrawRequested => state.render_frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn tex_bind_entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

impl State {
    fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No adapter");

        println!("Adapter: {:?}", adapter.get_info().name);

        let required = wgpu::Features::TEXTURE_ATOMIC;
        assert!(
            adapter.features().contains(required),
            "Texture atomics not supported"
        );

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: required,
            ..Default::default()
        }))
        .unwrap();

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

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let tex_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_ATOMIC | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let storage_view = storage_texture.create_view(&Default::default());

        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&Default::default());

        // Pipelines use auto-layout; bind groups derived from pipeline layouts.
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("clear"),
            compilation_options: Default::default(),
            cache: None,
        });
        let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &clear_pipeline.get_bind_group_layout(0),
            entries: &[tex_bind_entry(
                0,
                wgpu::BindingResource::TextureView(&storage_view),
            )],
        });

        let raster_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("write_atomic"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let raster_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &raster_pipeline.get_bind_group_layout(0),
            entries: &[tex_bind_entry(
                0,
                wgpu::BindingResource::TextureView(&storage_view),
            )],
        });

        let vis_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("visualize"),
                targets: &[Some(surface_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let vis_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &vis_pipeline.get_bind_group_layout(0),
            entries: &[tex_bind_entry(
                0,
                wgpu::BindingResource::TextureView(&storage_view),
            )],
        });

        State {
            instance,
            window,
            device,
            queue,
            surface,
            surface_config,
            width,
            height,
            _storage_texture: storage_texture,
            _dummy_texture: dummy_texture,
            clear_pipeline,
            clear_bg,
            raster_pipeline,
            raster_bg,
            dummy_view,
            vis_pipeline,
            vis_bg,
        }
    }

    fn render_frame(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => f,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                drop(texture);
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self.instance.create_surface(self.window.clone()).unwrap();
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            _ => return,
        };
        let frame_view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());

        // Clear texture to zero
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, &self.clear_bg, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // Write via textureAtomicMax in fragment shader
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.dummy_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.raster_pipeline);
            pass.set_bind_group(0, &self.raster_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Read texture in fragment shader to visualize
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.vis_pipeline);
            pass.set_bind_group(0, &self.vis_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit([enc.finish()]);
        frame.present();
    }
}
