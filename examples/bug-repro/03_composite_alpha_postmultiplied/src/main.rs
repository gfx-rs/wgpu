//! Repro: on macOS/Metal, `CompositeAlphaMode::PostMultiplied` composites as
//! if the surface contents were premultiplied.
//!
//! The Metal backend advertises `PostMultiplied` and maps it to
//! `CAMetalLayer.isOpaque = false`. However, Core Animation always interprets
//! non-opaque layer contents as premultiplied alpha ("The CoreAnimation
//! compositor requires all of its inputs be premultiplied by alpha" per Apple
//! DTS; see also the OpenGL ES Programming Guide, "Tuning Your OpenGL ES
//! App"). So an application that follows the documented `PostMultiplied`
//! contract and presents straight-alpha pixels gets premultiplied compositing
//! instead, making partially transparent content too bright.
//!
//! This example opens an opaque black backdrop window and, on top of it, a
//! transparent window configured with `PostMultiplied`. The transparent window
//! contains four horizontal bands written with blending disabled:
//!
//!   1. probe: (1.0, 0.0, 0.0, 0.5), straight-alpha half-transparent red
//!   2. reference: opaque (1.0, 0.0, 0.0), what a premultiplied
//!      misinterpretation of the probe produces over black
//!   3. reference: opaque (0.735, 0.0, 0.0), correct post-multiplication of
//!      the probe if the compositor blends in linear light
//!   4. reference: opaque (0.5, 0.0, 0.0), correct post-multiplication of the
//!      probe if the compositor blends encoded values
//!
//! Expected: band 1 matches band 3 or band 4.
//! Actual on macOS: band 1 matches band 2 (full-brightness red).

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::{LogicalPosition, LogicalSize},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId, WindowLevel},
};

const BACKDROP_POS: (f64, f64) = (100.0, 100.0);
const BACKDROP_SIZE: (f64, f64) = (700.0, 560.0);
const PROBE_POS: (f64, f64) = (200.0, 160.0);
const PROBE_SIZE: (f64, f64) = (400.0, 400.0);

const SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) band: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) band: u32) -> VsOut {
    let x = f32(vi & 1u) * 2.0 - 1.0;
    let t = f32(vi >> 1u);
    let y = 1.0 - 2.0 * (f32(band) + t) / 4.0;
    return VsOut(vec4f(x, y, 0.0, 1.0), band);
}

const COLORS = array<vec4f, 4>(
    vec4f(1.0, 0.0, 0.0, 0.5),
    vec4f(1.0, 0.0, 0.0, 1.0),
    vec4f(0.735, 0.0, 0.0, 1.0),
    vec4f(0.5, 0.0, 0.0, 1.0),
);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    var colors = COLORS;
    return colors[in.band];
}
"#;

fn main() {
    env_logger::init();
    println!(
        "The topmost band is straight-alpha (1.0, 0.0, 0.0, 0.5) presented with \
         CompositeAlphaMode::PostMultiplied over an opaque black window.\n\
         If PostMultiplied is honored, it must appear darker than band 2 (opaque full red),\n\
         matching band 3 or 4 instead. If it matches band 2, the compositor treated the\n\
         surface as premultiplied."
    );
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

struct Gpu {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct Pane {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: Option<wgpu::RenderPipeline>,
}

#[derive(Default)]
struct App {
    state: Option<(Gpu, Pane, Pane)>,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) {
        let backdrop_window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("backdrop (opaque black)")
                        .with_decorations(false)
                        .with_position(LogicalPosition::new(BACKDROP_POS.0, BACKDROP_POS.1))
                        .with_inner_size(LogicalSize::new(BACKDROP_SIZE.0, BACKDROP_SIZE.1)),
                )
                .unwrap(),
        );
        let probe_window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("probe (PostMultiplied)")
                        .with_transparent(true)
                        .with_decorations(false)
                        .with_window_level(WindowLevel::AlwaysOnTop)
                        .with_position(LogicalPosition::new(PROBE_POS.0, PROBE_POS.1))
                        .with_inner_size(LogicalSize::new(PROBE_SIZE.0, PROBE_SIZE.1)),
                )
                .unwrap(),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(event_loop.owned_display_handle()),
        ));
        let backdrop_surface = instance.create_surface(backdrop_window.clone()).unwrap();
        let probe_surface = instance.create_surface(probe_window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&probe_surface),
            ..Default::default()
        }))
        .unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        let caps = probe_surface.get_capabilities(&adapter);
        println!("supported alpha modes: {:?}", caps.alpha_modes);
        assert!(
            caps.alpha_modes
                .contains(&wgpu::CompositeAlphaMode::PostMultiplied),
            "PostMultiplied not supported on this surface; cannot run this repro"
        );
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.has_srgb_suffix())
            .unwrap_or(caps.formats[0]);

        let make_config = |size: winit::dpi::PhysicalSize<u32>,
                           alpha_mode: wgpu::CompositeAlphaMode| {
            wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                view_formats: vec![],
                alpha_mode,
                width: size.width.max(1),
                height: size.height.max(1),
                desired_maximum_frame_latency: 2,
                present_mode: wgpu::PresentMode::AutoVsync,
            }
        };

        let backdrop_config = make_config(
            backdrop_window.inner_size(),
            wgpu::CompositeAlphaMode::Opaque,
        );
        backdrop_surface.configure(&device, &backdrop_config);

        let probe_config = make_config(
            probe_window.inner_size(),
            wgpu::CompositeAlphaMode::PostMultiplied,
        );
        probe_surface.configure(&device, &probe_config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        // Blending disabled: fragment output values are written to the surface
        // exactly as returned by the shader.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
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
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        self.state = Some((
            Gpu {
                instance,
                device,
                queue,
            },
            Pane {
                window: backdrop_window,
                surface: backdrop_surface,
                config: backdrop_config,
                pipeline: None,
            },
            Pane {
                window: probe_window,
                surface: probe_surface,
                config: probe_config,
                pipeline: Some(pipeline),
            },
        ));
    }
}

fn render(gpu: &Gpu, pane: &mut Pane) {
    let surface_texture = match pane.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture) => texture,
        wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => return,
        wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
            drop(texture);
            pane.surface.configure(&gpu.device, &pane.config);
            return;
        }
        wgpu::CurrentSurfaceTexture::Outdated => {
            pane.surface.configure(&gpu.device, &pane.config);
            return;
        }
        wgpu::CurrentSurfaceTexture::Validation => panic!("surface validation error"),
        wgpu::CurrentSurfaceTexture::Lost => {
            pane.surface = gpu.instance.create_surface(pane.window.clone()).unwrap();
            pane.surface.configure(&gpu.device, &pane.config);
            return;
        }
    };
    let view = surface_texture.texture.create_view(&Default::default());

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let Some(pipeline) = &pane.pipeline {
            pass.set_pipeline(pipeline);
            pass.draw(0..4, 0..4);
        }
    }
    gpu.queue.submit([encoder.finish()]);
    pane.window.pre_present_notify();
    gpu.queue.present(surface_texture);
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            self.init(event_loop);
        }
        if let Some((_, backdrop, probe)) = &self.state {
            backdrop.window.request_redraw();
            probe.window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some((gpu, backdrop, probe)) = &mut self.state else {
            return;
        };
        let pane = if id == backdrop.window.id() {
            backdrop
        } else {
            probe
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => render(gpu, pane),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                pane.config.width = size.width;
                pane.config.height = size.height;
                pane.surface.configure(&gpu.device, &pane.config);
                pane.window.request_redraw();
            }
            _ => (),
        }
    }
}
