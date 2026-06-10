//! HDR surface test.
//!
//! Prints the surface's supported (format, color space) combinations, then
//! configures the surface with the most capable color space available
//! (HDR10 > extended linear scRGB > sRGB) and renders a luminance test
//! pattern:
//!
//! * Top row: grayscale patches at 50 / 100 / 203 / 400 / 1000 / 10000 nits.
//!   On an SDR output everything from 100 nits up clips to the same white;
//!   on a working HDR output each patch is visibly brighter than the last.
//! * Middle row: red / green / blue / cyan / magenta / yellow at 203 nits
//!   (BT.709 primaries — they should look *the same* in every mode; if they
//!   look oversaturated in HDR10 the gamut conversion is wrong).
//! * Bottom row: logarithmic luminance gradient from 1 to 10000 nits.
//!
//! Set `HDR_MODE=hdr10|hlg|scrgb|srgb` to force a particular color space
//! instead of auto-picking. Set `WGPU_BACKEND=vulkan` to force the backend.

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const SHADER: &str = r#"
struct Params {
    // 0 = sRGB SDR, 1 = extended linear scRGB, 2 = HDR10 PQ, 3 = HLG
    mode: u32,
    // 1 if the shader must apply the sRGB OETF itself (non-sRGB SDR format)
    encode_srgb: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle. Constructed in a single expression: a local
    // `var out: VertexOutput` would hit the same naga ArrayStride/Offset
    // issue described above (structs get explicit member offsets).
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return VertexOutput(vec4f(x, y, 0.0, 1.0), vec2f(x, -y) * 0.5 + 0.5);
}

// Note: indexed lookups are written as if-chains rather than local
// `array<...>` values, because naga's SPIR-V backend decorates array types
// with ArrayStride, which is invalid on function-local variables
// (VUID-StandaloneSpirv-None-10684; see
// https://github.com/gfx-rs/wgpu/issues/7696).

fn staircase_nits(i: u32) -> f32 {
    if i == 0u { return 50.0; }
    if i == 1u { return 100.0; }
    if i == 2u { return 203.0; }
    if i == 3u { return 400.0; }
    if i == 4u { return 1000.0; }
    return 10000.0;
}

fn primary_secondary(i: u32) -> vec3f {
    if i == 0u { return vec3f(1.0, 0.0, 0.0); }
    if i == 1u { return vec3f(0.0, 1.0, 0.0); }
    if i == 2u { return vec3f(0.0, 0.0, 1.0); }
    if i == 3u { return vec3f(0.0, 1.0, 1.0); }
    if i == 4u { return vec3f(1.0, 0.0, 1.0); }
    return vec3f(1.0, 1.0, 0.0);
}

// The test pattern, in linear BT.709 with absolute luminance in nits.
fn pattern_nits(uv: vec2f) -> vec3f {
    let i = min(u32(uv.x * 6.0), 5u);
    if uv.y < 0.3333 {
        // Grayscale staircase.
        return vec3f(staircase_nits(i));
    } else if uv.y < 0.6667 {
        // BT.709 primaries and secondaries at 203 nits.
        return primary_secondary(i) * 203.0;
    } else {
        // Log gradient, 1 nit -> 10000 nits.
        return vec3f(pow(10.0, uv.x * 4.0));
    }
}

fn srgb_oetf(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

// SMPTE ST 2084 (PQ) OETF; input is luminance normalized to 10000 nits.
fn pq_oetf(y: vec3f) -> vec3f {
    let m1 = 0.1593017578125;
    let m2 = 78.84375;
    let c1 = 0.8359375;
    let c2 = 18.8515625;
    let c3 = 18.6875;
    let yp = pow(max(y, vec3f(0.0)), vec3f(m1));
    return pow((c1 + c2 * yp) / (1.0 + c3 * yp), vec3f(m2));
}

// BT.2100 HLG OETF; input is scene luminance normalized to 1000-nit peak.
fn hlg_oetf(y: vec3f) -> vec3f {
    let a = 0.17883277;
    let b = 0.28466892;
    let c = 0.55991073;
    let lo = sqrt(3.0 * y);
    let hi = a * log(12.0 * y - b) + c;
    return select(hi, lo, y <= vec3f(1.0 / 12.0));
}

const BT709_TO_BT2020 = mat3x3f(
    vec3f(0.627402, 0.069095, 0.016394),
    vec3f(0.329292, 0.919544, 0.088028),
    vec3f(0.043306, 0.011360, 0.895578),
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let nits = pattern_nits(in.uv);

    var out: vec3f;
    switch params.mode {
        case 1u: {
            // Extended linear scRGB: BT.709 primaries, linear, 1.0 = 80 nits.
            out = nits / 80.0;
        }
        case 2u: {
            // HDR10: BT.2020 primaries, PQ-encoded absolute luminance.
            out = pq_oetf(BT709_TO_BT2020 * (nits / 10000.0));
        }
        case 3u: {
            // HLG: BT.2020 primaries, 1000-nit nominal peak.
            out = hlg_oetf(BT709_TO_BT2020 * (min(nits, vec3f(1000.0)) / 1000.0));
        }
        default: {
            // SDR sRGB: clip at 100 nits.
            out = clamp(nits / 100.0, vec3f(0.0), vec3f(1.0));
            if params.encode_srgb == 1u {
                out = srgb_oetf(out);
            }
        }
    }
    return vec4f(out, 1.0);
}
"#;

struct ModeChoice {
    format: wgpu::TextureFormat,
    color_space: wgpu::SurfaceColorSpace,
    shader_mode: u32,
}

/// Pick the most capable (format, color space) combination the surface
/// supports, preferring HDR10, then extended linear scRGB, then sRGB.
fn pick_mode(caps: &wgpu::SurfaceCapabilities, forced: Option<&str>) -> ModeChoice {
    use wgpu::{SurfaceColorSpace as Cs, SurfaceColorSpaces as Csf};

    // (color space, flag, shader mode, preferred formats in order)
    let preferences: &[(Cs, Csf, u32, &[wgpu::TextureFormat])] = &[
        (
            Cs::Hdr10,
            Csf::HDR10,
            2,
            &[
                wgpu::TextureFormat::Rgb10a2Unorm,
                wgpu::TextureFormat::Rgba16Float,
            ],
        ),
        (
            Cs::Hlg,
            Csf::HLG,
            3,
            &[
                wgpu::TextureFormat::Rgb10a2Unorm,
                wgpu::TextureFormat::Rgba16Float,
            ],
        ),
        (
            Cs::ExtendedSrgbLinear,
            Csf::EXTENDED_SRGB_LINEAR,
            1,
            &[wgpu::TextureFormat::Rgba16Float],
        ),
    ];

    let allowed = |cs: Cs| match forced {
        None => cs == Cs::Hdr10 || cs == Cs::ExtendedSrgbLinear,
        Some("hdr10") => cs == Cs::Hdr10,
        Some("hlg") => cs == Cs::Hlg,
        Some("scrgb") => cs == Cs::ExtendedSrgbLinear,
        Some("srgb") => false,
        Some(other) => panic!("unknown HDR_MODE {other:?} (use hdr10|hlg|scrgb|srgb)"),
    };

    for &(cs, flag, shader_mode, preferred_formats) in preferences {
        if !allowed(cs) {
            continue;
        }
        // Try the preferred formats first, then anything else that
        // supports this color space.
        let preferred = preferred_formats
            .iter()
            .copied()
            .filter(|&f| caps.color_spaces(f).contains(flag));
        let any = caps
            .format_capabilities
            .iter()
            .filter(|fc| fc.color_spaces.contains(flag))
            .map(|fc| fc.format);
        if let Some(format) = preferred.chain(any).next() {
            return ModeChoice {
                format,
                color_space: cs,
                shader_mode,
            };
        }
    }

    // SDR fallback.
    let format = caps.formats[0];
    ModeChoice {
        format,
        color_space: wgpu::SurfaceColorSpace::Auto,
        shader_mode: 0,
    }
}

struct State {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let info = adapter.get_info();
        println!("Adapter: {} ({:?})", info.name, info.backend);

        let caps = surface.get_capabilities(&adapter);
        println!("Surface formats and color spaces:");
        for fc in &caps.format_capabilities {
            println!("  {:?}: {:?}", fc.format, fc.color_spaces);
        }

        let forced = std::env::var("HDR_MODE").ok();
        let choice = pick_mode(&caps, forced.as_deref());
        println!(
            "Configuring surface with {:?} + {:?}",
            choice.format, choice.color_space
        );
        window.set_title(&format!(
            "wgpu HDR test — {:?} + {:?}",
            choice.format, choice.color_space
        ));

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: choice.format,
            color_space: choice.color_space,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hdr test pattern"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // mode, encode_srgb
        let params: [u32; 2] = [
            choice.shader_mode,
            (choice.shader_mode == 0 && !choice.format.is_srgb()) as u32,
        ];
        let params_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck_cast(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
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
                resource: params_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hdr test pattern"),
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
                targets: &[Some(choice.format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        State {
            window,
            device,
            queue,
            surface,
            config,
            pipeline,
            bind_group,
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&Default::default());
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
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(surface_texture);
    }
}

fn bytemuck_cast(params: &[u32; 2]) -> &[u8] {
    // Avoid a bytemuck dependency for two u32s.
    unsafe { core::slice::from_raw_parts(params.as_ptr().cast(), size_of_val(params)) }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("wgpu HDR test"))
                .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
