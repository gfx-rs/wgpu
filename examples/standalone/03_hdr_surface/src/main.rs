//! HDR surface test.
//!
//! Prints the surface's supported (format, color space) combinations, then
//! configures the surface with the most capable color space available
//! (HDR10 > extended linear scRGB > encoded extended-range sRGB > sRGB) and
//! renders a luminance test pattern:
//!
//! * Top row: grayscale patches at 50 / 100 / 203 / 400 / 1000 / 10000 nits.
//!   On an SDR output everything from 100 nits up clips to the same white;
//!   on a working HDR output each patch is visibly brighter than the last.
//! * Middle row: red / green / blue / cyan / magenta / yellow at 203 nits
//!   (BT.709 primaries — they should look *the same* in every mode; if they
//!   look oversaturated in HDR10 the gamut conversion is wrong).
//! * Bottom row: logarithmic luminance gradient from 1 to 10000 nits.
//!
//! Set `HDR_MODE=hdr10|hlg|scrgb|extended-srgb|extended-display-p3|srgb` (on the
//! web: a `?mode=` query parameter) to force a particular color space instead of
//! auto-picking.
//! Set `WGPU_BACKEND=vulkan` to force the backend.

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

const SHADER: &str = r#"
struct Params {
    // 0 = sRGB SDR, 1 = extended linear scRGB, 2 = HDR10 PQ, 3 = HLG,
    // 4 = encoded extended-range sRGB, 5 = encoded extended-range Display-P3
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

// Standard sRGB OETF, valid for inputs in [0, 1]. `pow` is undefined for
// negative inputs, so callers must clamp first (the SDR path does).
fn srgb_oetf(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

// Extended sRGB OETF: the sRGB transfer function continued beyond [0, 1] with
// odd (point) symmetry through the origin, so values >1.0 (brighter than SDR
// reference white) and <0.0 (out-of-gamut) are encoded rather than clamped.
// This is what the `ExtendedSrgb` color space (browser HDR canvas, Vulkan
// EXTENDED_SRGB_NONLINEAR, Metal ExtendedSRGB) expects on the wire.
fn srgb_oetf_extended(c: vec3f) -> vec3f {
    let s = sign(c);
    let a = abs(c);
    let lo = a * 12.92;
    let hi = 1.055 * pow(a, vec3f(1.0 / 2.4)) - 0.055;
    return s * select(hi, lo, a <= vec3f(0.0031308));
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

// Linear BT.709 (sRGB) -> linear Display-P3, both D65. Column-major.
const BT709_TO_DISPLAYP3 = mat3x3f(
    vec3f(0.8224621, 0.0331942, 0.0170826),
    vec3f(0.1775380, 0.9668058, 0.0723974),
    vec3f(0.0000000, 0.0000000, 0.9105199),
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
        case 4u: {
            // Encoded extended-range sRGB: BT.709 primaries, the sRGB OETF
            // extended beyond [0, 1]. Same normalization as scRGB
            // (1.0 = 80 nits), but sRGB-encoded rather than linear.
            out = srgb_oetf_extended(nits / 80.0);
        }
        case 5u: {
            // Encoded extended-range Display-P3: convert BT.709 -> P3 primaries
            // (linear), normalize like scRGB (1.0 = 80 nits), then apply the
            // extended sRGB OETF. The BT.709 test primaries look identical, just
            // carried in the wider P3 container.
            out = srgb_oetf_extended((BT709_TO_DISPLAYP3 * nits) / 80.0);
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

/// Print to stdout on native, the developer console on the web.
fn report(msg: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    println!("{msg}");
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&msg.into());
}

/// Report the display HDR snapshot returned by [`wgpu::Surface::display_hdr_info`]
/// — the read-only *sensor* that says what the panel can show right now. Every
/// field is advisory and platform-dependent (`None` == unknown here, *not* an
/// SDR display). This is also the manual-verification surface: on macOS, dimming
/// the display changes `headroom`/`hdr_active` live.
fn report_display_hdr_info(info: &wgpu::DisplayHdrInfo) {
    report("Display HDR snapshot (advisory; None = unknown on this platform):");
    report(&format!("  hdr_active:     {:?}", info.hdr_active));
    report(&format!("  luminance:      {:?}", info.luminance));
    report(&format!("  headroom:       {:?}", info.headroom));
    report(&format!("  chromaticity:   {:?}", info.chromaticity));
    report(&format!("  coarse:         {:?}", info.coarse));
    report(&format!("  bits_per_color: {:?}", info.bits_per_color));
    // The single value most tone-mappers want, plus the HDR-worthwhile gate.
    report(&format!(
        "  -> tone_map_headroom() = {:?}, has_hdr_headroom() = {}",
        info.tone_map_headroom(),
        info.has_hdr_headroom()
    ));
}

/// The forced mode, from the `HDR_MODE` environment variable on native or
/// the `?mode=` query parameter on the web.
fn forced_mode() -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("HDR_MODE").ok()
    }
    #[cfg(target_arch = "wasm32")]
    {
        let search = web_sys::window()?.location().search().ok()?;
        search
            .strip_prefix('?')?
            .split('&')
            .find_map(|pair| pair.strip_prefix("mode="))
            .map(str::to_owned)
    }
}

/// Run a future to completion: synchronously on native, in the browser's
/// event loop on the web (where blocking is not allowed).
fn spawn(future: impl core::future::Future<Output = ()> + 'static) {
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(future);
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(future);
}

struct ModeChoice {
    format: wgpu::TextureFormat,
    color_space: wgpu::SurfaceColorSpace,
    shader_mode: u32,
}

/// Pick the most capable (format, color space) combination the surface
/// supports, preferring HDR10, then extended linear scRGB, then encoded
/// extended-range sRGB, then sRGB.
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
        (
            Cs::ExtendedSrgb,
            Csf::EXTENDED_SRGB,
            4,
            &[wgpu::TextureFormat::Rgba16Float],
        ),
        (
            Cs::ExtendedDisplayP3,
            Csf::EXTENDED_DISPLAY_P3,
            5,
            &[wgpu::TextureFormat::Rgba16Float],
        ),
    ];

    // `ExtendedDisplayP3` is reachable only via an explicit `extended-display-p3`
    // request, not auto-picked: it needs gamut conversion and the BT.709 test
    // pattern looks the same as `ExtendedSrgb`, so auto-pick keeps the simpler
    // BT.709 HDR path.
    let allowed = |cs: Cs| match forced {
        None => cs == Cs::Hdr10 || cs == Cs::ExtendedSrgbLinear || cs == Cs::ExtendedSrgb,
        Some("hdr10") => cs == Cs::Hdr10,
        Some("hlg") => cs == Cs::Hlg,
        Some("scrgb") => cs == Cs::ExtendedSrgbLinear,
        Some("extended-srgb") => cs == Cs::ExtendedSrgb,
        Some("extended-display-p3") => cs == Cs::ExtendedDisplayP3,
        Some("srgb") => false,
        Some(other) => {
            panic!("unknown mode {other:?} (use hdr10|hlg|scrgb|extended-srgb|extended-display-p3|srgb)")
        }
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
    /// Kept so the display snapshot can be re-polled after startup —
    /// `display_hdr_info` takes the adapter, exactly like `get_capabilities`.
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    /// The most recent display snapshot, so a re-poll only logs on change.
    last_hdr_info: wgpu::DisplayHdrInfo,
    /// Frames since the last display re-poll (the snapshot is throttled rather
    /// than read every frame; see [`State::poll_display_hdr_info`]).
    frames_since_poll: u32,
}

/// How often (in frames) the demo re-polls the display snapshot. The values
/// change on human timescales, so a coarse interval still surfaces live changes
/// (e.g. dimming the display) without re-walking the OS display every frame.
const HDR_POLL_INTERVAL_FRAMES: u32 = 30;

impl State {
    async fn new(window: Arc<Window>) -> State {
        // `from_env_or_default` honors `WGPU_BACKEND` (e.g. `vulkan`, `dx12`)
        // so each backend's color-space path can be tested separately.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
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
        report(&format!("Adapter: {} ({:?})", info.name, info.backend));

        let caps = surface.get_capabilities(&adapter);
        report("Surface formats and color spaces:");
        for fc in &caps.format_capabilities {
            report(&format!("  {:?}: {:?}", fc.format, fc.color_spaces));
        }

        // Read the display sensor alongside the surface capabilities. An app
        // uses this to decide whether requesting HDR output is worthwhile and to
        // seed a tone-map target; here we just report it (and re-poll later).
        let hdr_info = surface.display_hdr_info(&adapter);
        report_display_hdr_info(&hdr_info);

        let forced = forced_mode();
        let choice = pick_mode(&caps, forced.as_deref());
        report(&format!(
            "Configuring surface with {:?} + {:?}",
            choice.format, choice.color_space
        ));
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
            adapter,
            device,
            queue,
            surface,
            config,
            pipeline,
            bind_group,
            last_hdr_info: hdr_info,
            frames_since_poll: 0,
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    /// Re-poll the display snapshot (throttled) and report it only when it
    /// changes.
    ///
    /// `display_hdr_info` is a snapshot, not a stream: wgpu owns no event loop
    /// and can't notify us, so an app re-queries from its own loop. Brightness,
    /// HDR-toggle, and monitor moves are not delivered as events — the value
    /// just changes — so a real app would also re-pick its color space / refresh
    /// its tone-map target here. A real app would re-query from its windowing
    /// events; this demo polls every [`HDR_POLL_INTERVAL_FRAMES`] frames, which
    /// still surfaces live changes to a human while avoiding a per-frame OS walk.
    fn poll_display_hdr_info(&mut self) {
        self.frames_since_poll += 1;
        if self.frames_since_poll < HDR_POLL_INTERVAL_FRAMES {
            return;
        }
        self.frames_since_poll = 0;

        let info = self.surface.display_hdr_info(&self.adapter);
        if info != self.last_hdr_info {
            report("Display HDR snapshot changed:");
            report_display_hdr_info(&info);
            self.last_hdr_info = info;
        }
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

struct App {
    /// Taken on the first `resumed` call so initialization happens once.
    proxy: Option<EventLoopProxy<State>>,
    state: Option<State>,
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(proxy) = self.proxy.take() else {
            return;
        };

        #[cfg_attr(
            not(target_arch = "wasm32"),
            expect(unused_mut, reason = "wasm32 re-assigns to specify canvas")
        )]
        let mut attributes = Window::default_attributes().with_title("wgpu HDR test");

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .expect("the page must have a <canvas id=\"canvas\">")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            attributes = attributes.with_canvas(Some(canvas));
        }

        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        // On native this blocks and the state arrives before `resumed`
        // returns; on the web it is delivered later via `user_event`.
        spawn(async move {
            let state = State::new(window).await;
            let _ = proxy.send_event(state);
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, state: State) {
        state.window.request_redraw();
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => {
                state.poll_display_hdr_info();
                state.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let event_loop = EventLoop::<State>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let app = App {
        proxy: Some(event_loop.create_proxy()),
        state: None,
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).unwrap();
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
}
