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
//! * Middle row: red / green / blue / cyan / magenta / yellow at 203 nits.
//!   These are BT.709 primaries, so they should look *the same* in every mode;
//!   if they look oversaturated in HDR10 the gamut conversion is wrong.
//! * Bottom row: logarithmic luminance gradient from 1 to 10000 nits.
//!
//! Pass a mode as the first argument —
//! `hdr10|hlg|scrgb|extended-srgb|extended-display-p3|srgb` (e.g. `cargo run --
//! hdr10`; on the web: a `?mode=` query parameter) — to force a particular color
//! space instead of auto-picking.
//! Set `WGPU_BACKEND=vulkan` to force the backend.

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

#[cfg(target_os = "macos")]
mod macos;

/// Print to stdout on native, the developer console on the web.
fn report(msg: impl std::fmt::Display) {
    #[cfg(not(target_arch = "wasm32"))]
    println!("{msg}");
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&msg.to_string().into());
}

/// Report the display HDR info returned by
/// [`wgpu::Surface::display_hdr_info`], the read-only query of what the panel can
/// show right now, tagged with the `source` that triggered it (startup, a window
/// event, or the macOS notification). Every field is advisory and
/// platform-dependent (`None` == unknown here, **not** an SDR display). It's also
/// a live check: on macOS, dimming the display changes `headroom` between
/// re-queries.
fn report_display_hdr_info(source: &str, info: &wgpu::DisplayHdrInfo) {
    report(format_args!(
        "Display HDR info [{source}] (advisory; None = unknown on this platform):"
    ));
    report(format_args!("  luminance:      {:?}", info.luminance));
    report(format_args!("  headroom:       {:?}", info.headroom));
    report(format_args!("  chromaticity:   {:?}", info.chromaticity));
    report(format_args!("  coarse:         {:?}", info.coarse));
    report(format_args!("  bits_per_color: {:?}", info.bits_per_color));
    // The one number a tone-mapper needs: how far above SDR white highlights can
    // go this frame (1.0 == none). Whether to use HDR at all is a separate
    // question, answered by the surface's color spaces (see `pick_mode`), not by
    // this live value.
    report(format_args!(
        "  -> tone_map_headroom() = {:?}",
        info.tone_map_headroom()
    ));
}

/// The forced mode, from the first positional CLI argument on native (e.g.
/// `cargo run -- hdr10`) or the `?mode=` query parameter on the web.
fn forced_mode() -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::args().nth(1)
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

/// Run a future to completion concurrently: on a worker thread on native, or in
/// the browser's event loop on the web (where blocking is not allowed).
#[cfg(not(target_arch = "wasm32"))]
fn spawn(future: impl core::future::Future<Output = ()> + Send + 'static) {
    std::thread::spawn(move || pollster::block_on(future));
}

/// Run a future to completion concurrently: on a worker thread on native, or in
/// the browser's event loop on the web (where blocking is not allowed).
#[cfg(target_arch = "wasm32")]
fn spawn(future: impl core::future::Future<Output = ()> + 'static) {
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
    const PREFERENCES: &[(Cs, Csf, u32, &[wgpu::TextureFormat])] = &[
        (
            Cs::Bt2100Pq,
            Csf::BT2100_PQ,
            2,
            &[
                wgpu::TextureFormat::Rgb10a2Unorm,
                wgpu::TextureFormat::Rgba16Float,
            ],
        ),
        (
            Cs::Bt2100Hlg,
            Csf::BT2100_HLG,
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
        None => cs == Cs::Bt2100Pq || cs == Cs::ExtendedSrgbLinear || cs == Cs::ExtendedSrgb,
        Some("hdr10") => cs == Cs::Bt2100Pq,
        Some("hlg") => cs == Cs::Bt2100Hlg,
        Some("scrgb") => cs == Cs::ExtendedSrgbLinear,
        Some("extended-srgb") => cs == Cs::ExtendedSrgb,
        Some("extended-display-p3") => cs == Cs::ExtendedDisplayP3,
        Some("srgb") => false,
        Some(other) => {
            panic!("unknown mode {other:?} (use hdr10|hlg|scrgb|extended-srgb|extended-display-p3|srgb)")
        }
    };

    for &(cs, flag, shader_mode, preferred_formats) in PREFERENCES {
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
    /// Kept so the display info can be re-queried after startup;
    /// `display_hdr_info` takes the adapter, exactly like `get_capabilities`.
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    /// The most recent display info, so a re-query only logs on change.
    last_hdr_info: wgpu::DisplayHdrInfo,
}

impl State {
    async fn new(
        window: Arc<Window>,
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
    ) -> State {
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
        report(format_args!("Adapter: {} ({:?})", info.name, info.backend));

        let caps = surface.get_capabilities(&adapter);
        report("Surface formats and color spaces:");
        for fc in &caps.format_capabilities {
            report(format_args!("  {:?}: {:?}", fc.format, fc.color_spaces));
        }

        // Spell out the HDR / wide-gamut spaces each format offers beyond the
        // universally-supported SDR `Srgb`/`DisplayP3`, i.e. exactly what an
        // app must see advertised here before it can request HDR output. On the
        // web this is the key signal: an fp16 (`Rgba16Float`) canvas
        // should advertise `ExtendedSrgb` / `ExtendedDisplayP3` whenever it is
        // configurable, regardless of whether the display is *currently* in HDR
        // mode (that state lives in the display info below, not here).
        let sdr = wgpu::SurfaceColorSpaces::SRGB | wgpu::SurfaceColorSpaces::DISPLAY_P3;
        report("HDR / wide-gamut color spaces (beyond SDR sRGB / Display-P3):");
        for fc in &caps.format_capabilities {
            let hdr = fc.color_spaces.difference(sdr);
            report(format_args!(
                "  {:?}: {}",
                fc.format,
                if hdr.is_empty() {
                    "none (SDR only)".to_owned()
                } else {
                    format!("{hdr:?}")
                }
            ));
        }

        let forced = forced_mode();
        if let Some(mode) = forced.as_deref() {
            report(format_args!(
                "Forced mode requested (CLI arg / ?mode=): {mode:?}"
            ));
        }
        let choice = pick_mode(&caps, forced.as_deref());
        // Make the SDR fallback explicit: landing on the `Auto` path after a
        // (non-`srgb`) mode was requested means the requested color space was
        // **not** advertised by `color_spaces()`, the situation this example
        // makes visible. On web + `Rgba16Float` the extended spaces should be
        // advertised regardless of display HDR state, so seeing this there
        // means the capability query is under-reporting.
        if matches!(choice.color_space, wgpu::SurfaceColorSpace::Auto)
            && forced.as_deref().is_some_and(|m| m != "srgb")
        {
            report(
                "  -> requested color space is NOT advertised by the surface; \
                 falling back to SDR (Auto).",
            );
        }
        // `pick_mode` already chose in one pass; `is_hdr()` on the result is the
        // whole capability branch — an HDR space is the one whose highlights you
        // scale by `tone_map_headroom()`.
        let dynamic_range = if choice.color_space.is_hdr() {
            "HDR"
        } else {
            "SDR"
        };
        report(format_args!(
            "Configuring surface with {:?} + {:?} ({dynamic_range})",
            choice.format, choice.color_space,
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

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // mode, encode_srgb
        let params: [u32; 2] = [
            choice.shader_mode,
            u32::from(choice.shader_mode == 0 && !choice.format.is_srgb()),
        ];
        let params_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: params.map(u32::to_ne_bytes).as_flattened(),
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
            // Seeded by the first main-thread query in `user_event`.
            last_hdr_info: wgpu::DisplayHdrInfo::default(),
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    /// Re-query the display HDR info and log it if it changed.
    ///
    /// `display_hdr_info` is a point-in-time read, so re-query when the display
    /// might have changed. winit reports window changes (`Resized` / `Moved`);
    /// this example also installs a macOS observer for the in-place changes winit
    /// has no event for (HDR toggled, or the brightness slider that moves EDR
    /// headroom). Other platforms' in-place triggers are left unwired.
    ///
    /// Call this on the main thread: on the Metal backend `display_hdr_info` reads
    /// main-thread-only `NSScreen`, and all triggers deliver here.
    ///
    /// macOS EDR headroom drifts and ramps over ~1-2 s, so apps that tone-map from
    /// it tend to re-read every frame; otherwise a query at startup is enough,
    /// since every value but that headroom is advisory.
    fn requery_display_hdr_info(&mut self, source: &str) {
        let info = self.surface.display_hdr_info(&self.adapter);
        // Change-gated: a printed line means this `source` changed something.
        if info != self.last_hdr_info {
            report_display_hdr_info(source, &info);
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
        let view = surface_texture.texture.create_view(&Default::default());

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

/// Events delivered to the winit loop from outside a `WindowEvent`.
enum UserEvent {
    /// The async setup finished; carries the initialized `State`. Boxed to keep
    /// the event small (`State` is large).
    Initialized(Box<State>),
    /// macOS: the display configuration changed, so re-query the HDR info. See
    /// [`macos::observe_screen_parameter_changes`].
    #[cfg(target_os = "macos")]
    ScreenParametersChanged,
}

struct App {
    /// Taken on the first `resumed` call so initialization happens once.
    proxy: Option<EventLoopProxy<UserEvent>>,
    state: Option<State>,
    /// macOS: holds the screen-change observer so it stays registered for the
    /// app's lifetime (dropping it removes the observer).
    #[cfg(target_os = "macos")]
    screen_observer: Option<macos::ScreenObserver>,
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(proxy) = self.proxy.take() else {
            return;
        };

        // macOS: start observing screen-parameter changes so the display HDR
        // info is re-queried reactively. AppKit posts the notification when the
        // display configuration changes — including the SDR brightness slider,
        // which moves the EDR headroom — and winit doesn't surface it.
        #[cfg(target_os = "macos")]
        {
            self.screen_observer = Some(macos::observe_screen_parameter_changes(proxy.clone()));
        }

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

        // Create the surface here, on the main thread: winit only hands out the
        // raw window handle from the thread that owns the window (on Windows,
        // `window_handle()` errors with `Unavailable` anywhere else).
        //
        // `InstanceDescriptor::new_without_display_handle_from_env` honors
        // `WGPU_BACKEND` (e.g. `vulkan`, `dx12`) so each backend's color-space
        // path can be exercised separately.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).unwrap();

        // The async tail (adapter, device, pipeline) runs off the main thread
        // (native) or in the browser event loop (web); the finished `State`
        // arrives back on the main thread via `user_event`.
        spawn(async move {
            let state = State::new(window, instance, surface).await;
            let _ = proxy.send_event(UserEvent::Initialized(Box::new(state)));
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Initialized(state) => {
                let mut state = *state;
                // The first display query runs here because it must be on the
                // main thread (see `requery_display_hdr_info`) and `State::new`
                // ran on a worker. It also seeds the baseline later re-queries
                // compare against.
                let info = state.surface.display_hdr_info(&state.adapter);
                report_display_hdr_info("initial query", &info);
                state.last_hdr_info = info;

                state.window.request_redraw();
                self.state = Some(state);
            }
            // macOS: the screen changed (HDR toggled, brightness moved, monitor
            // switched). Re-query — the headroom can change our tone-map target.
            #[cfg(target_os = "macos")]
            UserEvent::ScreenParametersChanged => {
                if let Some(state) = self.state.as_mut() {
                    state.requery_display_hdr_info("macOS screen-parameters notification");
                    state.window.request_redraw();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size);
                // A resize can also land the window on another monitor, so
                // re-query here too.
                state.requery_display_hdr_info("window resized");
                state.window.request_redraw();
            }
            // The main signal for a monitor change. winit doesn't deliver it on
            // Wayland or web, so they won't react (see `requery_display_hdr_info`).
            WindowEvent::Moved(_) => state.requery_display_hdr_info("window moved"),
            // The test pattern is static, so we render on demand (startup, OS
            // expose, resize) rather than spinning a continuous redraw loop.
            WindowEvent::RedrawRequested => state.render(),
            _ => {}
        }
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let app = App {
        proxy: Some(event_loop.create_proxy()),
        state: None,
        #[cfg(target_os = "macos")]
        screen_observer: None,
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
