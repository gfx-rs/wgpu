//! Deterministic reproducer for <https://github.com/gfx-rs/wgpu/issues/8224>
//! (`surface.configure()` during window resize leaks memory and slows down
//! `queue.submit()`).
//!
//! The original report was produced by dragging a window by hand for six
//! minutes and watching the process RSS in a task manager. That makes the
//! result impossible to compare between two runs, between two machines, or
//! before and after a candidate fix, because the number of `configure()` calls
//! is unknown and the compositor is in the loop.
//!
//! This repro removes the hand from the loop. It never asks the window manager
//! to resize anything. It drives `Surface::configure()` itself, a fixed number
//! of times, over a fixed sequence of extents, and reports numbers instead of
//! impressions:
//!
//! * `rss` - process resident set size, read from `/proc/self/statm` on Linux.
//!   This is the number the original report was based on.
//! * `submit` - mean wall time of `Queue::submit()` over the last reporting
//!   window, which is the slowdown the issue title refers to.
//! * the `wgpu` internal counters from `Device::get_internal_counters()`, which
//!   are the object and memory populations `wgpu` itself is responsible for.
//! * `gpu_alloc` totals from `Device::generate_allocator_report()`, when the
//!   backend provides them.
//!
//! The point of reporting all four together is that they separate the possible
//! causes. If RSS climbs while the `wgpu` counters stay flat, the growth is not
//! `wgpu` object bookkeeping and not `wgpu` sub-allocated device memory, so it
//! belongs to the loader, the driver, or an enabled validation layer. If a
//! counter climbs with RSS, that counter names the leaked object type.
//!
//! What the counters do not cover: swapchain images are owned by the
//! presentation engine and are not counted by `hal.textures`, and neither the
//! Vulkan loader, the driver, nor an enabled validation layer allocates through
//! anything `wgpu` counts. So a flat counter column does not mean nothing
//! leaked, it means nothing `wgpu` tracks leaked, which is exactly the
//! distinction this repro exists to make.
//!
//! # Modes
//!
//! The three modes are an A/B/C for the claim in the issue that the leak
//! follows `configure()` and not the frame loop.
//!
//! * `configure-present` (default) reconfigures, acquires, clears, submits and
//!   presents every iteration. This is what the reported application does.
//! * `configure-only` reconfigures every iteration and does nothing else. This
//!   isolates swapchain creation and destruction.
//! * `present-only` configures once and then acquires, clears, submits and
//!   presents every iteration. This is the control: the issue reports that no
//!   leak occurs when `configure()` is not called.
//!
//! # Usage
//!
//! ```sh
//! cargo run -p wgpu-bug-repro-03-surface-reconfigure-leak --release
//! cargo run -p wgpu-bug-repro-03-surface-reconfigure-leak --release -- --mode present-only
//! cargo run -p wgpu-bug-repro-03-surface-reconfigure-leak --release -- --iterations 20000
//! ```
//!
//! To attribute growth to the validation layers, run it twice, once with
//! `WGPU_VALIDATION=0` (or without the Vulkan validation layers installed) and
//! once with them enabled, and compare the `rss` columns.

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Base surface extent. The sweep varies the configured extent around this.
const BASE_WIDTH: u32 = 640;
const BASE_HEIGHT: u32 = 480;

/// Iterations to run before the "before" sample is taken, so that lazily
/// initialized allocations and first-use driver caches are not counted as a
/// leak.
const DEFAULT_WARMUP: u32 = 200;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// Reconfigure, then draw and present, every iteration.
    ConfigurePresent,
    /// Reconfigure every iteration and do nothing else.
    ConfigureOnly,
    /// Configure once, then draw and present every iteration.
    PresentOnly,
}

impl Mode {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "configure-present" => Some(Mode::ConfigurePresent),
            "configure-only" => Some(Mode::ConfigureOnly),
            "present-only" => Some(Mode::PresentOnly),
            _ => None,
        }
    }

    fn reconfigures(self) -> bool {
        matches!(self, Mode::ConfigurePresent | Mode::ConfigureOnly)
    }

    fn presents(self) -> bool {
        matches!(self, Mode::ConfigurePresent | Mode::PresentOnly)
    }
}

struct Args {
    mode: Mode,
    iterations: u32,
    report_every: u32,
    warmup: u32,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            mode: Mode::ConfigurePresent,
            iterations: 5000,
            report_every: 500,
            warmup: DEFAULT_WARMUP,
        };

        let mut argv = std::env::args().skip(1);
        while let Some(arg) = argv.next() {
            let mut value = || {
                argv.next()
                    .unwrap_or_else(|| panic!("{arg} requires a value"))
            };
            match arg.as_str() {
                "--mode" => {
                    let raw = value();
                    args.mode = Mode::parse(&raw).unwrap_or_else(|| {
                        panic!(
                            "unknown mode {raw:?}, expected one of \
                             configure-present, configure-only, present-only"
                        )
                    });
                }
                "--iterations" => args.iterations = value().parse().expect("--iterations"),
                "--report-every" => args.report_every = value().parse().expect("--report-every"),
                "--warmup" => args.warmup = value().parse().expect("--warmup"),
                "--help" | "-h" => {
                    println!(
                        "usage: [--mode configure-present|configure-only|present-only] \
                         [--iterations N] [--report-every N] [--warmup N]"
                    );
                    std::process::exit(0);
                }
                other => panic!("unknown argument {other:?}, try --help"),
            }
        }

        assert!(args.report_every > 0, "--report-every must be positive");
        args
    }
}

/// The extent used on iteration `i`.
///
/// The steps are coprime with the ranges so the sequence visits every extent in
/// the range before repeating, and every consecutive pair of iterations asks for
/// a different extent. That matters: reconfiguring to the same extent lets a
/// driver keep the existing swapchain, which would hide the thing we are
/// measuring.
fn extent_for(iteration: u32) -> (u32, u32) {
    let width = BASE_WIDTH + (iteration.wrapping_mul(7)) % 128;
    let height = BASE_HEIGHT + (iteration.wrapping_mul(13)) % 96;
    (width, height)
}

/// Resident set size in KiB, on platforms where it can be read without adding a
/// dependency. This is the number the original bug report is expressed in.
fn rss_kib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        // Field 2 of /proc/self/statm is the resident set size in pages.
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        // 4 KiB pages. wgpu does not depend on libc, so sysconf(_SC_PAGESIZE) is
        // not available here; every Linux target wgpu supports uses 4 KiB pages
        // except some aarch64 kernels, where this under-reports by a constant
        // factor and the trend still holds.
        Some(pages * 4)
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // No /proc, and no libc dependency here, so ask ps. This only runs at
        // report points, so the fork cost is not in the measured path.
        let pid = std::process::id().to_string();
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
    {
        None
    }
}

/// One row of the report.
#[derive(Clone)]
struct Sample {
    iteration: u32,
    configures: u64,
    frames: u64,
    rss_kib: Option<u64>,
    submit_mean: Duration,
    textures: isize,
    texture_views: isize,
    memory_allocations: isize,
    texture_memory: isize,
    buffer_memory: isize,
    command_encoders: isize,
    fences: isize,
    allocator_reserved_bytes: Option<u64>,
}

impl Sample {
    fn take(
        device: &wgpu::Device,
        iteration: u32,
        configures: u64,
        frames: u64,
        submit_mean: Duration,
    ) -> Self {
        let counters = device.get_internal_counters();
        let hal = counters.hal;
        let report = device.generate_allocator_report();
        Sample {
            iteration,
            configures,
            frames,
            rss_kib: rss_kib(),
            submit_mean,
            textures: hal.textures.read(),
            texture_views: hal.texture_views.read(),
            memory_allocations: hal.memory_allocations.read(),
            texture_memory: hal.texture_memory.read(),
            buffer_memory: hal.buffer_memory.read(),
            command_encoders: hal.command_encoders.read(),
            fences: hal.fences.read(),
            allocator_reserved_bytes: report.map(|r| r.total_reserved_bytes),
        }
    }
}

fn header() {
    println!(
        "{:>8}  {:>10}  {:>11}  {:>8}  {:>6}  {:>8}  {:>12}  {:>12}  {:>4}  {:>3}  {:>12}",
        "iter",
        "rss_kib",
        "submit_us",
        "textures",
        "views",
        "mem_alloc",
        "tex_mem",
        "buf_mem",
        "enc",
        "fnc",
        "gpu_reserved",
    );
}

fn row(s: &Sample) {
    let fmt_opt_u64 = |v: Option<u64>| match v {
        Some(v) => v.to_string(),
        None => "n/a".to_string(),
    };
    println!(
        "{:>8}  {:>10}  {:>11.1}  {:>8}  {:>6}  {:>8}  {:>12}  {:>12}  {:>4}  {:>3}  {:>12}",
        s.iteration,
        fmt_opt_u64(s.rss_kib),
        s.submit_mean.as_secs_f64() * 1e6,
        s.textures,
        s.texture_views,
        s.memory_allocations,
        s.texture_memory,
        s.buffer_memory,
        s.command_encoders,
        s.fences,
        fmt_opt_u64(s.allocator_reserved_bytes),
    );
}

fn summarize(mode: Mode, before: &Sample, after: &Sample) {
    let iters = after.iteration.saturating_sub(before.iteration).max(1);
    println!();
    println!("--- summary ({mode:?}) ---");
    println!(
        "measured over {iters} iterations (iteration {} to {}, warmup excluded)",
        before.iteration, after.iteration
    );

    let configures = after.configures.saturating_sub(before.configures);
    let frames = after.frames.saturating_sub(before.frames);
    println!("configure() calls:  {configures}");
    println!("frames presented:   {frames}");

    match (before.rss_kib, after.rss_kib) {
        (Some(b), Some(a)) => {
            let delta = a as i64 - b as i64;
            println!(
                "rss:                {b} KiB -> {a} KiB  ({delta:+} KiB, \
                 {:+.3} KiB per iteration)",
                delta as f64 / iters as f64
            );
            let bytes = delta as f64 * 1024.0;
            if configures > 0 {
                println!(
                    "rss per configure:  {:+.1} bytes",
                    bytes / configures as f64
                );
            }
            if frames > 0 {
                println!("rss per frame:      {:+.1} bytes", bytes / frames as f64);
            }
            println!(
                "                    Attribute with the modes: run present-only to get the \
                 per-frame rate, then subtract frames * that rate from the configure-present \
                 total to get the part that belongs to configure()."
            );
        }
        _ => println!("rss:                not available on this platform"),
    }

    println!(
        "submit mean:        {:.1} us -> {:.1} us",
        before.submit_mean.as_secs_f64() * 1e6,
        after.submit_mean.as_secs_f64() * 1e6
    );

    let counter = |name: &str, b: isize, a: isize| {
        let delta = a - b;
        let flag = if delta == 0 { "flat" } else { "GREW" };
        println!("{name:<20}{b} -> {a}  ({delta:+}) {flag}");
    };
    counter("hal textures:       ", before.textures, after.textures);
    counter(
        "hal texture_views:  ",
        before.texture_views,
        after.texture_views,
    );
    counter(
        "hal memory_allocs:  ",
        before.memory_allocations,
        after.memory_allocations,
    );
    counter(
        "hal texture_memory: ",
        before.texture_memory,
        after.texture_memory,
    );
    counter(
        "hal buffer_memory:  ",
        before.buffer_memory,
        after.buffer_memory,
    );
    counter(
        "hal cmd_encoders:   ",
        before.command_encoders,
        after.command_encoders,
    );
    counter("hal fences:         ", before.fences, after.fences);

    println!();
    println!(
        "Read this as: any counter marked GREW is a wgpu-owned population that \
         did not settle. If every counter is flat and rss still climbed, the \
         growth is not wgpu object bookkeeping or wgpu sub-allocated device \
         memory, and the next place to look is the Vulkan loader, the driver, \
         or an enabled validation layer."
    );
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App {
        args,
        state: None,
        done: false,
    };
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    args: Args,
    state: Option<State>,
    done: bool,
}

struct State {
    _window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    iteration: u32,
    /// Number of `Surface::configure()` calls made so far.
    configures: u64,
    /// Number of frames acquired, submitted and presented so far.
    frames: u64,
    /// Total time spent inside `Queue::submit()` since the last report.
    submit_time: Duration,
    /// Number of `Queue::submit()` calls since the last report.
    submit_count: u32,
    acquire_failures: u32,

    before: Option<Sample>,
    last: Option<Sample>,
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
                        .with_title("wgpu #8224 surface reconfigure stress")
                        .with_inner_size(PhysicalSize::new(BASE_WIDTH, BASE_HEIGHT)),
                )
                .unwrap(),
        );
        self.state = Some(State::new(window, &self.args));
        header();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Deliberately ignore `WindowEvent::Resized`. This repro drives
        // `configure()` on its own schedule so that the iteration count is
        // exact and the compositor is not part of the measurement.
        if let WindowEvent::CloseRequested = event {
            event_loop.exit();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            return;
        }
        let Some(state) = &mut self.state else { return };

        state.step(&self.args);

        if state.iteration >= self.args.iterations {
            self.done = true;
            state.finish(&self.args);
            event_loop.exit();
        }
    }
}

impl State {
    fn step(&mut self, args: &Args) {
        let i = self.iteration;

        if args.mode.reconfigures() {
            let (width, height) = extent_for(i);
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
            self.configures += 1;
        }

        if args.mode.presents() {
            self.draw_frame();
        }

        self.iteration += 1;

        let done_warmup = self.iteration == args.warmup;
        let report_due = self.iteration > args.warmup
            && (self.iteration - args.warmup).is_multiple_of(args.report_every);

        if done_warmup || report_due {
            let submit_mean = if self.submit_count == 0 {
                Duration::ZERO
            } else {
                self.submit_time / self.submit_count
            };
            self.submit_time = Duration::ZERO;
            self.submit_count = 0;

            let sample = Sample::take(
                &self.device,
                self.iteration,
                self.configures,
                self.frames,
                submit_mean,
            );
            row(&sample);
            if self.before.is_none() {
                self.before = Some(sample.clone());
            }
            self.last = Some(sample);
        }
    }

    fn draw_frame(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            // An extent that does not match the window is expected here: this
            // repro reconfigures without asking the window manager to resize.
            // The reconfigure has already happened, which is what is being
            // measured, so a missed frame is not a problem.
            _ => {
                self.acquire_failures += 1;
                return;
            }
        };

        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            // A clear-only pass is enough. It still goes through render pass and
            // framebuffer creation in the backend, which is the part of the
            // frame that is keyed on the surface extent.
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        let command_buffer = encoder.finish();

        let start = Instant::now();
        self.queue.submit([command_buffer]);
        self.submit_time += start.elapsed();
        self.submit_count += 1;

        self.queue.present(frame);
        self.frames += 1;
    }

    fn finish(&mut self, args: &Args) {
        if self.acquire_failures > 0 {
            println!();
            println!(
                "note: {} of {} acquisitions did not return a usable texture",
                self.acquire_failures, args.iterations
            );
        }
        match (self.before.take(), self.last.take()) {
            (Some(before), Some(after)) if before.iteration != after.iteration => {
                summarize(args.mode, &before, &after)
            }
            _ => println!(
                "\nnot enough samples to summarize; \
                 run with more --iterations than --warmup"
            ),
        }
    }

    fn new(window: Arc<Window>, args: &Args) -> Self {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        instance_desc.flags |= wgpu::InstanceFlags::advanced_debugging();
        let instance = wgpu::Instance::new(instance_desc);
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No adapter");

        let info = adapter.get_info();
        println!("backend:  {:?}", info.backend);
        println!("adapter:  {} ({:?})", info.name, info.device_type);
        println!("driver:   {} {}", info.driver, info.driver_info);
        println!(
            "mode:     {:?}, iterations: {}, warmup: {}, report every: {}",
            args.mode, args.iterations, args.warmup, args.report_every
        );
        println!();

        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats[0];
        // No vsync, so the loop is bounded by the reconfigure cost rather than
        // by the refresh rate.
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::AutoVsync
        };
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: BASE_WIDTH,
            height: BASE_HEIGHT,
            present_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        State {
            _window: window,
            device,
            queue,
            surface,
            surface_config,
            iteration: 0,
            configures: 0,
            frames: 0,
            submit_time: Duration::ZERO,
            submit_count: 0,
            acquire_failures: 0,
            before: None,
            last: None,
        }
    }
}
