//! Repro for various issues with queue presentation
//!
//! The 2 current bugs being tested are presentation after no usage of surface texture
//! and queue destruction immediately after present
//!
//! Passing `--emulate-view-formats` additionally tests the fallback for surface
//! view formats, where the application renders into a separate texture that is
//! copied into the swapchain image when presenting.
//!
//! Passing `--draw-before-present` runs a minimal clear pass into the sRGB view
//! format of the surface texture, whose surface format is the non-sRGB variant.
//!
//! Passing `--manual-close` keeps the window open and presenting until it is
//! closed by the user, instead of exiting after the first present.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
mod emulated;

fn main() {
    env_logger::init();
    let options = Options::from_args();
    if options.manual_close {
        println!("The window stays open and presents until you close it");
    }
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App {
        state: None,
        options,
    };
    event_loop.run_app(&mut app).unwrap();
}

/// Command line flags.
#[derive(Clone, Copy)]
struct Options {
    /// Force surface view formats to be emulated.
    emulate_view_formats: bool,
    /// Run a clear pass into the sRGB view format before presenting.
    draw_before_present: bool,
    /// Keep the window open until it is closed by the user, instead of exiting
    /// after the first present.
    manual_close: bool,
}

impl Options {
    fn from_args() -> Self {
        let mut options = Self {
            emulate_view_formats: false,
            draw_before_present: false,
            manual_close: false,
        };
        for argument in std::env::args().skip(1) {
            match argument.as_str() {
                "--emulate-view-formats" => options.emulate_view_formats = true,
                "--draw-before-present" => options.draw_before_present = true,
                "--manual-close" => options.manual_close = true,
                _ => log::warn!("ignoring unknown argument {argument:?}"),
            }
        }
        options
    }
}

/// Picks a surface format and the sRGB view format of it: the surface format is
/// the non-sRGB variant, and its sRGB variant is used as the view format.
/// Formats without an sRGB variant, like `Rgba16Float`, are skipped.
fn pick_surface_and_view_formats(
    formats: &[wgpu::TextureFormat],
) -> Option<(wgpu::TextureFormat, wgpu::TextureFormat)> {
    formats
        .iter()
        .copied()
        .find(|format| !format.has_srgb_suffix() && format.add_srgb_suffix() != *format)
        .map(|format| (format, format.add_srgb_suffix()))
}

struct App {
    state: Option<State>,
    options: Options,
}

enum State {
    /// Uses the `wgpu` API with whatever the backend supports natively.
    Wgpu(WgpuState),
    /// Forces surface view formats to be emulated.
    #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
    Emulated(emulated::EmulatedState),
}

impl State {
    fn new(window: Arc<Window>, options: Options) -> Self {
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
        if options.emulate_view_formats {
            return Self::Emulated(emulated::EmulatedState::new(window, options));
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        if options.emulate_view_formats {
            log::error!("--emulate-view-formats is not supported on this platform");
        }

        Self::Wgpu(WgpuState::new(window, options))
    }

    fn window(&self) -> &Window {
        match self {
            Self::Wgpu(state) => &state.window,
            #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
            Self::Emulated(state) => state.window(),
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        match self {
            Self::Wgpu(state) => state.resize(size),
            #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
            Self::Emulated(state) => state.resize(size),
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        match self {
            Self::Wgpu(state) => state.redraw(event_loop),
            #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
            Self::Emulated(state) => state.redraw(event_loop),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Presentation bugs"))
                .unwrap(),
        );
        self.state = Some(State::new(window, self.options));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => {
                log::debug!("the window was closed");
                event_loop.exit();
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.resize(size);
            }
            WindowEvent::RedrawRequested => state.redraw(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window().request_redraw();
        }
    }
}

struct WgpuState {
    window: Arc<Window>,
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: Option<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    view_format: wgpu::TextureFormat,
    draw_before_present: bool,
    manual_close: bool,
}

impl WgpuState {
    fn new(window: Arc<Window>, options: Options) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        instance_desc.flags |= wgpu::InstanceFlags::advanced_debugging();
        let instance = wgpu::Instance::new(instance_desc);
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No adapter");

        println!("Adapter: {:?}", adapter.get_info().name);

        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();

        let caps = surface.get_capabilities(&adapter);
        let (surface_format, view_format) = if options.draw_before_present {
            pick_surface_and_view_formats(&caps.formats)
                .expect("the surface has no format with an sRGB variant")
        } else {
            let surface_format = caps.formats[0];
            (surface_format, surface_format.add_srgb_suffix())
        };
        if options.draw_before_present {
            println!(
                "Clearing the view format {view_format:?} of the surface format {surface_format:?}"
            );
        }
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: if options.draw_before_present {
                vec![view_format]
            } else {
                vec![]
            },
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        Self {
            window,
            instance,
            device,
            queue: Some(queue),
            surface,
            surface_config,
            view_format,
            draw_before_present: options.draw_before_present,
            manual_close: options.manual_close,
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.reconfigure();
    }

    fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(_) | wgpu::CurrentSurfaceTexture::Outdated => {
                self.reconfigure();
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self.instance.create_surface(self.window.clone()).unwrap();
                self.reconfigure();
                return;
            }
            _ => return,
        };
        // Without `--manual-close`, take the queue so that dropping it right
        // after presenting exercises queue destruction.
        let queue = if self.manual_close {
            self.queue.clone()
        } else {
            self.queue.take()
        };
        let Some(queue) = queue else {
            return;
        };
        if self.draw_before_present {
            // The minimal clear pass: a color attachment with `LoadOp::Clear`
            // and no draw calls at all.
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("(example) clear view"),
                format: Some(self.view_format),
                ..Default::default()
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("(example) clear"),
                });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("(example) clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            queue.submit([encoder.finish()]);
        }
        // Immediately present the surface texture and, unless the window is
        // closed manually, drop the queue, which should cause a full wait.
        queue.present(frame);
        if !self.manual_close {
            event_loop.exit();
        }
    }
}
