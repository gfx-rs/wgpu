//! Support for the `--emulate-view-formats` mode.
//!
//! Whether surface view formats can go on the swapchain images is reported by
//! [`wgpu_hal::SurfaceCapabilities::native_view_formats`], which backends set to
//! `true` whenever they can (Vulkan with `VK_KHR_swapchain_mutable_format`,
//! Metal and DX12 always). To reach the emulated path on such a backend, this
//! mode sets `wgpu-core` up from a `wgpu-hal` instance and wraps the hal adapter
//! so that it reports the capability as `false`.

use std::sync::Arc;

use wgpu_types as wgt;
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::{pick_surface_and_view_formats, Options};

#[cfg(target_os = "linux")]
type HalApi = wgpu_hal::api::Vulkan;
#[cfg(target_os = "windows")]
type HalApi = wgpu_hal::api::Dx12;
#[cfg(target_os = "macos")]
type HalApi = wgpu_hal::api::Metal;

pub struct EmulatedState {
    // Dropped in declaration order: the surface has to be destroyed before the
    // window it was created from, and before the device and the instance that
    // created it.
    surface: Arc<wgpu_core::instance::Surface>,
    queue: Option<Arc<wgpu_core::device::queue::Queue>>,
    device: Arc<wgpu_core::device::Device>,
    instance: Arc<wgpu_core::instance::Instance>,
    window: Arc<Window>,
    surface_config: wgpu::SurfaceConfiguration,
    view_format: wgt::TextureFormat,
    draw_before_present: bool,
    manual_close: bool,
}

impl EmulatedState {
    pub fn new(window: Arc<Window>, options: Options) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = unsafe { create_instance::<HalApi>(&window) };
        let surface = unsafe { create_surface(&instance, &window) };

        let backend = <HalApi as wgpu_hal::Api>::VARIANT;
        let adapters = unsafe {
            instance
                .raw(backend)
                .expect("the instance has no hal instance for the backend")
                .enumerate_adapters(
                    surface
                        .surface_per_backend
                        .get(&backend)
                        .map(|hal_surface| &**hal_surface),
                )
        };
        let mut exposed = adapters
            .into_iter()
            .next()
            .expect("the backend exposed no adapter");

        // Claim that the swapchain images can't carry the view formats, which
        // makes wgpu-core hand out an intermediate texture that it copies into
        // the swapchain image when presenting.
        exposed.adapter = Box::new(EmulateSurfaceViewFormats {
            inner: exposed.adapter,
        });
        let adapter = unsafe { instance.create_adapter_from_hal(exposed) };

        let (device, queue) = adapter
            .request_device(&wgpu_core::device::DeviceDescriptor::default())
            .expect("failed to create the device");

        let caps = surface
            .get_capabilities(&adapter)
            .expect("failed to get the surface capabilities");
        let (format, view_format) = pick_surface_and_view_formats(&caps.formats)
            .expect("the surface has no format with an sRGB variant");

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![view_format],
            desired_maximum_frame_latency: 2,
        };
        surface
            .configure(&device, &surface_config)
            .expect("failed to configure the surface");

        println!("Adapter: {:?}", adapter.get_info().name);
        println!("Emulating the view format {view_format:?} of the surface format {format:?}");
        if options.draw_before_present {
            println!("Clearing the {view_format:?} view before presenting");
        }

        Self {
            surface,
            queue: Some(queue),
            device,
            instance,
            window,
            surface_config,
            view_format,
            draw_before_present: options.draw_before_present,
            manual_close: options.manual_close,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.reconfigure();
    }

    pub fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(error) => {
                log::error!("failed to acquire the surface texture: {error}");
                event_loop.exit();
                return;
            }
        };

        match &output.status {
            wgpu::SurfaceStatus::Good | wgpu::SurfaceStatus::Suboptimal => {}
            wgpu::SurfaceStatus::Lost => {
                self.surface = unsafe { create_surface(&self.instance, &self.window) };
                self.reconfigure();
                return;
            }
            status => {
                log::warn!("surface status {status:?}, reconfiguring");
                self.reconfigure();
                return;
            }
        }

        let Some(texture) = output.texture.clone() else {
            log::error!("no surface texture was handed out");
            event_loop.exit();
            return;
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
            self.clear_pass(&texture, &queue);
        }

        // Immediately present the surface texture and, unless the window is
        // closed manually, drop the queue, which should cause a full wait.
        match self.surface.present() {
            Ok(status) => log::debug!("presented with status {status:?}"),
            Err(error) => log::error!("failed to present: {error}"),
        }
        drop(output);
        drop(queue);
        if !self.manual_close {
            event_loop.exit();
        }
    }

    /// Runs a minimal clear pass into a view of `texture` with the differing
    /// view format: a color attachment with `LoadOp::Clear` and no draw calls.
    fn clear_pass(
        &self,
        texture: &Arc<wgpu_core::resource::Texture>,
        queue: &Arc<wgpu_core::device::queue::Queue>,
    ) {
        let view = texture.create_view(&wgpu_core::resource::TextureViewDescriptor {
            label: None,
            format: Some(self.view_format),
            ..Default::default()
        });

        let encoder = self
            .device
            .create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

        {
            let color_attachments = [Some(wgpu_core::command::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                load_op: wgt::LoadOp::Clear(wgt::Color::BLUE),
                store_op: wgt::StoreOp::Store,
            })];
            let mut pass =
                encoder.begin_render_pass(wgpu_core::command::ResolvedRenderPassDescriptor {
                    label: None,
                    color_attachments: std::borrow::Cow::Borrowed(&color_attachments),
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            pass.end();
        }

        let command_buffer = encoder.finish(&wgt::CommandBufferDescriptor::default());
        queue.submit(&[command_buffer]);
    }

    fn reconfigure(&mut self) {
        if let Err(error) = self.surface.configure(&self.device, &self.surface_config) {
            log::error!("failed to configure the surface: {error}");
        }
    }
}

/// Creates a `wgpu-core` instance around a real `wgpu-hal` instance.
unsafe fn create_instance<A: wgpu_hal::Api>(window: &Window) -> Arc<wgpu_core::instance::Instance> {
    use raw_window_handle::HasDisplayHandle as _;

    let display_handle = window
        .display_handle()
        .expect("the window has no display handle");
    let descriptor = wgpu_hal::InstanceDescriptor {
        name: "bug-repro-02-present-bugs",
        flags: wgpu::InstanceFlags::advanced_debugging().with_env(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        telemetry: None,
        display: Some(display_handle),
    };

    let hal_instance = unsafe { <A::Instance as wgpu_hal::Instance>::init(&descriptor) }
        .expect("failed to create the hal instance");
    wgpu_core::instance::Instance::from_hal_instance::<A>(
        "bug-repro-02-present-bugs".to_owned(),
        hal_instance,
    )
}

unsafe fn create_surface(
    instance: &Arc<wgpu_core::instance::Instance>,
    window: &Window,
) -> Arc<wgpu_core::instance::Surface> {
    use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};

    let display_handle = window
        .display_handle()
        .expect("the window has no display handle")
        .as_raw();
    let window_handle = window
        .window_handle()
        .expect("the window has no window handle")
        .as_raw();

    unsafe { instance.create_surface(Some(display_handle), window_handle) }
        .expect("failed to create the surface")
}

/// Wraps a hal adapter, reporting that the swapchain images can't carry view
/// formats other than the surface format.
struct EmulateSurfaceViewFormats {
    inner: Box<dyn wgpu_hal::DynAdapter>,
}

impl wgpu_hal::DynResource for EmulateSurfaceViewFormats {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }
}

impl wgpu_hal::DynAdapter for EmulateSurfaceViewFormats {
    unsafe fn open(
        &self,
        features: wgpu::Features,
        limits: &wgpu::Limits,
        memory_hints: &wgpu::MemoryHints,
    ) -> Result<wgpu_hal::DynOpenDevice, wgpu_hal::DeviceError> {
        unsafe { self.inner.open(features, limits, memory_hints) }
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgpu::TextureFormat,
    ) -> wgpu_hal::TextureFormatCapabilities {
        unsafe { self.inner.texture_format_capabilities(format) }
    }

    unsafe fn surface_capabilities(
        &self,
        surface: &dyn wgpu_hal::DynSurface,
    ) -> Option<wgpu_hal::SurfaceCapabilities> {
        let mut capabilities = unsafe { self.inner.surface_capabilities(surface) }?;
        capabilities.native_view_formats = false;
        Some(capabilities)
    }

    unsafe fn surface_display_hdr_info(
        &self,
        surface: &dyn wgpu_hal::DynSurface,
    ) -> Option<wgpu::DisplayHdrInfo> {
        unsafe { self.inner.surface_display_hdr_info(surface) }
    }

    unsafe fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        unsafe { self.inner.get_presentation_timestamp() }
    }

    fn get_ordered_buffer_usages(&self) -> wgpu::BufferUses {
        self.inner.get_ordered_buffer_usages()
    }

    fn get_ordered_texture_usages(&self) -> wgpu::TextureUses {
        self.inner.get_ordered_texture_usages()
    }
}
