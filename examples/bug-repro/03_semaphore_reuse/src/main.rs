//! Repro for a binary-semaphore reuse hazard in `wgpu_hal::vulkan`'s swapchain
//! acquire path.
//!
//! `wgpu_core::present::Surface::get_current_texture_inner` calls the real
//! `vkAcquireNextImageKHR` (via `hal::Surface::acquire_texture`) *before* it
//! checks whether a texture is already outstanding:
//!
//! ```ignore
//! let (texture, status) = match unsafe { suf.acquire_texture(...) } {
//!     Ok(ast) => {
//!         // ... wraps `ast` in a `Texture` ...
//!         if present.acquired_texture.is_some() {
//!             return Err(SurfaceError::AlreadyAcquired); // too late: we already acquired!
//!         }
//!         present.acquired_texture = Some(texture.clone());
//!         // ...
//!     }
//!     // ...
//! };
//! ```
//!
//! So calling [`wgpu::Surface::get_current_texture`] a second time before
//! presenting (or discarding) the first result doesn't harmlessly no-op: it
//! performs a second, real acquisition, then throws the result away as a
//! validation error. That thrown-away acquisition's `VkSemaphore` was
//! signaled by the presentation engine, but — because it's discarded before
//! ever becoming a [`wgpu::SurfaceTexture`] — it never goes through
//! `wgpu::SurfaceTexture`'s `Drop` (which would call `Surface::discard`, and
//! properly release the semaphore). It also never gets used in a
//! `Queue::submit` call, which is the *only* place `wgpu_hal::vulkan` ever
//! waits on one of these semaphores
//! (see `wgpu_hal::vulkan::swapchain::native::SwapchainAcquireSemaphore`).
//!
//! Each acquisition — good or thrown-away — advances the swapchain's
//! acquire-semaphore ring irrespective of whether that semaphore ever gets
//! waited on. So a semaphore poisoned this way eventually comes back around
//! and gets passed to `vkAcquireNextImageKHR` a second time while still
//! signaled from the first, unwaited acquisition — violating
//! [VUID-vkAcquireNextImageKHR-semaphore-01286][vuid], which requires the
//! semaphore to be unsignaled.
//!
//! [vuid]: https://docs.vulkan.org/spec/latest/chapters/VK_KHR_surface/wsi.html#VUID-vkAcquireNextImageKHR-semaphore-01286
//!
//! Run with `RUST_LOG=wgpu_hal=info` (or similar) to see the Vulkan
//! validation layer's report. Note that `wgpu_hal::vulkan`'s swapchain also
//! has a *separate*, already-identified bug where the per-swapchain fence
//! passed to every `vkAcquireNextImageKHR` call is never waited on or reset
//! outside Windows, which trips VUID-vkAcquireNextImageKHR-fence-10066 on
//! the very first repeated acquire. That VUID — and, once enough images have
//! been leaked, other synchronization-hazard warnings too — will almost
//! certainly show up in the log as noise. This example is specifically about
//! the semaphore one, VUID-vkAcquireNextImageKHR-semaphore-01286.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// How many acquire/present cycles to run. Each cycle poisons one semaphore
/// slot, so this only needs to exceed the swapchain's image count (usually
/// 2-4) for the ring to wrap around onto a poisoned slot. Kept small: every
/// poisoned acquisition also permanently holds a real swapchain image (it's
/// thrown away before we ever get a handle to present or discard it), so
/// enough iterations will exhaust the image pool and turn later acquisitions
/// into genuine (slow) timeouts rather than new violations.
const ITERATIONS: u32 = 6;

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
    ran: bool,
}

struct State {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Semaphore reuse repro"))
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
            WindowEvent::RedrawRequested => {
                // Run once: this is a one-shot repro, not an animation.
                if !self.ran {
                    self.ran = true;
                    state.run_repro();
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

impl State {
    fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        // Force the Vulkan validation layer on, regardless of the caller's
        // environment, so the VUID report shows up unconditionally.
        instance_desc.flags |= wgpu::InstanceFlags::debugging();
        let instance = wgpu::Instance::new(instance_desc);
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No adapter");

        println!(
            "Adapter: {:?} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();

        let surface_format = surface.get_capabilities(&adapter).formats[0];
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        State {
            window,
            device,
            queue,
            surface,
        }
    }

    /// Repeatedly acquire a texture, then — before presenting it — acquire a
    /// *second* one. The second acquisition is real at the Vulkan level (see
    /// the module doc comment) but gets reported to us as a validation
    /// error, and its handle is thrown away. Only then do we present the
    /// first, legitimately-held texture and move to the next iteration.
    fn run_repro(&mut self) {
        println!(
            "Running {ITERATIONS} acquire/poison/present cycles; watch the log for \
             VUID-vkAcquireNextImageKHR-semaphore-01286."
        );
        for i in 0..ITERATIONS {
            let held = match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                other => {
                    println!("[{i}] first get_current_texture() returned {other:?}, aborting");
                    return;
                }
            };

            // Deliberately violate the "present or drop before re-acquiring"
            // contract. Wrap it in an error scope so the validation error
            // `get_current_texture` raises for `AlreadyAcquired` doesn't
            // panic (there's no `on_uncaptured_error` handler registered).
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let poisoned = self.surface.get_current_texture();
            let error = pollster::block_on(scope.pop());
            match (&poisoned, &error) {
                (wgpu::CurrentSurfaceTexture::Validation, Some(err)) => {
                    println!("[{i}] second get_current_texture() raised (expected): {err}");
                }
                (wgpu::CurrentSurfaceTexture::Timeout, None) => {
                    println!(
                        "[{i}] second get_current_texture() timed out — likely because earlier \
                         leaked acquisitions have exhausted the swapchain's image pool (expected \
                         once enough images have been leaked; not a sign the bug is fixed)"
                    );
                }
                _ => {
                    println!(
                        "[{i}] second get_current_texture() returned {poisoned:?} / {error:?} \
                         (expected a Validation status with an AlreadyAcquired error — did the \
                         AlreadyAcquired check move earlier than the acquire call?)"
                    );
                }
            }

            // Present the texture we legitimately hold. We never wrote to
            // it, so wgpu inserts an implicit clear-and-transition
            // submission, which correctly waits on and consumes *its*
            // acquire semaphore. The second, thrown-away acquisition's
            // semaphore gets no such treatment.
            self.queue.present(held);
        }
        println!(
            "Done. If a poisoned semaphore slot got reacquired above, the Vulkan \
             validation layer should have logged VUID-vkAcquireNextImageKHR-semaphore-01286."
        );
    }
}
