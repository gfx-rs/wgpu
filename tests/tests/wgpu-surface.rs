//! Surface tests that need a real window.
//!
//! These cannot live in the `wgpu-gpu` harness: that harness runs each test on a
//! worker thread, and `winit` requires the main thread. This binary therefore
//! forces `libtest-mimic` to run trials inline on the main thread.
//!
//! For web, this is covered by an ordinary gpu test,
//! `tests/tests/wgpu-gpu/surface_configure.rs`.

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main();
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::Arc;

    use winit::{
        application::ApplicationHandler,
        event_loop::{ActiveEventLoop, EventLoop, OwnedDisplayHandle},
        platform::run_on_demand::EventLoopExtRunOnDemand,
        window::{Window, WindowAttributes},
    };

    struct Harness {
        failure: Option<String>,
    }

    impl ApplicationHandler for Harness {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            let window = Arc::new(
                event_loop
                    .create_window(WindowAttributes::default().with_visible(false))
                    .expect("failed to create window"),
            );
            let display_handle = event_loop.owned_display_handle();
            if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                surface_configure(window, display_handle)
            })) {
                self.failure = Some(format!("{e:?}"));
            }
            event_loop.exit();
        }

        fn window_event(
            &mut self,
            _: &ActiveEventLoop,
            _: winit::window::WindowId,
            _: winit::event::WindowEvent,
        ) {
        }
    }

    fn surface_configure(window: Arc<Window>, display_handle: OwnedDisplayHandle) {
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display_handle)),
        );

        let surface = instance.create_surface(window).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no adapter supports the window surface");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits:
                wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
            default_queue: wgpu::QueueDescriptor { label: None },
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .unwrap();

        let mut config = surface.get_default_config(&adapter, 256, 256).unwrap();
        surface.configure(&device, &config);

        // Reconfigure the surface (regression test for https://github.com/gfx-rs/wgpu/issues/10410).
        config.width = 512;
        config.height = 512;
        surface.configure(&device, &config);

        if let wgpu::CurrentSurfaceTexture::Success(frame) = surface.get_current_texture() {
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            queue.submit(Some(enc.finish()));
            queue.present(frame);
        }
    }

    pub fn main() {
        let mut args = libtest_mimic::Arguments::from_args();
        // `winit` requires the main thread; this makes libtest-mimic run inline.
        args.test_threads = Some(1);

        let trial = libtest_mimic::Trial::test("surface_configure", || {
            let mut builder = EventLoop::builder();
            #[cfg(target_os = "macos")]
            {
                use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
                // Keep the test process out of the Dock and stop it stealing focus.
                builder.with_activation_policy(ActivationPolicy::Prohibited);
            }

            let mut event_loop = match builder.build() {
                Ok(event_loop) => event_loop,
                Err(e) => panic!("could not create event loop (no display?): {e}"),
            };

            let mut harness = Harness { failure: None };
            event_loop.run_app_on_demand(&mut harness).unwrap();
            match harness.failure {
                Some(failure) => Err(libtest_mimic::Failed::from(failure)),
                None => Ok(()),
            }
        });

        libtest_mimic::run(&args, vec![trial]).exit_if_failed();
    }
}
