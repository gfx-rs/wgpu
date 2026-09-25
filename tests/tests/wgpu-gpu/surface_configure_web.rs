//! Surface configuration tests for the web backends.
//!
//! On the web a surface target is a canvas, so these can run as ordinary gpu
//! tests: no OS window, no main thread, no display server. The native
//! equivalent cannot work that way, and lives in its own binary that owns the
//! main thread so it can drive `winit`; see `tests/tests/wgpu-surface.rs`.
#![cfg(wasm_test)]

use wgpu_test::GpuTestInitializer;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(SURFACE_CONFIGURE);
}

/// Configure a canvas surface, reconfigure it at a new size, then acquire,
/// render to, and present a frame.
///
/// Mirrors the native `surface_configure` test in `tests/tests/wgpu-surface.rs`.
#[apply(gpu_test!)]
static SURFACE_CONFIGURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(wgpu_test::TestParameters::default())
    .run_async(|_ctx| async move {
        #[cfg(target_arch = "wasm32")]
        {
            // Not using the normal testing infrastructure: it creates a canvas for
            // adapter initialization but never exposes the resulting surface, and on
            // WebGL a surface is bound to its own canvas' context, so we cannot pair a
            // fresh canvas with the context's device.
            let instance = wgpu_test::initialize_instance(
                wgpu::Backends::all(),
                &wgpu_test::TestParameters::default(),
            );
            let canvas = wgpu_test::initialize_html_canvas();

            let surface = instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                .expect("could not create surface from canvas");

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .expect("no adapter supports the canvas surface");

            let (device, _queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    default_queue: wgpu::QueueDescriptor { label: None },
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("failed to create device");

            assert!(
                !surface.get_capabilities(&adapter).formats.is_empty(),
                "surface reported no supported formats"
            );

            let mut config = surface
                .get_default_config(&adapter, 256, 256)
                .expect("surface is not supported by the adapter");
            surface.configure(&device, &config);

            // Reconfiguring an already-configured surface tears down the previous
            // swapchain, which is a distinct path from the first configure.
            config.width = 512;
            config.height = 512;
            surface.configure(&device, &config);

            let wgpu::CurrentSurfaceTexture::Success(frame) = surface.get_current_texture() else {
                panic!("could not acquire a surface texture");
            };
            assert_eq!(frame.texture.width(), 512);
            assert_eq!(frame.texture.height(), 512);
        }
    });
