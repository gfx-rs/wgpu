#![cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use parking_lot::Mutex;

struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
}

fn fixture() -> Fixture {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let surface = instance.create_noop_surface().unwrap();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .unwrap();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

    let config = surface.get_default_config(&adapter, 64, 64).unwrap();
    surface.configure(&device, &config);

    Fixture {
        device,
        queue,
        surface,
    }
}

fn acquire(surface: &wgpu::Surface<'_>) -> wgpu::SurfaceTexture {
    match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture) => texture,
        other => panic!("expected a surface texture, got {other:?}"),
    }
}

fn register_uncaptured_error_handler(device: &wgpu::Device) -> Arc<Mutex<Vec<wgpu::Error>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let errors_clone = errors.clone();

    device.on_uncaptured_error(Arc::new(move |error| {
        errors_clone.lock().push(error);
    }));

    errors
}

#[test]
fn drop_surface_texture_before_surface() {
    let f = fixture();
    let errors = register_uncaptured_error_handler(&f.device);

    let texture = acquire(&f.surface);
    drop(texture);
    drop(f.surface);

    assert!(errors.lock().is_empty());
}

#[test]
fn drop_surface_before_surface_texture() {
    let f = fixture();
    let errors = register_uncaptured_error_handler(&f.device);

    let texture = acquire(&f.surface);
    drop(f.surface);
    drop(texture);

    assert!(errors.lock().is_empty());
}

#[test]
fn drop_surface_before_surface_texture_during_unwind() {
    let f = fixture();
    let errors = register_uncaptured_error_handler(&f.device);

    let texture = acquire(&f.surface);
    drop(f.surface);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _texture = texture;
        panic!("unwind with a live surface texture");
    }));

    assert!(result.is_err());
    assert!(errors.lock().is_empty());
}

#[test]
fn present_then_drop_surface() {
    let f = fixture();
    let errors = register_uncaptured_error_handler(&f.device);

    f.queue.present(acquire(&f.surface));
    drop(f.surface);

    assert!(errors.lock().is_empty());
}

#[test]
fn present_after_surface_dropped() {
    let f = fixture();

    let errors = register_uncaptured_error_handler(&f.device);

    let texture = acquire(&f.surface);
    drop(f.surface);
    f.queue.present(texture);

    assert!(errors.lock().is_empty());
}

#[test]
fn create_noop_surface_requires_noop_backend() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::empty(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    assert!(instance.create_noop_surface().is_err());
}
