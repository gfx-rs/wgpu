//! Tests of the [`wgpu`] library API that are not run against a particular GPU.

mod api;
mod noop;

/// Obtain a device using [`wgpu::Backend::Noop`].
/// This should never fail.
fn request_noop_device() -> (wgpu::Device, wgpu::Queue) {
    request_noop_device_with_desc(&wgpu::DeviceDescriptor::default())
}

fn request_noop_device_with_desc(desc: &wgpu::DeviceDescriptor) -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions { enable: true },
            ..Default::default()
        },
        ..Default::default()
    });

    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("adapter");
    assert_eq!(adapter.get_info().backend, wgpu::Backend::Noop);

    pollster::block_on(adapter.request_device(desc)).expect("device")
}
