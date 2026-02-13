fn noop_adapter() -> wgpu::Adapter {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("noop backend adapter absent when it should be")
}

#[test]
fn request_no_experimental_features() {
    let adapter = noop_adapter();

    let dq = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        // Not experimental
        required_features: wgpu::Features::FLOAT32_FILTERABLE,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        ..Default::default()
    }));

    assert!(dq.is_ok());
}

#[test]
fn request_experimental_features() {
    let adapter = noop_adapter();

    let dq = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        // Experimental
        required_features: wgpu::Features::EXPERIMENTAL_MESH_SHADER,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }));

    assert!(dq.is_ok());
}

#[test]
fn request_experimental_features_when_not_enabled() {
    let adapter = noop_adapter();

    let dq = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        // Experimental
        required_features: wgpu::Features::EXPERIMENTAL_MESH_SHADER,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        ..Default::default()
    }));

    assert!(dq.is_err());
}

#[test]
fn request_multiple_experimental_features_when_not_enabled() {
    let adapter = noop_adapter();

    let dq = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        // Experimental
        required_features: wgpu::Features::EXPERIMENTAL_MESH_SHADER
            | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        ..Default::default()
    }));

    assert!(dq.is_err());
}
