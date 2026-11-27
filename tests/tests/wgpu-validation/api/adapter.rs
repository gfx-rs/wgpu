use wgpu::*;

#[test]
fn request_device_returns_device_with_core_features_and_limits() {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::NOOP,
        backend_options: BackendOptions {
            noop: NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .unwrap();

    let (device, _queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        required_features: Features::empty(),
        required_limits: Limits::downlevel_webgl2_defaults(),
        ..Default::default()
    }))
    .unwrap();

    let mut default_limits = Limits::default();
    default_limits.zero_native_only();
    default_limits = default_limits.or_better_values_from(&Limits::downlevel_webgl2_defaults());

    assert_eq!(device.limits(), default_limits);
    assert_eq!(device.features(), Features::CORE_FEATURES_AND_LIMITS);
}
