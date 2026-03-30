//! Limit bucketing tests
//!
//! See [`wgpu_core::limits`].

use wgpu_core as wgc;
use wgpu_types as wgt;

const UPLEVEL_FEATURES: wgt::Features = {
    use wgt::Features;

    Features::DEPTH_CLIP_CONTROL
        .union(Features::DEPTH32FLOAT_STENCIL8)
        .union(Features::TEXTURE_COMPRESSION_BC)
        .union(Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
        .union(Features::TIMESTAMP_QUERY)
        .union(Features::INDIRECT_FIRST_INSTANCE)
        .union(Features::RG11B10UFLOAT_RENDERABLE)
        .union(Features::BGRA8UNORM_STORAGE)
        .union(Features::FLOAT32_FILTERABLE)
        .union(Features::DUAL_SOURCE_BLENDING)
        .union(Features::PRIMITIVE_INDEX)
        .union(Features::SUBGROUP)
        .union(Features::IMMEDIATES)
};

fn create_noop_global(options: wgt::NoopBackendOptions) -> wgc::global::Global {
    wgc::global::Global::new(
        "test",
        wgt::instance::InstanceDescriptor {
            backends: wgt::Backends::NOOP,
            backend_options: wgt::BackendOptions {
                noop: options,
                ..Default::default()
            },
            ..wgt::instance::InstanceDescriptor::new_without_display_handle()
        },
        None,
    )
}

#[test]
fn enabled() {
    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        device_type: Some(wgt::DeviceType::DiscreteGpu),
        features: Some(wgt::Features::empty()),
        subgroup_min_size: Some(4),
        subgroup_max_size: Some(128),
        ..Default::default() // noop defaults to max limits
    });

    let adapter_id = global
        .request_adapter(
            &wgt::RequestAdapterOptions {
                apply_limit_buckets: true,
                ..Default::default()
            },
            wgt::Backends::NOOP,
            None,
        )
        .unwrap();

    let limits = global.adapter_limits(adapter_id);
    let features = global.adapter_features(adapter_id);
    let info = global.adapter_get_info(adapter_id);

    // Max limits should be replaced with "default"
    assert_eq!(limits, wgt::Limits::defaults());
    assert_eq!(features, wgt::Features::empty());
    assert_eq!(info.subgroup_min_size, 4);
    assert_eq!(info.subgroup_max_size, 128);
}

#[test]
fn exempt_features() {
    const EXEMPT_FEATURES: wgt::Features = wgt::Features::EXTERNAL_TEXTURE
        .union(wgt::Features::TEXTURE_FORMAT_NV12)
        .union(wgt::Features::TEXTURE_FORMAT_P010)
        .union(wgt::Features::TEXTURE_FORMAT_16BIT_NORM);

    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        device_type: Some(wgt::DeviceType::DiscreteGpu),
        features: Some(EXEMPT_FEATURES.union(wgt::Features::SUBGROUP)),
        subgroup_min_size: Some(4),
        subgroup_max_size: Some(128),
        ..Default::default() // noop defaults to max limits
    });

    let adapter_id = global
        .request_adapter(
            &wgt::RequestAdapterOptions {
                apply_limit_buckets: true,
                ..Default::default()
            },
            wgt::Backends::NOOP,
            None,
        )
        .unwrap();

    let limits = global.adapter_limits(adapter_id);
    let features = global.adapter_features(adapter_id);
    let info = global.adapter_get_info(adapter_id);

    // Max limits should be replaced with "default" bucket
    assert_eq!(limits, wgt::Limits::defaults());
    assert_eq!(features, EXEMPT_FEATURES);
    assert_eq!(info.subgroup_min_size, 4);
    assert_eq!(info.subgroup_max_size, 128);
}

#[test]
fn limits_below_minimums_returns_no_adapter() {
    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        limits: Some(wgt::Limits {
            max_texture_dimension_2d: 1024,
            ..wgt::Limits::default()
        }),
        device_type: Some(wgt::DeviceType::DiscreteGpu),
        features: Some(wgt::Features::empty()),
        ..Default::default()
    });

    let result = global.request_adapter(
        &wgt::RequestAdapterOptions {
            apply_limit_buckets: true,
            ..Default::default()
        },
        wgt::Backends::NOOP,
        None,
    );

    // Device is below WebGPU minimums, so no bucket matches
    assert!(result.is_err());
}

#[test]
fn device_creation_exceeding_bucket_fails() {
    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        device_type: Some(wgt::DeviceType::DiscreteGpu),
        features: Some(wgt::Features::empty()),
        ..Default::default()
    });

    let adapter_id = global
        .request_adapter(
            &wgt::RequestAdapterOptions {
                apply_limit_buckets: true,
                ..Default::default()
            },
            wgt::Backends::NOOP,
            None,
        )
        .unwrap();

    // The "default" bucket has max_bind_groups = 4
    let result = global.adapter_request_device(
        adapter_id,
        &wgt::DeviceDescriptor {
            required_limits: wgt::Limits {
                max_bind_groups: 8,
                ..wgt::Limits::default()
            },
            ..Default::default()
        },
        None,
        None,
    );

    assert!(result.is_err());
}

#[test]
fn subgroup_sizes_fixed_when_unsupported() {
    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        device_type: Some(wgt::DeviceType::DiscreteGpu),
        features: Some(wgt::Features::empty()),
        subgroup_min_size: Some(64),
        subgroup_max_size: Some(64),
        ..Default::default()
    });

    let adapter_id = global
        .request_adapter(
            &wgt::RequestAdapterOptions {
                apply_limit_buckets: true,
                ..Default::default()
            },
            wgt::Backends::NOOP,
            None,
        )
        .unwrap();

    let info = global.adapter_get_info(adapter_id);
    let features = global.adapter_features(adapter_id);

    // Since the "default" bucket doesn't have the `subgroups` feature:
    //  - bucket match should succeed despite subgroup size range being narrower than bucket,
    //  - subgroup sizes should be replaced with WebGPU's fixed defaults.
    assert!(!features.contains(wgt::Features::SUBGROUP));
    assert_eq!(info.subgroup_min_size, 4);
    assert_eq!(info.subgroup_max_size, 128);
}

#[test]
fn fallback_adapter() {
    // DeviceType::Cpu with empty features should match "fallback" bucket
    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        device_type: Some(wgt::DeviceType::Cpu),
        features: Some(wgt::Features::empty()),
        ..Default::default()
    });

    let adapter_id = global
        .request_adapter(
            &wgt::RequestAdapterOptions {
                apply_limit_buckets: true,
                ..Default::default()
            },
            wgt::Backends::NOOP,
            None,
        )
        .unwrap();

    let info = global.adapter_get_info(adapter_id);

    // Should match fallback bucket which is a CPU/fallback adapter
    assert_eq!(info.device_type, wgt::DeviceType::Cpu);
}

// Subgroup limits are not treated like regular limits (where a device qualifies
// if its max limits meet or exceed the bucket). A device qualifies if its max
// subgroup size is the same or less than the bucket's max, and conversely for
// min. i.e. the device's subgroup size range must be a subset of the bucket's
// subgroup size range.

#[test]
fn subgroup_max_above_bucket() {
    // Construct a device with UPLEVEL_FEATURES and a subgroup max size of 65.
    // This device is disqualified from all tiers besides NO_F16 due to not
    // having f16 support, and disqualified from NO_F16 due to the subgroup max
    // size.
    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        device_type: Some(wgt::DeviceType::DiscreteGpu),
        subgroup_min_size: Some(32),
        subgroup_max_size: Some(65),
        features: Some(UPLEVEL_FEATURES),
        ..Default::default()
    });

    let adapter_id = global
        .request_adapter(
            &wgt::RequestAdapterOptions {
                apply_limit_buckets: true,
                ..Default::default()
            },
            wgt::Backends::NOOP,
            None,
        )
        .unwrap();

    let features = global.adapter_features(adapter_id);

    assert!(!features.contains(wgt::Features::SUBGROUP));
}

#[test]
fn subgroup_min_below_bucket() {
    // The uplevel buckets with smallest subgroup_min_size are M1 (4) and I1/I2 (8). We
    // construct a device that has subgroup_min_size = 7 and max_vertex_attributes = 29, so
    // that it will not qualify for any UPLEVEL bucket (which means it won't get subgroups).
    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        device_type: Some(wgt::DeviceType::DiscreteGpu),
        subgroup_min_size: Some(7),
        subgroup_max_size: Some(32),
        limits: Some(wgt::Limits {
            max_vertex_attributes: 29, // Disqualify from M1 bucket
            ..Default::default()
        }),
        ..Default::default()
    });

    let adapter_id = global
        .request_adapter(
            &wgt::RequestAdapterOptions {
                apply_limit_buckets: true,
                ..Default::default()
            },
            wgt::Backends::NOOP,
            None,
        )
        .unwrap();

    let features = global.adapter_features(adapter_id);

    assert!(!features.contains(wgt::Features::SUBGROUP));
}

#[test]
fn enumerate_adapters_bucketing_enabled() {
    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        device_type: Some(wgt::DeviceType::DiscreteGpu),
        features: Some(wgt::Features::SUBGROUP),
        ..Default::default()
    });

    let adapters = global.enumerate_adapters(wgt::Backends::NOOP, true);
    assert_eq!(adapters.len(), 1);

    let adapter_id = adapters[0];
    let limits = global.adapter_limits(adapter_id);

    // With bucketing, should have bucketed limits
    assert_eq!(limits, wgt::Limits::defaults());
}

#[test]
fn enumerate_adapters_bucketing_disabled() {
    let custom_limits = wgt::Limits {
        max_bind_groups: 99,
        ..wgt::Limits::default()
    };

    let global = create_noop_global(wgt::NoopBackendOptions {
        enable: true,
        limits: Some(custom_limits),
        ..Default::default()
    });

    let adapters = global.enumerate_adapters(wgt::Backends::NOOP, false);
    assert_eq!(adapters.len(), 1);

    let adapter_id = adapters[0];
    let limits = global.adapter_limits(adapter_id);

    // Without bucketing, should have raw limits
    assert_eq!(limits.max_bind_groups, 99);
}
