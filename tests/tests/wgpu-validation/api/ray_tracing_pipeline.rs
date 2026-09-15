use wgpu::*;
use wgpu_test::fail;

fn check_visibility_requires_ray_tracing_pipelines_feature(visibility: ShaderStages) {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor::default());

    fail(
        &device,
        || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ray tracing pipeline visibility"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            })
        },
        Some("EXPERIMENTAL_RAY_TRACING_PIPELINES"),
    );
}

#[test]
fn ray_generation_visibility_requires_ray_tracing_pipelines_feature() {
    check_visibility_requires_ray_tracing_pipelines_feature(ShaderStages::RAY_GENERATION);
}

#[test]
fn any_hit_visibility_requires_ray_tracing_pipelines_feature() {
    check_visibility_requires_ray_tracing_pipelines_feature(ShaderStages::ANY_HIT);
}

#[test]
fn closest_hit_visibility_requires_ray_tracing_pipelines_feature() {
    check_visibility_requires_ray_tracing_pipelines_feature(ShaderStages::CLOSEST_HIT);
}

#[test]
fn miss_visibility_requires_ray_tracing_pipelines_feature() {
    check_visibility_requires_ray_tracing_pipelines_feature(ShaderStages::MISS);
}
