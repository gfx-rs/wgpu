use wgpu::*;
use wgpu_test::fail;

fn check_visibility_requires_mesh_shader_feature(visibility: ShaderStages) {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor::default());

    fail(
        &device,
        || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("mesh shader visibility"),
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
        Some("EXPERIMENTAL_MESH_SHADER"),
    );
}

#[test]
fn task_visibility_requires_mesh_shader_feature() {
    check_visibility_requires_mesh_shader_feature(ShaderStages::TASK);
}

#[test]
fn mesh_visibility_requires_mesh_shader_feature() {
    check_visibility_requires_mesh_shader_feature(ShaderStages::MESH);
}
