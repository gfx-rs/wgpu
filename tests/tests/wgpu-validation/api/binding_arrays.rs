use std::num::NonZeroU32;

use wgpu::*;
use wgpu_test::fail;

#[test]
fn dynamic_offset() {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor {
        required_features: Features::TEXTURE_BINDING_ARRAY,
        required_limits: Limits {
            max_binding_array_elements_per_shader_stage: 4,
            ..Limits::default()
        },
        ..DeviceDescriptor::default()
    });

    // Check that you can't create a bind group with both dynamic offset and binding array
    fail(
        &device,
        || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Test1"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: Some(NonZeroU32::new(4).unwrap()),
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        },
        Some("binding array and a dynamically offset buffer"),
    );
}

#[test]
fn uniform_buffer() {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor {
        required_features: Features::TEXTURE_BINDING_ARRAY,
        required_limits: Limits {
            max_binding_array_elements_per_shader_stage: 4,
            ..Limits::default()
        },
        ..DeviceDescriptor::default()
    });

    // Check that you can't create a bind group with both uniform buffer and binding array
    fail(
        &device,
        || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Test2"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: Some(NonZeroU32::new(4).unwrap()),
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        },
        Some("binding array and a uniform buffer"),
    );
}
