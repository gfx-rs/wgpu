use std::num::NonZeroU32;

use wgpu::*;
use wgpu_test::{
    fail, gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext,
};

#[gpu_test]
static VALIDATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::TEXTURE_BINDING_ARRAY)
            .limits(Limits {
                max_dynamic_storage_buffers_per_pipeline_layout: 1,
                ..Limits::downlevel_defaults()
            })
            .expect_fail(
                // https://github.com/gfx-rs/wgpu/issues/6950
                FailureCase::backend(Backends::VULKAN).validation_error("has not been destroyed"),
            ),
    )
    .run_async(validation);

async fn validation(ctx: TestingContext) {
    // Check that you can't create a bind group with both dynamic offset and binding array
    fail(
        &ctx.device,
        || {
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
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

    // Check that you can't create a bind group with both uniform buffer and binding array
    fail(
        &ctx.device,
        || {
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
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
