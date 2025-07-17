//! D3D12 samplers are fun and we're doing a decent amount of polyfilling with them.
//!
//! Do some tests to ensure things are working correctly and nothing gets mad.

use wgpu_test::{did_oom, gpu_test, valid, GpuTestConfiguration, TestParameters, TestingContext};

// A number large enough to likely cause sampler caches to run out of space
// on some devices.
const PROBABLY_PROBLEMATIC_SAMPLER_COUNT: u32 = 8 * 1024;

#[gpu_test]
static SAMPLER_DEDUPLICATION: GpuTestConfiguration =
    GpuTestConfiguration::new().run_sync(sampler_deduplication);

// Create a large number of samplers from the same two descriptors.
//
// Sampler deduplication in the backend should ensure this doesn't cause any issues.
fn sampler_deduplication(ctx: TestingContext) {
    // Create 2 different sampler descriptors
    let desc1 = wgpu::SamplerDescriptor {
        label: Some("sampler1"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    };

    let desc2 = wgpu::SamplerDescriptor {
        label: Some("sampler2"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    };

    // Now create a bunch of samplers with these descriptors
    let samplers = (0..PROBABLY_PROBLEMATIC_SAMPLER_COUNT)
        .map(|i| {
            let desc = if i % 2 == 0 { &desc1 } else { &desc2 };
            valid(&ctx.device, || ctx.device.create_sampler(desc))
        })
        .collect::<Vec<_>>();

    drop(samplers);
}

#[gpu_test]
static SAMPLER_CREATION_FAILURE: GpuTestConfiguration =
    GpuTestConfiguration::new().run_sync(sampler_creation_failure);

/// We want to test that sampler creation properly fails when we hit internal sampler
/// cache limits. As we don't actually know what the limit is, we first create as many
/// samplers as we can until we get the first failure.
///
/// This failure being caught ensures that the error catching machinery on samplers
/// is working as expected.
///
/// We then clear all samplers and poll the device, which should leave the caches
/// completely empty.
///
/// We then try to create the same number of samplers to ensure the cache was entirely
/// cleared.
fn sampler_creation_failure(ctx: TestingContext) {
    let desc = wgpu::SamplerDescriptor {
        label: Some("sampler1"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    };

    let mut sampler_storage = Vec::with_capacity(PROBABLY_PROBLEMATIC_SAMPLER_COUNT as usize);

    for i in 0..PROBABLY_PROBLEMATIC_SAMPLER_COUNT {
        let (failed, sampler) = did_oom(&ctx.device, || {
            ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                lod_min_clamp: i as f32 * 0.01,
                ..desc
            })
        });

        if failed {
            break;
        }

        sampler_storage.push(sampler);
    }

    let failed_count = sampler_storage.len();

    sampler_storage.clear();
    ctx.device.poll(wgpu::PollType::Wait).unwrap();

    for i in 0..failed_count {
        valid(&ctx.device, || {
            eprintln!("Trying to create sampler {i}");
            let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                lod_min_clamp: i as f32 * 0.01,
                // Change the max clamp to ensure the sampler is using different cache slots from
                // the previous run.
                lod_max_clamp: 200.0,
                ..desc
            });
            sampler_storage.push(sampler);
        });
    }
}

const SINGLE_GROUP_BINDINGS: &str = r#"
@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var sampler0: sampler;
@group(0) @binding(2) var sampler1: sampler;
@group(0) @binding(3) var sampler2: sampler;

@group(1) @binding(0) var<storage, read_write> results: array<vec4f, 3>;
"#;

const MULTI_GROUP_BINDINGS: &str = r#"
@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var sampler0: sampler;
@group(1) @binding(0) var sampler1: sampler;
@group(2) @binding(0) var sampler2: sampler;

@group(3) @binding(0) var<storage, read_write> results: array<vec4f, 3>;
"#;

const SAMPLER_CODE: &str = r#"
@compute @workgroup_size(1, 1, 1)
fn cs_main() {
    // When sampling a 2x2 texture at the bottom left, we can change the address mode
    // on S/T to get different values. This allows us to make sure the right sampler
    // is being used.
    results[0] = textureSampleLevel(texture, sampler0, vec2f(0.0, 1.0), 0.0);
    results[1] = textureSampleLevel(texture, sampler1, vec2f(0.0, 1.0), 0.0);
    results[2] = textureSampleLevel(texture, sampler2, vec2f(0.0, 1.0), 0.0);
}
"#;

enum GroupType {
    Single,
    Multi,
}

#[gpu_test]
static SAMPLER_SINGLE_BIND_GROUP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            // In OpenGL textures cannot be used with multiple samplers.
            .skip(wgpu_test::FailureCase::backend(wgpu::Backends::GL)),
    )
    .run_sync(|ctx| sampler_bind_group(ctx, GroupType::Single));

#[gpu_test]
static SAMPLER_MULTI_BIND_GROUP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            // In OpenGL textures cannot be used with multiple samplers.
            .skip(wgpu_test::FailureCase::backend(wgpu::Backends::GL)),
    )
    .run_sync(|ctx| sampler_bind_group(ctx, GroupType::Multi));

fn sampler_bind_group(ctx: TestingContext, group_type: GroupType) {
    let bindings = match group_type {
        GroupType::Single => SINGLE_GROUP_BINDINGS,
        GroupType::Multi => MULTI_GROUP_BINDINGS,
    };

    let full_shader = format!("{bindings}\n{SAMPLER_CODE}");

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            source: wgpu::ShaderSource::Wgsl(full_shader.into()),
            label: None,
        });

    let mut bind_group_layouts = Vec::new();

    match group_type {
        GroupType::Single => {
            let bgl = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("combination_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

            bind_group_layouts.push(bgl);
        }
        GroupType::Multi => {
            let bgl0 = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("multiple_bgl0"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

            let bgl1 = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("multiple_bgl1"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    }],
                });

            let bgl2 = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("multiple_bgl2"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    }],
                });

            bind_group_layouts.push(bgl0);
            bind_group_layouts.push(bgl1);
            bind_group_layouts.push(bgl2);
        }
    }

    let output_bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("output_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let mut bgl_references: Vec<_> = bind_group_layouts.iter().collect();

    bgl_references.push(&output_bgl);

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &bgl_references,
            push_constant_ranges: &[],
        });

    let input_image = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("input_image"),
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let input_image_view = input_image.create_view(&wgpu::TextureViewDescriptor::default());

    let image_data: [u8; 16] = [
        255, 0, 0, 255, /* */ 0, 255, 0, 255, //
        0, 0, 255, 255, /* */ 255, 255, 255, 255, //
    ];

    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &input_image,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &image_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
    );

    let address_modes = [
        (
            wgpu::AddressMode::ClampToEdge,
            wgpu::AddressMode::ClampToEdge,
        ),
        (wgpu::AddressMode::Repeat, wgpu::AddressMode::ClampToEdge),
        (wgpu::AddressMode::ClampToEdge, wgpu::AddressMode::Repeat),
    ];

    let samplers = address_modes.map(|(address_mode_u, address_mode_v)| {
        ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u,
            address_mode_v,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        })
    });

    let mut bind_groups = Vec::new();

    match group_type {
        GroupType::Single => {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("combination_bg"),
                layout: &bind_group_layouts[0],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_image_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&samplers[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&samplers[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&samplers[2]),
                    },
                ],
            });

            bind_groups.push(bg);
        }
        GroupType::Multi => {
            let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("multiple_bg0"),
                layout: &bind_group_layouts[0],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_image_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&samplers[0]),
                    },
                ],
            });

            let bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("multiple_bg1"),
                layout: &bind_group_layouts[1],
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&samplers[1]),
                }],
            });

            let bg2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("multiple_bg2"),
                layout: &bind_group_layouts[2],
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&samplers[2]),
                }],
            });

            bind_groups.push(bg0);
            bind_groups.push(bg1);
            bind_groups.push(bg2);
        }
    }

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_buffer"),
        size: 48,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let transfer_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("transfer_buffer"),
        size: 48,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let output_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("output_bg"),
        layout: &output_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &output_buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    let mut bg_references = bind_groups.iter().collect::<Vec<_>>();

    bg_references.push(&output_bg);

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            cache: None,
            compilation_options: Default::default(),
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        for (i, &bg) in bg_references.iter().enumerate() {
            cpass.set_bind_group(i as u32, bg, &[]);
        }
        cpass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &transfer_buffer, 0, 48);

    ctx.queue.submit([encoder.finish()]);
    let buffer_slice = transfer_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device.poll(wgpu::PollType::Wait).unwrap();

    let buffer_data = buffer_slice.get_mapped_range();

    let f32_buffer: &[f32] = bytemuck::cast_slice(&buffer_data);

    let correct_values: [f32; 12] = [
        0.0, 0.0, 1.0, 1.0, //
        0.5, 0.5, 1.0, 1.0, //
        0.5, 0.0, 0.5, 1.0, //
    ];
    let iter = f32_buffer.iter().zip(correct_values.iter());
    for (&result, &value) in iter {
        approx::assert_relative_eq!(result, value, max_relative = 0.02);
    }
}
