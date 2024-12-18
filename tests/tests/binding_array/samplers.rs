use std::num::{NonZeroU32, NonZeroU64};

use wgpu::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static BINDING_ARRAY_SAMPLERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            )
            .limits(Limits {
                max_samplers_per_shader_stage: 2,
                ..Limits::default()
            }),
    )
    .run_async(|ctx| async move { binding_array_samplers(ctx, false).await });

#[gpu_test]
static PARTIAL_BINDING_ARRAY_SAMPLERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | Features::PARTIALLY_BOUND_BINDING_ARRAY,
            )
            .limits(Limits {
                max_samplers_per_shader_stage: 4,
                ..Limits::default()
            }),
    )
    .run_async(|ctx| async move { binding_array_samplers(ctx, true).await });

async fn binding_array_samplers(ctx: TestingContext, partially_bound: bool) {
    let shader = r#"
        @group(0) @binding(0)
        var samplers: binding_array<sampler>;
        @group(0) @binding(1)
        var texture: texture_2d<f32>;
        @group(0) @binding(2)
        var<storage, read_write> output_values: array<u32>;

        @compute
        @workgroup_size(2, 1, 1)
        fn compMain(@builtin(global_invocation_id) id: vec3u) {
            output_values[id.x] = pack4x8unorm(textureSampleLevel(texture, samplers[id.x], vec2f(0.25 + (0.5 * 0.25), 0.5), 0.0));
        }
    "#;

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binding Array Texture"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

    let input_image: [u8; 8] = [
        255, 0, 0, 255, //
        0, 255, 0, 255, //
    ];

    let expected_output: [u8; 8] = [
        191, 64, 0, 255, //
        255, 0, 0, 255, //
    ];

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 2,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    ctx.queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &input_image,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        Extent3d {
            width: 2,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    let input_view = texture.create_view(&TextureViewDescriptor::default());

    let samplers = [
        ctx.device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1000.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }),
        ctx.device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1000.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }),
    ];

    let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 4 * 2,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let multiplier = if partially_bound { 2 } else { 1 };

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: Some(NonZeroU32::new(2 * multiplier).unwrap()),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });

    let sampler_references: Vec<_> = samplers.iter().collect();

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::SamplerArray(&sampler_references),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&input_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("compMain"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut render_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.dispatch_workgroups(1, 1, 1);
    }

    let readback_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 4 * 2,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, 4 * 2);

    ctx.queue.submit(Some(encoder.finish()));

    readback_buffer.slice(..).map_async(MapMode::Read, |_| {});
    ctx.device.poll(Maintain::Wait);

    let readback_buffer_slice = readback_buffer.slice(..).get_mapped_range();

    assert_eq!(&readback_buffer_slice[0..8], &expected_output[..]);
}
