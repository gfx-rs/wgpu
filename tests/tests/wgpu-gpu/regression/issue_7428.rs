use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(TEXTURE_BINDING_VIEW_DIM_D2_ARRAY);
}

/// On the GLES backend, if you need a texture view in a view dimension other than what is inferred,
/// you must pass texture_binding_view_dimension. This tests that this happens on all backends properly.
#[apply(gpu_test!)]
static TEXTURE_BINDING_VIEW_DIM_D2_ARRAY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(wgpu::Limits {
                max_storage_buffers_per_shader_stage: 1,
                ..wgpu::Limits::downlevel_defaults()
            }),
    )
    .run_async(|ctx| async move { test_impl(&ctx).await });

async fn test_impl(ctx: &TestingContext) {
    const TEST_PIXEL: [u8; 4] = [255, 0, 0, 255];
    const OUTPUT_SIZE: u64 = 4 * 4; // vec4f

    let texture_d2_array = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
        texture_binding_view_dimension: Some(wgpu::TextureViewDimension::D2Array),
    });

    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture_d2_array,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &TEST_PIXEL,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        texture_d2_array.size(),
    );

    let texture_view = texture_d2_array.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(1),
        base_array_layer: 0,
        ..Default::default()
    });

    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: OUTPUT_SIZE,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: OUTPUT_SIZE,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("issue_7428.wgsl"));

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, OUTPUT_SIZE);

    ctx.queue.submit(Some(encoder.finish()));

    readback_buf.map_async(wgpu::MapMode::Read, .., Result::unwrap);

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let result = readback_buf.get_mapped_range(..).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, f32>(&result),
        &[1.0, 0.0, 0.0, 1.0],
    );
}
