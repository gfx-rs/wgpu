use std::num::NonZeroU32;

use wgpu_test::{gpu_test, image::ReadbackBuffers, GpuTestConfiguration, TestParameters};

#[gpu_test]
static PARTIALLY_BOUNDED_ARRAY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            )
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        let device = &ctx.device;

        let texture_extent = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let texture_view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },

                count: NonZeroU32::new(4),
            }],
        });

        let cs_module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&[&texture_view]),
            }],
            layout: &bind_group_layout,
            label: Some("bind group"),
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        let readback_buffers = ReadbackBuffers::new(&ctx.device, &storage_texture);
        readback_buffers.copy_from(&ctx.device, &mut encoder, &storage_texture);

        ctx.queue.submit(Some(encoder.finish()));

        readback_buffers
            .assert_buffer_contents(&ctx, bytemuck::bytes_of(&[4.0f32, 3.0, 2.0, 1.0]))
            .await;
    });
