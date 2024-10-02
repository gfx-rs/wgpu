//! Tests for image atomics.

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static IMAGE_ATOMICS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgt::Limits {
                max_storage_textures_per_shader_stage: 1,
                max_compute_invocations_per_workgroup: 64,
                max_compute_workgroup_size_x: 4,
                max_compute_workgroup_size_y: 4,
                max_compute_workgroup_size_z: 4,
                max_compute_workgroups_per_dimension: 64,
                ..wgt::Limits::downlevel_webgl2_defaults()
            })
            .features(
                wgpu::Features::TEXTURE_INT64_ATOMIC
                    | wgpu::Features::SHADER_INT64
                    | wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            ),
    )
    .run_sync(|ctx| {
        let size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };
        let bind_group_layout_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::R64Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }];

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bind_group_layout_entries,
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("image_atomics.wgsl"));
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("image atomics pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::R64Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::SHADER_ATOMIC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R64Uint),
            aspect: wgpu::TextureAspect::All,
            ..wgpu::TextureViewDescriptor::default()
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut rpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, Some(&bind_group), &[]);
        rpass.dispatch_workgroups(1, 1, 1);
        drop(rpass);
        ctx.queue.submit(Some(encoder.finish()));
    });
