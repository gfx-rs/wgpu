//! Tests for image atomics.

use wgpu::ShaderModuleDescriptor;
use wgpu_test::{
    fail, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, TestParameters, TestingContext,
};

#[gpu_test]
static IMAGE_64_ATOMICS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits {
                max_storage_textures_per_shader_stage: 1,
                max_compute_invocations_per_workgroup: 64,
                max_compute_workgroup_size_x: 4,
                max_compute_workgroup_size_y: 4,
                max_compute_workgroup_size_z: 4,
                max_compute_workgroups_per_dimension: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
                ..wgpu::Limits::downlevel_webgl2_defaults()
            })
            .features(
                wgpu::Features::TEXTURE_ATOMIC
                    | wgpu::Features::TEXTURE_INT64_ATOMIC
                    | wgpu::Features::SHADER_INT64,
            ),
    )
    .run_async(|ctx| async move {
        test_format(
            ctx,
            wgpu::TextureFormat::R64Uint,
            wgpu::include_wgsl!("image_64_atomics.wgsl"),
        )
        .await;
    });

#[gpu_test]
static IMAGE_32_ATOMICS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits {
                max_storage_textures_per_shader_stage: 1,
                max_compute_invocations_per_workgroup: 64,
                max_compute_workgroup_size_x: 4,
                max_compute_workgroup_size_y: 4,
                max_compute_workgroup_size_z: 4,
                max_compute_workgroups_per_dimension: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
                ..wgpu::Limits::downlevel_webgl2_defaults()
            })
            .features(wgpu::Features::TEXTURE_ATOMIC),
    )
    .run_async(|ctx| async move {
        test_format(
            ctx,
            wgpu::TextureFormat::R32Uint,
            wgpu::include_wgsl!("image_32_atomics.wgsl"),
        )
        .await;
    });

async fn test_format(
    ctx: TestingContext,
    format: wgpu::TextureFormat,
    desc: ShaderModuleDescriptor<'_>,
) {
    let pixel_bytes = format.target_pixel_byte_cost().unwrap();
    let size = wgpu::Extent3d {
        width: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        height: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        depth_or_array_layers: 1,
    };
    let bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::Atomic,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    };

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bind_group_layout_entry],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    let shader = ctx.device.create_shader_module(desc);
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
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::STORAGE_ATOMIC
            | wgpu::TextureUsages::COPY_SRC,
        mip_level_count: 1,
        sample_count: 1,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        format: Some(format),
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
    rpass.dispatch_workgroups(size.width, size.height, 1);
    drop(rpass);

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &tex);
    readback_buffers.copy_from(&ctx.device, &mut encoder, &tex);

    ctx.queue.submit([encoder.finish()]);

    let padding = [0].repeat(pixel_bytes as usize - size_of::<u32>());
    let data: Vec<u8> = (0..size.width as usize * size.height as usize)
        .flat_map(|i| {
            let x = i as u32 % size.width;
            let y = i as u32 / size.width;
            [bytemuck::bytes_of(&u32::min(x, y)), &padding].concat()
        })
        .collect();

    readback_buffers.assert_buffer_contents(&ctx, &data).await;
}

#[gpu_test]
static IMAGE_ATOMICS_NOT_ENABLED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        let size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };

        fail(
            &ctx.device,
            || {
                let _ = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    dimension: wgpu::TextureDimension::D2,
                    size,
                    format: wgpu::TextureFormat::R32Uint,
                    usage: wgpu::TextureUsages::STORAGE_ATOMIC,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[],
                });
            },
            Some("Texture usages TextureUsages(STORAGE_ATOMIC) are not allowed on a texture of type R32Uint"),
        );
    });

#[gpu_test]
static IMAGE_ATOMICS_NOT_SUPPORTED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_ATOMIC))
    .run_sync(|ctx| {
        let size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };

        fail(
            &ctx.device,
            || {
                let _ = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    dimension: wgpu::TextureDimension::D2,
                    size,
                    format: wgpu::TextureFormat::R8Uint,
                    usage: wgpu::TextureUsages::STORAGE_ATOMIC,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[],
                });
            },
            Some("Texture usages TextureUsages(STORAGE_ATOMIC) are not allowed on a texture of type R8Uint"),
        );
    });
