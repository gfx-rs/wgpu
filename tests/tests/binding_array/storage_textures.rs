use std::num::NonZeroU32;

use wgpu::*;
use wgpu_test::{
    gpu_test, image::ReadbackBuffers, FailureCase, GpuTestConfiguration, TestParameters,
    TestingContext,
};

#[gpu_test]
static BINDING_ARRAY_STORAGE_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                    | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            )
            .limits(Limits {
                max_storage_textures_per_shader_stage: 17,
                ..Limits::default()
            })
            .expect_fail(FailureCase::backend(Backends::METAL)),
    )
    .run_async(|ctx| async move { binding_array_storage_textures(ctx, false).await });

#[gpu_test]
static PARTIAL_BINDING_ARRAY_STORAGE_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::PARTIALLY_BOUND_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                    | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            )
            .limits(Limits {
                max_storage_textures_per_shader_stage: 33,
                ..Limits::default()
            })
            .expect_fail(FailureCase::backend(Backends::METAL)),
    )
    .run_async(|ctx| async move { binding_array_storage_textures(ctx, true).await });

async fn binding_array_storage_textures(ctx: TestingContext, partially_bound: bool) {
    let shader = r#"
        @group(0) @binding(0)
        var textures: binding_array<texture_storage_2d<rgba8unorm, read_write> >;

        @compute
        @workgroup_size(4, 4, 1)
        fn compMain(@builtin(global_invocation_id) id: vec3u) {
            // Read from the 4x4 textures in 0-15, then write to the 4x4 texture in 16

            let pixel = vec2u(id.xy);
            let index = pixel.y * 4 + pixel.x;

            let color = textureLoad(textures[index], vec2u(0));
            textureStore(textures[16], pixel, color);
        }
    "#;

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binding Array Texture"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

    let image = image::load_from_memory(include_bytes!("../3x3_colors.png")).unwrap();
    // Resize image to 4x4
    let image = image
        .resize_exact(4, 4, image::imageops::FilterType::Gaussian)
        .into_rgba8();

    // Create one texture for each pixel
    let mut input_views = Vec::with_capacity(64);
    for data in image.pixels() {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &data.0,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        input_views.push(texture.create_view(&TextureViewDescriptor::default()));
    }

    let output_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Output Texture"),
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let output_view = output_texture.create_view(&TextureViewDescriptor::default());

    let multiplier = if partially_bound { 2 } else { 1 };

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: Some(NonZeroU32::new(4 * 4 * multiplier + 1).unwrap()),
            }],
        });

    let mut input_view_references: Vec<_> = input_views.iter().collect();
    input_view_references.push(&output_view);

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureViewArray(&input_view_references),
        }],
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

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &output_texture);
    readback_buffers.copy_from(&ctx.device, &mut encoder, &output_texture);

    ctx.queue.submit(Some(encoder.finish()));

    readback_buffers.assert_buffer_contents(&ctx, &image).await;
}
