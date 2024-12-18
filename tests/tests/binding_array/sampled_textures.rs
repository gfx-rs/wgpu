use std::num::NonZeroU32;

use wgpu::*;
use wgpu_test::{
    gpu_test, image::ReadbackBuffers, GpuTestConfiguration, TestParameters, TestingContext,
};

#[gpu_test]
static BINDING_ARRAY_SAMPLED_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            )
            .limits(Limits {
                max_sampled_textures_per_shader_stage: 16,
                ..Limits::default()
            }),
    )
    .run_async(|ctx| async move { binding_array_sampled_textures(ctx, false).await });

#[gpu_test]
static PARTIAL_BINDING_ARRAY_SAMPLED_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | Features::PARTIALLY_BOUND_BINDING_ARRAY,
            )
            .limits(Limits {
                max_sampled_textures_per_shader_stage: 32,
                ..Limits::default()
            }),
    )
    .run_async(|ctx| async move { binding_array_sampled_textures(ctx, false).await });

/// Test to see how texture bindings array work and additionally making sure
/// that non-uniform indexing is working correctly.
///
/// If non-uniform indexing is not working correctly, AMD will produce the wrong
/// output due to non-native support for non-uniform indexing within a WARP.
async fn binding_array_sampled_textures(ctx: TestingContext, partially_bound: bool) {
    let shader = r#"
        @group(0) @binding(0)
        var textures: binding_array<texture_2d<f32>>;

        @vertex
        fn vertMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(
                vec2f(-1.0, -1.0),
                vec2f(3.0, -1.0),
                vec2f(-1.0, 3.0)
            );

            return vec4<f32>(positions[id], 0.0, 1.0);
        }

        @fragment
        fn fragMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
            let pixel = vec2u(floor(pos.xy));
            let index = pixel.y * 4 + pixel.x;

            return textureLoad(textures[index], vec2u(0), 0);
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
            format: TextureFormat::Rgba8UnormSrgb,
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
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let output_view = output_texture.create_view(&TextureViewDescriptor::default());

    let count = if partially_bound { 32 } else { 16 };

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(NonZeroU32::new(count).unwrap()),
            }],
        });

    let input_view_references: Vec<_> = input_views.iter().collect();

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
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vertMain"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fragMain"),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            cache: None,
            multiview: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &output_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &output_texture);
    readback_buffers.copy_from(&ctx.device, &mut encoder, &output_texture);

    ctx.queue.submit(Some(encoder.finish()));

    readback_buffers.assert_buffer_contents(&ctx, &image).await;
}
