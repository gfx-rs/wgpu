use wgpu::util::DeviceExt;
use wgpu_test::{
    gpu_test, image::ReadbackBuffers, GpuTestConfiguration, TestParameters, TestingContext,
};

/// We thought we had an OpenGL bug that, when running without explicit in-shader locations,
/// we will not properly bind uniform buffers to both the vertex and fragment
/// shaders. This turned out to not reproduce at all with this test case.
///
/// However, it also caught issues with the push constant implementation,
/// making sure that it works correctly with different definitions for the push constant
/// block in vertex and fragment shaders.
///
/// This test needs to be able to run on GLES 3.0
///
/// What this test does is render a 2x2 texture. Each pixel corresponds to a different
/// data source.
///
/// top left: Vertex Shader / Uniform Buffer
/// top right: Vertex Shader / Push Constant
/// bottom left: Fragment Shader / Uniform Buffer
/// bottom right: Fragment Shader / Push Constant
///
/// We then validate the data is correct from every position.
#[gpu_test]
static MULTI_STAGE_DATA_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::PUSH_CONSTANTS)
            .limits(wgpu::Limits {
                max_push_constant_size: 16,
                ..Default::default()
            }),
    )
    .run_async(multi_stage_data_binding_test);

async fn multi_stage_data_binding_test(ctx: TestingContext) {
    // We use different shader modules to allow us to use different
    // types for the uniform and push constant blocks between stages.
    let vs_sm = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("issue_3349.vs.wgsl"));

    let fs_sm = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("issue_3349.fs.wgsl"));

    // We start with u8s then convert to float, to make sure we don't have
    // cross-vendor rounding issues unorm.
    let input_as_unorm: [u8; 4] = [25_u8, 50, 75, 100];
    let input = input_as_unorm.map(|v| v as f32 / 255.0);

    let buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let pll = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pll"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                range: 0..16,
            }],
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pll),
            vertex: wgpu::VertexState {
                module: &vs_sm,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_sm,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture"),
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Important: NOT srgb.
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rpass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bg, &[]);
        rpass.set_push_constants(
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            0,
            bytemuck::cast_slice(&input),
        );
        rpass.draw(0..3, 0..1);
    }

    let buffers = ReadbackBuffers::new(&ctx.device, &texture);
    buffers.copy_from(&ctx.device, &mut encoder, &texture);
    ctx.queue.submit([encoder.finish()]);

    let result = input_as_unorm.repeat(4);
    buffers.assert_buffer_contents(&ctx, &result).await;
}
