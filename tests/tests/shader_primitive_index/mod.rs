use wgpu::util::DeviceExt;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

//
// These tests render two triangles to a 2x2 render target. The first triangle
// in the vertex buffer covers the bottom-left pixel, the second triangle
// covers the top-right pixel.
// XY layout of the render target, with two triangles:
//
//     (-1,1)   (0,1)   (1,1)
//        +-------+-------+
//        |       |   o   |
//        |       |  / \  |
//        |       | /   \ |
//        |       |o-----o|
// (-1,0) +-------+-------+ (1,0)
//        |   o   |       |
//        |  / \  |       |
//        | /   \ |       |
//        |o-----o|       |
//        +-------+-------+
//     (-1,-1)  (0,-1)  (1,-1)
//
//
// The fragment shader outputs color based on builtin(primitive_index):
//
//         if ((index % 2u) == 0u) {
//             return vec4<f32>(1.0, 0.0, 0.0, 1.0);
//         } else {
//             return vec4<f32>(0.0, 0.0, 1.0, 1.0);
//         }
//
// draw() renders directly from the vertex buffer: the first (bottom-left)
// triangle is colored red, the other one (top-right) will be blue.
// draw_indexed() draws the triangles in the opposite order, using index
// buffer [3, 4, 5, 0, 1, 2]. This also swaps the resulting pixel colors.
//

#[gpu_test]
static DRAW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::SHADER_PRIMITIVE_INDEX),
    )
    .run_async(|ctx| async move {
        //
        //   +-----+-----+
        //   |white|blue |
        //   +-----+-----+
        //   | red |white|
        //   +-----+-----+
        //
        let expected = [
            255, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255,
        ];
        pulling_common(ctx, &expected, |rpass| {
            rpass.draw(0..6, 0..1);
        })
        .await;
    });

#[gpu_test]
static DRAW_INDEXED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::SHADER_PRIMITIVE_INDEX),
    )
    .run_async(|ctx| async move {
        //
        //   +-----+-----+
        //   |white| red |
        //   +-----+-----+
        //   |blue |white|
        //   +-----+-----+
        //
        let expected = [
            255, 255, 255, 255, 255, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ];
        pulling_common(ctx, &expected, |rpass| {
            rpass.draw_indexed(0..6, 0, 0..1);
        })
        .await;
    });

async fn pulling_common(
    ctx: TestingContext,
    expected: &[u8],
    draw_command: impl FnOnce(&mut wgpu::RenderPass<'_>),
) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("primitive_index.wgsl"));

    let two_triangles_xy: [f32; 12] = [
        -1.0, -1.0, 0.0, -1.0, -0.5, 0.0, // left triangle, negative x, negative y
        0.0, 0.0, 1.0, 0.0, 0.5, 1.0, // right triangle, positive x, positive y
    ];
    let vertex_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&two_triangles_xy),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

    let indices = [3u32, 4, 5, 0, 1, 2]; // index buffer flips triangle order
    let index_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

    let width = 2;
    let height = 2;
    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let color_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let readback_buffer = wgpu_test::image::ReadbackBuffers::new(&ctx.device, &color_texture);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &color_view,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        draw_command(&mut rpass);
    }
    readback_buffer.copy_from(&ctx.device, &mut encoder, &color_texture);
    ctx.queue.submit(Some(encoder.finish()));
    readback_buffer.assert_buffer_contents(&ctx, expected).await;
}
