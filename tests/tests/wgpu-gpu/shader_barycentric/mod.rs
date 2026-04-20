use wgpu::util::DeviceExt;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<wgpu_test::GpuTestInitializer>) {
    vec.push(BARYCENTRIC);
    vec.push(BARYCENTRIC_NO_PERSPECTIVE);
}

//
// This test renders one triangle to a 2x2 render target. The triangle
// covers the bottom-left, bottom-right, and the top-left pixel.
// XY layout of the render target, with the triangle:
//
//     (-1,1)  (0,1)  (1,1)
//        +------+------+
//        |      |      |
//        |   o  |      |
//        |   |\ |      |
//        |   | \|      |
// (-1,0) +---|--\------+ (1,0)
//        |   |  |\     |
//        |   |  | \    |
//        |   o--+--o   |
//        |      |      |
//        +------+------+
//     (-1,-1) (0,-1) (1,-1)
//
// The fragment shader outputs color based on builtin(barycentric):
//
//     return vec4<f32>(bary * 1.1 - 0.05, 1.0);
//

#[gpu_test]
static BARYCENTRIC: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::SHADER_BARYCENTRICS),
    )
    .run_async(|ctx| barycentric(ctx, false));

#[gpu_test]
static BARYCENTRIC_NO_PERSPECTIVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::SHADER_BARYCENTRICS),
    )
    .run_async(|ctx| barycentric(ctx, true));

async fn barycentric(ctx: TestingContext, no_perspective: bool) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("barycentric.wgsl"));

    let n = -0.505;
    let p = 0.51;
    let triangle_xy: [f32; 6] = [n, n, p, n, n, p];
    let vertex_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&triangle_xy),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

    let indices = [0u32, 1, 2];
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
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: if no_perspective {
                    Some("fs_main_no_perspective")
                } else {
                    Some("fs_main")
                },
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
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
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw(0..3, 0..1);
    }
    readback_buffer.copy_from(&ctx.device, &mut encoder, &color_texture);
    ctx.queue.submit(Some(encoder.finish()));

    //
    //   +-----+-----+
    //   |blue |white|
    //   +-----+-----+
    //   | red |green|
    //   +-----+-----+
    //
    let expected = [
        0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 255, 0, 255,
    ];
    readback_buffer
        .assert_buffer_contents(&ctx, &expected)
        .await;
}
