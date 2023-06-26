use wasm_bindgen_test::*;

use wgpu::util::DeviceExt;
use wgpu_test::{image, initialize_test, TestParameters, TestingContext};
use wgt::COPY_BYTES_PER_ROW_ALIGNMENT;

//
// These tests render four triangles to a 64x2 render target. The first and third triangle
// in the vertex buffer covers the bottom-left pixel, the second and forth triangle
// covers the top-right pixel.
// XY layout of the render target, with two triangles:
//
//     (-1,1)   (0,1)   (1,1)
//        +-------+-------+
//        |       |o-----o|
//        |       ||   / ||
//        |       || /   ||
//        |       |o-----o|
// (-1,0) +-------+-------+ (1,0)
//        |o-----o|       |
//        ||   / ||       |
//        || /   ||       |
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
// draw() renders directly from the vertex buffer: the bottom-left
// rect is colored red, the other rect (top-right) will be blue.
// draw_indexed() draws the rects in the opposite order.
// This also swaps the resulting pixel colors.
//

const TEXTURE_HEIGHT: u32 = 2;
const TEXTURE_WIDTH: u32 = 64;
const BUFFER_SIZE: usize = (COPY_BYTES_PER_ROW_ALIGNMENT * TEXTURE_HEIGHT) as usize;

#[test]
#[wasm_bindgen_test]
fn draw() {
    //
    //   +-----+-----+
    //   |white|blue |
    //   +-----+-----+
    //   | red |white|
    //   +-----+-----+
    //
    let mut expected: [u8; BUFFER_SIZE] = [255; BUFFER_SIZE];
    expected[(BUFFER_SIZE / 4)..][..BUFFER_SIZE / 4].copy_from_slice(
        &(std::iter::repeat([0, 0, 255, 255])
            .flatten()
            .take(BUFFER_SIZE / 4)
            .collect::<Vec<u8>>()),
    );
    expected[((2 * BUFFER_SIZE) / 4)..][..BUFFER_SIZE / 4].copy_from_slice(
        &(std::iter::repeat([255, 0, 0, 255])
            .flatten()
            .take(BUFFER_SIZE / 4)
            .collect::<Vec<u8>>()),
    );
    println!("dasndkas");
    initialize_test(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::SHADER_PRIMITIVE_INDEX),
        |ctx| {
            pulling_common(ctx, &expected, |rpass| {
                rpass.draw(0..12, 0..1);
            })
        },
    );
}

#[test]
#[wasm_bindgen_test]
fn draw_indexed() {
    //
    //   +-----+-----+
    //   |white| red |
    //   +-----+-----+
    //   |blue |white|
    //   +-----+-----+
    //
    let mut expected: [u8; BUFFER_SIZE] = [255; BUFFER_SIZE];
    expected[(BUFFER_SIZE / 4)..][..BUFFER_SIZE / 4].copy_from_slice(
        &(std::iter::repeat([255, 0, 0, 255])
            .flatten()
            .take(BUFFER_SIZE / 4)
            .collect::<Vec<u8>>()),
    );
    expected[((2 * BUFFER_SIZE) / 4)..][..BUFFER_SIZE / 4].copy_from_slice(
        &(std::iter::repeat([0, 0, 255, 255])
            .flatten()
            .take(BUFFER_SIZE / 4)
            .collect::<Vec<u8>>()),
    );
    initialize_test(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::SHADER_PRIMITIVE_INDEX),
        |ctx| {
            pulling_common(ctx, &expected, |rpass| {
                rpass.draw_indexed(0..12, 0, 0..1);
            })
        },
    );
}

fn pulling_common(
    ctx: TestingContext,
    expected: &[u8; BUFFER_SIZE],
    function: impl FnOnce(&mut wgpu::RenderPass<'_>),
) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("primitive_index.wgsl"));

    let first_bottom_right: [f32; 6] = [-1.0, 0.0, 0.0, 0.0, -1.0, -1.0];
    let second_top_left: [f32; 6] = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let third_bottom_right: [f32; 6] = [0.0, 0.0, -1.0, -1.0, 0.0, -1.0];
    let forth_top_left: [f32; 6] = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let vertex_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(
                &[
                    first_bottom_right,
                    second_top_left,
                    third_bottom_right,
                    forth_top_left,
                ]
                .concat(),
            ),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

    let indices = [9u16, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]; // index buffer flips triangle order
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
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                entry_point: "vs_main",
                module: &shader,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                entry_point: "fs_main",
                module: &shader,
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

    let color_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: TEXTURE_WIDTH,
            height: TEXTURE_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let readback_buffer = image::ReadbackBuffers::new(&ctx.device, &color_texture);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: true,
                },
                resolve_target: None,
                view: &color_view,
            })],
            depth_stencil_attachment: None,
            label: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        function(&mut rpass);
    }
    readback_buffer.copy_from(&ctx.device, &mut encoder, &color_texture);
    ctx.queue.submit(Some(encoder.finish()));

    assert!(readback_buffer.check_color_contents(&ctx.device, expected));
}
