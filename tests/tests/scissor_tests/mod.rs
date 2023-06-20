use std::mem::size_of;

use wgpu::util::DeviceExt;
use wgpu_test::{initialize_test, TestParameters, TestingContext};
use wgt::BufferAddress;

struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn scissor_test_impl(ctx: &TestingContext, scissor_rect: Rect, expected_data: [u8; 16]) {
    let vertex_data: [f32; 12] = [
        -1.0_f32, -1.0_f32, 0.0_f32, 1.0_f32, -1.0_f32, 0.0_f32, 1.0_f32, 1.0_f32, 0.0_f32,
        -1.0_f32, 1.0_f32, 0.0_f32,
    ];
    let index_data: [i16; 6] = [0, 1, 3, 1, 3, 2];

    let vertex_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

    let index_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Offscreen texture"),
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("solid_white.wgsl"));

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (size_of::<f32>() * 3) as BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                entry_point: "vs_main",
                module: &shader,
            },
            primitive: wgpu::PrimitiveState::default(),
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

    {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderpass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&pipeline);
            render_pass.set_scissor_rect(
                scissor_rect.x,
                scissor_rect.y,
                scissor_rect.width,
                scissor_rect.height,
            );
            render_pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_vertex_buffer(0, vertex_buf.slice(..));
            render_pass.draw_indexed(
                0..(index_buf.size() as usize / size_of::<i16>()) as u32,
                0,
                0..1,
            );
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
    let data = wgpu_test::image::capture_rgba_u8_texture(
        ctx,
        &texture,
        wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
    );
    assert_eq!(data.len(), 16);
    for i in 0..16 {
        assert_eq!(data[i], expected_data[i]);
    }
}

#[test]
fn scissor_test_full_rect() {
    initialize_test(TestParameters::default(), |ctx| {
        scissor_test_impl(
            &ctx,
            Rect {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            [255; 16],
        );
    })
}

#[test]
fn scissor_test_empty_rect() {
    initialize_test(TestParameters::default(), |ctx| {
        scissor_test_impl(
            &ctx,
            Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            [0; 16],
        );
    })
}

#[test]
fn scissor_test_empty_rect_with_offset() {
    initialize_test(TestParameters::default(), |ctx| {
        scissor_test_impl(
            &ctx,
            Rect {
                x: 1,
                y: 1,
                width: 0,
                height: 0,
            },
            [0; 16],
        );
    })
}

#[test]
fn scissor_test_custom_rect() {
    let mut expected_result = [0; 16];
    expected_result[4..][..4].copy_from_slice(&[255; 4]);
    initialize_test(TestParameters::default(), |ctx| {
        scissor_test_impl(
            &ctx,
            Rect {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            },
            expected_result,
        );
    })
}
