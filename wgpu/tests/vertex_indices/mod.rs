use std::num::NonZeroU64;

use wasm_bindgen_test::*;
use wgpu::util::DeviceExt;

use crate::common::{initialize_test, TestParameters, TestingContext};

fn pulling_common(
    ctx: TestingContext,
    expected: &[u32],
    function: impl FnOnce(&mut wgpu::RenderPass<'_>),
) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("draw.vert.wgsl"));

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                visibility: wgpu::ShaderStages::VERTEX,
                count: None,
            }],
        });

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * expected.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let ppl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&ppl),
            vertex: wgpu::VertexState {
                buffers: &[],
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

    let dummy = ctx
        .device
        .create_texture_with_data(
            &ctx.queue,
            &wgpu::TextureDescriptor {
                label: Some("dummy"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            &[0, 0, 0, 1],
        )
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            ops: wgpu::Operations::default(),
            resolve_target: None,
            view: &dummy,
        })],
        depth_stencil_attachment: None,
        label: None,
        timestamp_writes: &[],
    });

    rpass.set_pipeline(&pipeline);
    rpass.set_bind_group(0, &bg, &[]);
    function(&mut rpass);

    drop(rpass);

    ctx.queue.submit(Some(encoder.finish()));
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.device.poll(wgpu::Maintain::Wait);
    let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

    assert_eq!(data, expected);
}

#[test]
#[wasm_bindgen_test]
fn draw() {
    initialize_test(TestParameters::default().test_features_limits(), |ctx| {
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb| {
            cmb.draw(0..6, 0..1);
        })
    })
}

#[test]
#[wasm_bindgen_test]
fn draw_vertex_offset() {
    initialize_test(
        TestParameters::default()
            .test_features_limits()
            .backend_failure(wgpu::Backends::DX11),
        |ctx| {
            pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb| {
                cmb.draw(0..3, 0..1);
                cmb.draw(3..6, 0..1);
            })
        },
    )
}

#[test]
#[wasm_bindgen_test]
fn draw_instanced() {
    initialize_test(TestParameters::default().test_features_limits(), |ctx| {
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb| {
            cmb.draw(0..3, 0..2);
        })
    })
}

#[test]
#[wasm_bindgen_test]
fn draw_instanced_offset() {
    initialize_test(
        TestParameters::default()
            .test_features_limits()
            .backend_failure(wgpu::Backends::DX11),
        |ctx| {
            pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb| {
                cmb.draw(0..3, 0..1);
                cmb.draw(0..3, 1..2);
            })
        },
    )
}
