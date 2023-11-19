use std::num::NonZeroU64;

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

fn pulling_common<'b, F>(ctx: TestingContext, expected: &[u32], function: F)
where
    F: for<'a> FnOnce(&mut wgpu::RenderPass<'a>, &'a &'b ()) + 'b,
{
    let index_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("index buffer"),
        contents: bytemuck::cast_slice(&[0u32, 1, 2, 3, 4, 5]),
        usage: wgpu::BufferUsages::INDEX,
    });

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

    let buffer_size = 4 * expected.len() as u64;
    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buffer.as_entire_binding(),
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
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            ops: wgpu::Operations::default(),
            resolve_target: None,
            view: &dummy,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    rpass.set_pipeline(&pipeline);
    rpass.set_bind_group(0, &bg, &[]);
    function(&mut rpass, &&());

    drop(rpass);

    encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, buffer_size);

    ctx.queue.submit(Some(encoder.finish()));
    let slice = cpu_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.device.poll(wgpu::Maintain::Wait);
    let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

    assert_eq!(data, expected);
}

fn draw_indirect(ctx: &TestingContext, dii: wgpu::util::DrawIndirect) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: dii.as_bytes(),
            usage: wgpu::BufferUsages::INDIRECT,
        })
}

fn draw_indexed_indirect(
    ctx: &TestingContext,
    dii: wgpu::util::DrawIndexedIndirect,
) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: dii.as_bytes(),
            usage: wgpu::BufferUsages::INDIRECT,
        })
}

#[gpu_test]
static DRAW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb, _| {
            cmb.draw(0..6, 0..1);
        })
    });

#[gpu_test]
static DRAW_VERTEX_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb, _| {
            cmb.draw(0..3, 0..1);
            cmb.draw(3..6, 0..1);
        })
    });

#[gpu_test]
static DRAW_BASE_VERTEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        pulling_common(ctx, &[0, 0, 0, 3, 4, 5, 6, 7, 8], |cmb, _| {
            cmb.draw_indexed(0..6, 3, 0..1);
        })
    });

#[gpu_test]
static DRAW_INSTANCED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb, _| {
            cmb.draw(0..3, 0..2);
        })
    });

#[gpu_test]
static DRAW_INSTANCED_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb, _| {
            cmb.draw(0..3, 0..1);
            cmb.draw(0..3, 1..2);
        })
    });

#[gpu_test]
static DRAW_INDIRECT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        let indirect = draw_indirect(
            &ctx,
            wgpu::util::DrawIndirect {
                vertex_count: 6,
                instance_count: 1,
                base_vertex: 0,
                base_instance: 0,
            },
        );

        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb, _| {
            cmb.draw_indirect(&indirect, 0);
        })
    });

#[gpu_test]
static DRAW_INDIRECT_VERTEX_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        let call1 = draw_indirect(
            &ctx,
            wgpu::util::DrawIndirect {
                vertex_count: 3,
                instance_count: 1,
                base_vertex: 0,
                base_instance: 0,
            },
        );
        let call2 = draw_indirect(
            &ctx,
            wgpu::util::DrawIndirect {
                vertex_count: 3,
                instance_count: 1,
                base_vertex: 3,
                base_instance: 0,
            },
        );
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb, _| {
            cmb.draw_indirect(&call1, 0);
            cmb.draw_indirect(&call2, 0);
        })
    });

#[gpu_test]
static DRAW_INDIRECT_BASE_VERTEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        let indirect = draw_indexed_indirect(
            &ctx,
            wgpu::util::DrawIndexedIndirect {
                vertex_count: 6,
                instance_count: 1,
                vertex_offset: 3,
                base_index: 0,
                base_instance: 0,
            },
        );
        pulling_common(ctx, &[0, 0, 0, 3, 4, 5, 6, 7, 8], |cmb, _| {
            cmb.draw_indexed_indirect(&indirect, 0);
        })
    });

#[gpu_test]
static DRAW_INDIRECT_INSTANCED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        let indirect = draw_indirect(
            &ctx,
            wgpu::util::DrawIndirect {
                vertex_count: 3,
                instance_count: 2,
                base_vertex: 0,
                base_instance: 0,
            },
        );
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb, _| {
            cmb.draw_indirect(&indirect, 0);
        })
    });

#[gpu_test]
static DRAW_INDIRECT_INSTANCED_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(|ctx| {
        let call1 = draw_indirect(
            &ctx,
            wgpu::util::DrawIndirect {
                vertex_count: 3,
                instance_count: 1,
                base_vertex: 0,
                base_instance: 0,
            },
        );
        let call2 = draw_indirect(
            &ctx,
            wgpu::util::DrawIndirect {
                vertex_count: 3,
                instance_count: 1,
                base_vertex: 0,
                base_instance: 1,
            },
        );
        pulling_common(ctx, &[0, 1, 2, 3, 4, 5], |cmb, _| {
            cmb.draw_indirect(&call1, 0);
            cmb.draw_indirect(&call2, 0);
        })
    });
