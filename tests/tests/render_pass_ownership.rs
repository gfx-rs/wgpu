//! Tests that render passes take ownership of resources that are associated with.
//! I.e. once a resource is passed in to a render pass, it can be dropped.
//!
//! TODO: Methods that take resources that weren't tested here:
//! * rpass.draw_indexed_indirect(indirect_buffer, indirect_offset)
//! * rpass.execute_bundles(render_bundles)
//! * rpass.multi_draw_indirect(indirect_buffer, indirect_offset, count)
//! * rpass.multi_draw_indexed_indirect(indirect_buffer, indirect_offset, count)
//! * rpass.multi_draw_indirect_count
//! * rpass.multi_draw_indexed_indirect_count
//!
use std::{mem::size_of, num::NonZeroU64};

use wgpu::util::DeviceExt as _;
use wgpu_test::{gpu_test, valid, GpuTestConfiguration, TestParameters, TestingContext};

// Minimal shader with buffer based side effect - only needed to check whether the render pass has executed at all.
const SHADER_SRC: &str = "
@group(0) @binding(0)
var<storage, read_write> buffer: array<vec4f>;

var<private> positions: array<vec2f, 3> = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    return vec4f(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    buffer[0] *= 2.0;
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}";

#[gpu_test]
static RENDER_PASS_RESOURCE_OWNERSHIP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(render_pass_resource_ownership);

async fn render_pass_resource_ownership(ctx: TestingContext) {
    let ResourceSetup {
        gpu_buffer,
        cpu_buffer,
        buffer_size,
        indirect_buffer,
        vertex_buffer,
        index_buffer,
        bind_group,
        pipeline,
        color_attachment_view,
        color_attachment_resolve_view,
        depth_stencil_view,
        occlusion_query_set,
    } = resource_setup(&ctx);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_attachment_view,
                resolve_target: Some(&color_attachment_resolve_view),
                ops: wgpu::Operations::default(),
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_stencil_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: Some(&occlusion_query_set),
        });

        // Drop render pass attachments right away.
        drop(color_attachment_view);
        drop(color_attachment_resolve_view);
        drop(depth_stencil_view);

        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.begin_occlusion_query(0);
        rpass.draw_indirect(&indirect_buffer, 0);
        rpass.end_occlusion_query();

        // Now drop all resources we set. Then do a device poll to make sure the resources are really not dropped too early, no matter what.
        drop(pipeline);
        drop(bind_group);
        drop(indirect_buffer);
        drop(vertex_buffer);
        drop(index_buffer);
        drop(occlusion_query_set);
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
    }

    assert_render_pass_executed_normally(encoder, gpu_buffer, cpu_buffer, buffer_size, ctx).await;
}

#[gpu_test]
static RENDER_PASS_QUERY_SET_OWNERSHIP_PIPELINE_STATISTICS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .features(wgpu::Features::PIPELINE_STATISTICS_QUERY),
        )
        .run_async(render_pass_query_set_ownership_pipeline_statistics);

async fn render_pass_query_set_ownership_pipeline_statistics(ctx: TestingContext) {
    let ResourceSetup {
        gpu_buffer,
        cpu_buffer,
        buffer_size,
        vertex_buffer,
        index_buffer,
        bind_group,
        pipeline,
        color_attachment_view,
        depth_stencil_view,
        ..
    } = resource_setup(&ctx);

    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("query_set"),
        ty: wgpu::QueryType::PipelineStatistics(
            wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS,
        ),
        count: 1,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_attachment_view,
                resolve_target: None,
                ops: wgpu::Operations::default(),
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_stencil_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.begin_pipeline_statistics_query(&query_set, 0);
        rpass.draw(0..3, 0..1);
        rpass.end_pipeline_statistics_query();

        // Drop the query set. Then do a device poll to make sure it's not dropped too early, no matter what.
        drop(query_set);
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
    }

    assert_render_pass_executed_normally(encoder, gpu_buffer, cpu_buffer, buffer_size, ctx).await;
}

#[gpu_test]
static RENDER_PASS_QUERY_SET_OWNERSHIP_TIMESTAMPS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits().features(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        ))
        .run_async(render_pass_query_set_ownership_timestamps);

async fn render_pass_query_set_ownership_timestamps(ctx: TestingContext) {
    let ResourceSetup {
        gpu_buffer,
        cpu_buffer,
        buffer_size,
        color_attachment_view,
        depth_stencil_view,
        pipeline,
        bind_group,
        vertex_buffer,
        index_buffer,
        ..
    } = resource_setup(&ctx);

    let query_set_timestamp_writes = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("query_set_timestamp_writes"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let query_set_write_timestamp = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("query_set_write_timestamp"),
        ty: wgpu::QueryType::Timestamp,
        count: 1,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_attachment_view,
                resolve_target: None,
                ops: wgpu::Operations::default(),
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_stencil_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                query_set: &query_set_timestamp_writes,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
            ..Default::default()
        });
        rpass.write_timestamp(&query_set_write_timestamp, 0);

        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw(0..3, 0..1);

        // Drop the query sets. Then do a device poll to make sure they're not dropped too early, no matter what.
        drop(query_set_timestamp_writes);
        drop(query_set_write_timestamp);
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
    }

    assert_render_pass_executed_normally(encoder, gpu_buffer, cpu_buffer, buffer_size, ctx).await;
}

#[gpu_test]
static RENDER_PASS_KEEP_ENCODER_ALIVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(render_pass_keep_encoder_alive);

async fn render_pass_keep_encoder_alive(ctx: TestingContext) {
    let ResourceSetup {
        bind_group,
        vertex_buffer,
        index_buffer,
        pipeline,
        color_attachment_view,
        depth_stencil_view,
        ..
    } = resource_setup(&ctx);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &color_attachment_view,
            resolve_target: None,
            ops: wgpu::Operations::default(),
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_stencil_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });

    // Now drop the encoder - it is kept alive by the compute pass.
    // To do so, we have to make the compute pass forget the lifetime constraint first.
    let mut rpass = rpass.forget_lifetime();
    drop(encoder);

    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    // Record some a draw command.
    rpass.set_pipeline(&pipeline);
    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
    rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    rpass.draw(0..3, 0..1);

    // Dropping the pass will still execute the pass, even though there's no way to submit it.
    // Ideally, this would log an error, but the encoder is not dropped until the compute pass is dropped,
    // making this a valid operation.
    // (If instead the encoder was explicitly destroyed or finished, this would be an error.)
    valid(&ctx.device, || drop(rpass));
}

async fn assert_render_pass_executed_normally(
    mut encoder: wgpu::CommandEncoder,
    gpu_buffer: wgpu::Buffer,
    cpu_buffer: wgpu::Buffer,
    buffer_size: u64,
    ctx: TestingContext,
) {
    encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, buffer_size);
    ctx.queue.submit([encoder.finish()]);
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    let data = cpu_buffer.slice(..).get_mapped_range();

    let floats: &[f32] = bytemuck::cast_slice(&data);
    assert!(floats[0] >= 2.0);
    assert!(floats[1] >= 4.0);
    assert!(floats[2] >= 6.0);
    assert!(floats[3] >= 8.0);
}

// Setup ------------------------------------------------------------

struct ResourceSetup {
    gpu_buffer: wgpu::Buffer,
    cpu_buffer: wgpu::Buffer,
    buffer_size: u64,

    indirect_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,

    color_attachment_view: wgpu::TextureView,
    color_attachment_resolve_view: wgpu::TextureView,
    depth_stencil_view: wgpu::TextureView,
    occlusion_query_set: wgpu::QuerySet,
}

fn resource_setup(ctx: &TestingContext) -> ResourceSetup {
    let sm = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

    let buffer_size = 4 * size_of::<f32>() as u64;

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(buffer_size),
                },
                count: None,
            }],
        });

    let gpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            contents: bytemuck::bytes_of(&[1.0_f32, 2.0, 3.0, 4.0]),
        });

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let vertex_count = 3;
    let indirect_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_buffer"),
            usage: wgpu::BufferUsages::INDIRECT,
            contents: wgpu::util::DrawIndirectArgs {
                vertex_count,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            }
            .as_bytes(),
        });

    let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vertex_buffer"),
        usage: wgpu::BufferUsages::VERTEX,
        size: size_of::<u32>() as u64 * vertex_count as u64,
        mapped_at_creation: false,
    });

    let index_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            usage: wgpu::BufferUsages::INDEX,
            contents: bytemuck::cast_slice(&[0_u32, 1, 2]),
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let target_size = wgpu::Extent3d {
        width: 4,
        height: 4,
        depth_or_array_layers: 1,
    };
    let target_msaa = 4;
    let target_format = wgpu::TextureFormat::Bgra8UnormSrgb;

    let target_desc = wgpu::TextureDescriptor {
        label: Some("target_tex"),
        size: target_size,
        mip_level_count: 1,
        sample_count: target_msaa,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[target_format],
    };
    let target_tex = ctx.device.create_texture(&target_desc);
    let target_tex_resolve = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target_resolve"),
        sample_count: 1,
        ..target_desc
    });

    let color_attachment_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let color_attachment_resolve_view =
        target_tex_resolve.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_stencil_format = wgpu::TextureFormat::Depth32Float;
    let depth_stencil = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_stencil"),
        format: depth_stencil_format,
        view_formats: &[depth_stencil_format],
        ..target_desc
    });
    let depth_stencil_view = depth_stencil.create_view(&wgpu::TextureViewDescriptor::default());

    let occlusion_query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("occ_query_set"),
        ty: wgpu::QueryType::Occlusion,
        count: 1,
    });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sm,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Uint32],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sm,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_stencil_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: target_msaa,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

    ResourceSetup {
        gpu_buffer,
        cpu_buffer,
        buffer_size,

        indirect_buffer,
        vertex_buffer,
        index_buffer,
        bind_group,
        pipeline,

        color_attachment_view,
        color_attachment_resolve_view,
        depth_stencil_view,
        occlusion_query_set,
    }
}
