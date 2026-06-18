//! Tests for query set initialization tracking.
//!
//! Unwritten query slots must resolve to zero.
//! Written slots must not be affected.

use wgpu::*;
use wgpu_test::{
    gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        // Occlusion — no features required.
        UNWRITTEN_OCCLUSION_RESOLVES_TO_ZERO,
        USED_EMPTY_OCCLUSION_RESOLVES_TO_ZERO,
        SAME_ENCODER_RESOLVES_CORRECTLY,
        WRITTEN_IN_PRIOR_SUBMIT_RESOLVES_CORRECTLY,
        WRITTEN_IN_PRIOR_SUBMIT_AND_SAME_ENCODER_RESOLVES_CORRECTLY,
        // Timestamp — requires TIMESTAMP_QUERY.
        UNWRITTEN_TIMESTAMP_RESOLVES_TO_ZERO,
        WRITTEN_TIMESTAMP_RESOLVES_TO_NONZERO,
        // Inside-pass timestamp — requires TIMESTAMP_QUERY_INSIDE_PASSES.
        INSIDE_PASS_WRITTEN_TIMESTAMP_RESOLVES_TO_NONZERO,
        // Encoder timestamp — requires TIMESTAMP_QUERY_INSIDE_ENCODERS.
        ENCODER_WRITTEN_TIMESTAMP_RESOLVES_TO_NONZERO,
        // Pipeline statistics — requires PIPELINE_STATISTICS_QUERY.
        UNWRITTEN_PIPELINE_STATISTICS_RESOLVES_TO_ZERO,
        WRITTEN_PIPELINE_STATISTICS_RESOLVES_TO_NONZERO,
    ]);
}

#[gpu_test]
static UNWRITTEN_OCCLUSION_RESOLVES_TO_ZERO: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move {
        const COUNT: u32 = 4;
        let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Occlusion, COUNT);

        let mut enc = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        enc.resolve_query_set(&qs, 0..COUNT, &dst, 0);
        ctx.queue.submit(Some(enc.finish()));

        let result = read_resolve_buffer(&ctx, &dst, size).await;
        assert!(
            result.iter().all(|&v| v == 0),
            "unwritten occlusion slots must resolve to zero, got {:?}",
            result
        );
    });

/// A query that was properly begun and ended (but with no actual draws, so the
/// result is 0) should also resolve to zero.
#[gpu_test]
static USED_EMPTY_OCCLUSION_RESOLVES_TO_ZERO: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move {
        let view = ctx
            .device
            .create_texture(&TextureDescriptor {
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
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default());

        let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Occlusion, 1);

        let mut enc = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: Some(&qs),
                multiview_mask: None,
            });
            pass.begin_occlusion_query(0);
            pass.end_occlusion_query();
        }
        enc.resolve_query_set(&qs, 0..1, &dst, 0);
        ctx.queue.submit(Some(enc.finish()));

        let result = read_resolve_buffer(&ctx, &dst, size).await;
        assert_eq!(
            result[0], 0,
            "used-empty occlusion query must resolve to 0, got {}",
            result[0]
        );
    });

/// Slots 0 and 3 are written before the resolve.
/// Slot 1 is written after the resolve.
/// Slot 2 is never written.
#[gpu_test]
static SAME_ENCODER_RESOLVES_CORRECTLY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // See https://github.com/gfx-rs/wgpu/issues/9665.
            .expect_fail(FailureCase::webgl2()),
    )
    .run_async(|ctx| async move {
        const COUNT: u32 = 4;
        let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Occlusion, COUNT);
        let (view, pipeline) = create_occlusion_render_resources(&ctx);

        let mut enc = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        write_occlusion_query(&mut enc, &view, &pipeline, &qs, 0);
        write_occlusion_query(&mut enc, &view, &pipeline, &qs, 3);
        enc.resolve_query_set(&qs, 0..COUNT, &dst, 0);
        write_occlusion_query(&mut enc, &view, &pipeline, &qs, 1);

        ctx.queue.submit(Some(enc.finish()));

        let result = read_resolve_buffer(&ctx, &dst, size).await;
        assert_ne!(
            result[0], 0,
            "slot 0 (written before resolve) must be non-zero"
        );
        assert_eq!(result[1], 0, "slot 1 (written after resolve) must be zero");
        assert_eq!(result[2], 0, "slot 2 (never written) must be zero");
        assert_ne!(
            result[3], 0,
            "slot 3 (written before resolve) must be non-zero"
        );
    });

/// Write queries in submit 1, resolve them in submit 2.
#[gpu_test]
static WRITTEN_IN_PRIOR_SUBMIT_RESOLVES_CORRECTLY: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                // See https://github.com/gfx-rs/wgpu/issues/9665.
                .expect_fail(FailureCase::webgl2()),
        )
        .run_async(|ctx| async move {
            const COUNT: u32 = 2;
            let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Occlusion, COUNT);
            let (view, pipeline) = create_occlusion_render_resources(&ctx);

            // Submit 1: write both slots.
            let mut enc1 = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            write_occlusion_query(&mut enc1, &view, &pipeline, &qs, 0);
            write_occlusion_query(&mut enc1, &view, &pipeline, &qs, 1);
            ctx.queue.submit(Some(enc1.finish()));
            ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

            // Submit 2: resolve only.
            let mut enc2 = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            enc2.resolve_query_set(&qs, 0..COUNT, &dst, 0);
            ctx.queue.submit(Some(enc2.finish()));

            let result = read_resolve_buffer(&ctx, &dst, size).await;
            assert!(
                result.iter().all(|&v| v != 0),
                "slots written in a prior submit must resolve to non-zero, got {:?}",
                result
            );
        });

/// Slot 0 written in submit 1.
/// Slot 1 written in submit 2 before the resolve.
/// Slot 2 is never written.
#[gpu_test]
static WRITTEN_IN_PRIOR_SUBMIT_AND_SAME_ENCODER_RESOLVES_CORRECTLY: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                // See https://github.com/gfx-rs/wgpu/issues/9665.
                .expect_fail(FailureCase::webgl2()),
        )
        .run_async(|ctx| async move {
            const COUNT: u32 = 3;
            let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Occlusion, COUNT);
            let (view, pipeline) = create_occlusion_render_resources(&ctx);

            // Submit 1: write slot 0.
            let mut enc1 = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            write_occlusion_query(&mut enc1, &view, &pipeline, &qs, 0);
            ctx.queue.submit(Some(enc1.finish()));
            ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

            // Submit 2: write slot 1, then resolve.
            let mut enc2 = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            write_occlusion_query(&mut enc2, &view, &pipeline, &qs, 1);
            enc2.resolve_query_set(&qs, 0..COUNT, &dst, 0);
            ctx.queue.submit(Some(enc2.finish()));

            let result = read_resolve_buffer(&ctx, &dst, size).await;
            assert_ne!(
                result[0], 0,
                "slot 0 (written in prior submit) must be non-zero"
            );
            assert_ne!(
                result[1], 0,
                "slot 1 (written in current encoder before resolve) must be non-zero"
            );
            assert_eq!(result[2], 0, "slot 2 (unwritten) must be zero");
        });

#[gpu_test]
static UNWRITTEN_TIMESTAMP_RESOLVES_TO_ZERO: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(Features::TIMESTAMP_QUERY))
    .run_async(|ctx| async move {
        let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Timestamp, 1);

        let mut enc = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        enc.resolve_query_set(&qs, 0..1, &dst, 0);
        ctx.queue.submit(Some(enc.finish()));

        let result = read_resolve_buffer(&ctx, &dst, size).await;
        assert_eq!(
            result[0], 0,
            "unwritten timestamp slot must resolve to zero, got {}",
            result[0]
        );
    });

#[gpu_test]
static WRITTEN_TIMESTAMP_RESOLVES_TO_NONZERO: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(Features::TIMESTAMP_QUERY),
    )
    .run_async(|ctx| async move {
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl("@compute @workgroup_size(1) fn cs_main() {}".into()),
        });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        const COUNT: u32 = 2;
        let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Timestamp, COUNT);

        let mut enc = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: &qs,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.resolve_query_set(&qs, 0..COUNT, &dst, 0);
        ctx.queue.submit(Some(enc.finish()));

        let result = read_resolve_buffer(&ctx, &dst, size).await;
        assert!(
            result.iter().all(|&v| v != 0),
            "written timestamp slots must be non-zero, got {:?}",
            result
        );
    });

#[gpu_test]
static INSIDE_PASS_WRITTEN_TIMESTAMP_RESOLVES_TO_NONZERO: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .features(Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES),
        )
        .run_async(|ctx| async move {
            let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl("@compute @workgroup_size(1) fn cs_main() {}".into()),
            });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: Some("cs_main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            const COUNT: u32 = 2;
            let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Timestamp, COUNT);

            let mut enc = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.write_timestamp(&qs, 0);
                pass.dispatch_workgroups(1, 1, 1);
            }
            enc.resolve_query_set(&qs, 0..COUNT, &dst, 0);
            ctx.queue.submit(Some(enc.finish()));

            let result = read_resolve_buffer(&ctx, &dst, size).await;
            assert_ne!(
                result[0], 0,
                "slot 0 (written inside pass) must be non-zero"
            );
            assert_eq!(result[1], 0, "slot 1 (unwritten) must be zero");
        });

#[gpu_test]
static ENCODER_WRITTEN_TIMESTAMP_RESOLVES_TO_NONZERO: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .features(Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        )
        .run_async(|ctx| async move {
            const COUNT: u32 = 2;
            let (qs, dst, size) = create_resolve_resources(&ctx, QueryType::Timestamp, COUNT);

            let mut enc = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            enc.write_timestamp(&qs, 0);
            enc.resolve_query_set(&qs, 0..COUNT, &dst, 0);
            ctx.queue.submit(Some(enc.finish()));

            let result = read_resolve_buffer(&ctx, &dst, size).await;
            assert_ne!(result[0], 0, "slot 0 (written) must be non-zero");
            assert_eq!(result[1], 0, "slot 1 (unwritten) must be zero");
        });

#[gpu_test]
static UNWRITTEN_PIPELINE_STATISTICS_RESOLVES_TO_ZERO: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .features(Features::PIPELINE_STATISTICS_QUERY),
        )
        .run_async(|ctx| async move {
            let (qs, dst, size) = create_resolve_resources(
                &ctx,
                QueryType::PipelineStatistics(PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS),
                1,
            );

            let mut enc = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            enc.resolve_query_set(&qs, 0..1, &dst, 0);
            ctx.queue.submit(Some(enc.finish()));

            let result = read_resolve_buffer(&ctx, &dst, size).await;
            assert_eq!(
                result[0], 0,
                "unwritten pipeline-statistics slot must be zero, got {}",
                result[0]
            );
        });

#[gpu_test]
static WRITTEN_PIPELINE_STATISTICS_RESOLVES_TO_NONZERO: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .features(Features::PIPELINE_STATISTICS_QUERY),
        )
        .run_async(|ctx| async move {
            let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl("@compute @workgroup_size(1) fn cs_main() {}".into()),
            });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: Some("cs_main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let (qs, dst, size) = create_resolve_resources(
                &ctx,
                QueryType::PipelineStatistics(PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS),
                1,
            );

            let mut enc = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.begin_pipeline_statistics_query(&qs, 0);
                pass.dispatch_workgroups(4, 1, 1);
                pass.end_pipeline_statistics_query();
            }
            enc.resolve_query_set(&qs, 0..1, &dst, 0);
            ctx.queue.submit(Some(enc.finish()));

            let result = read_resolve_buffer(&ctx, &dst, size).await;
            assert_eq!(
                result[0], 4,
                "written pipeline-statistics slot must be 4, got {}",
                result[0]
            );
        });

fn create_occlusion_render_resources(ctx: &TestingContext) -> (TextureView, RenderPipeline) {
    let view = ctx
        .device
        .create_texture(&TextureDescriptor {
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
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default());
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            "
            @vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
                var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                return vec4(pos[vi], 0.0, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4<f32> {
                return vec4(1.0);
            }
            "
            .into(),
        ),
    });
    let pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
    (view, pipeline)
}

fn write_occlusion_query(
    enc: &mut CommandEncoder,
    view: &TextureView,
    pipeline: &RenderPipeline,
    qs: &QuerySet,
    slot: u32,
) {
    let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Discard,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: Some(qs),
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.begin_occlusion_query(slot);
    pass.draw(0..3, 0..1);
    pass.end_occlusion_query();
}

fn create_resolve_resources(
    ctx: &TestingContext,
    ty: QueryType,
    count: u32,
) -> (QuerySet, Buffer, u64) {
    let qs = ctx.device.create_query_set(&QuerySetDescriptor {
        label: None,
        ty,
        count,
    });
    let size = count as u64 * QUERY_SIZE as u64;
    let dst = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("resolve"),
        size,
        usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&dst, 0, &vec![0xAAu8; size as usize]);
    (qs, dst, size)
}

async fn read_resolve_buffer(ctx: &TestingContext, src: &Buffer, size: u64) -> Vec<u64> {
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(src, 0, &readback, 0, size);
    ctx.queue.submit(Some(enc.finish()));
    readback.slice(..).map_async(MapMode::Read, |_| ());
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    let view = readback.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice::<u8, u64>(&view).to_vec()
}
