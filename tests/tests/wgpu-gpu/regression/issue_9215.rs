use std::mem::size_of;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext};

fn acceleration_structure_limits() -> wgpu::Limits {
    wgpu::Limits::default().using_minimum_supported_acceleration_structure_values()
}

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(BLAS_AND_TLAS_IN_SAME_COMMAND_BUFFER);
    vec.push(BLAS_AND_TLAS_IN_SEPARATE_COMMAND_BUFFERS);
    vec.push(ENCODER_DISCARDED_AFTER_BLAS_BUILD);
    vec.push(COMMAND_BUFFER_DROPPED_WITHOUT_SUBMIT);
    vec.push(BLIT_BETWEEN_BUILDS_IN_SAME_COMMAND_BUFFER);
    vec.push(COMPUTE_PASS_BETWEEN_BUILDS_IN_SAME_COMMAND_BUFFER);
    vec.push(COMPACTED_BLAS_IN_PENDING_WRITES);
    vec.push(ENCODER_DISCARDED_AFTER_PUBLISHED_CLOSE);
    vec.push(COMMAND_BUFFER_DROPPED_AFTER_PUBLISHED_CLOSE);
}

// The failures are races, so a pass is evidence rather than proof. On the
// hardware this bug reproduces on (Apple M4 Max), the unfixed failure rate of
// the affected configurations is tens of percent per iteration, which 40
// repetitions turn into a strong signal while keeping the test fast.
const ITERATIONS: u32 = 40;

// `rayQueryGetCommittedIntersection().kind` for a committed triangle hit.
const TRIANGLE_HIT_KIND: u32 = 1;

// On Metal, acceleration structure builds are not ordered against each other
// within an encoder, nor across command buffers committed back to back, so a
// TLAS build could consume a BLAS that is still building. The ray query below
// then commits no intersection (kind 0) instead of the triangle (kind 1).
//
// Each iteration builds a fresh BLAS and a TLAS referencing it, traces a ray
// straight up through the triangle, and reads back the committed intersection
// kind.
//
// See <https://github.com/gfx-rs/wgpu/issues/9215>.

struct TraceSetup {
    vertex_buffer: wgpu::Buffer,
    blas_size: wgpu::BlasTriangleGeometrySizeDescriptor,
    blas: wgpu::Blas,
    tlas: wgpu::Tlas,
    hit_buffer: wgpu::Buffer,
    read_back: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn create_trace_setup(ctx: &TestingContext, pipeline: &wgpu::ComputePipeline) -> TraceSetup {
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("BLAS vertices"),
        contents: bytemuck::cast_slice(&[[1.0_f32, 1.0, 0.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]]),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });

    let blas_size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas_flags = wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD;
    let blas = ctx.device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: None,
            flags: blas_flags,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );
    let mut tlas = ctx.device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        max_instances: 1,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });

    tlas[0] = Some(wgpu::TlasInstance::new(
        &blas,
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ],
        0,
        0xff,
    ));

    let hit_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hit kind"),
        size: size_of::<u32>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let read_back = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hit kind readback"),
        size: hit_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tlas.as_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hit_buffer.as_entire_binding(),
            },
        ],
    });

    TraceSetup {
        vertex_buffer,
        blas_size,
        blas,
        tlas,
        hit_buffer,
        read_back,
        bind_group,
    }
}

fn blas_build_entry(setup: &TraceSetup) -> wgpu::BlasBuildEntry<'_> {
    wgpu::BlasBuildEntry {
        blas: &setup.blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
            size: &setup.blas_size,
            vertex_buffer: &setup.vertex_buffer,
            first_vertex: 0,
            vertex_stride: 12,
            index_buffer: None,
            first_index: None,
            transform_buffer: None,
            transform_buffer_offset: None,
        }]),
    }
}

// Records the TLAS build, the ray query, and the readback copy onto `encoder`.
fn record_tlas_trace_and_copy(
    setup: &TraceSetup,
    pipeline: &wgpu::ComputePipeline,
    encoder: &mut wgpu::CommandEncoder,
) {
    encoder.build_acceleration_structures([], [&setup.tlas]);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &setup.bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &setup.hit_buffer,
        0,
        &setup.read_back,
        0,
        setup.hit_buffer.size(),
    );
}

async fn read_hit_kind(ctx: &TestingContext, setup: &TraceSetup) -> u32 {
    let slice = setup.read_back.slice(..);
    slice.map_async(wgpu::MapMode::Read, Result::unwrap);
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let hit_kind = {
        let view = slice.get_mapped_range().unwrap();
        u32::from_ne_bytes(view[..size_of::<u32>()].try_into().unwrap())
    };
    setup.read_back.unmap();
    hit_kind
}

fn create_ray_query_pipeline(ctx: &TestingContext) -> wgpu::ComputePipeline {
    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("issue_9215.wgsl"));
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
}

async fn run_iterations(ctx: TestingContext, separate_command_buffers: bool) {
    let pipeline = create_ray_query_pipeline(&ctx);

    for i in 0..ITERATIONS {
        let setup = create_trace_setup(&ctx, &pipeline);

        if separate_command_buffers {
            let mut blas_encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let blas_entry = blas_build_entry(&setup);
            blas_encoder.build_acceleration_structures([&blas_entry], []);

            let mut tlas_encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            record_tlas_trace_and_copy(&setup, &pipeline, &mut tlas_encoder);

            ctx.queue
                .submit([blas_encoder.finish(), tlas_encoder.finish()]);
        } else {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let blas_entry = blas_build_entry(&setup);
            encoder.build_acceleration_structures([&blas_entry], []);
            record_tlas_trace_and_copy(&setup, &pipeline, &mut encoder);

            ctx.queue.submit([encoder.finish()]);
        }

        let hit_kind = read_hit_kind(&ctx, &setup).await;
        assert_eq!(
            hit_kind, TRIANGLE_HIT_KIND,
            "iteration {i}: ray query did not commit a triangle hit"
        );
    }
}

// A BLAS build is recorded and then thrown away: either by discarding the
// encoder, or by finishing and dropping the command buffer without submitting
// it. The consumer below is fully encoded before the producer goes away, but
// the producer's encoder is not: wgpu-core recycles hal command encoders, so
// any per-command-buffer state the producer leaves behind is handed to a
// later recording. The submission here must run its own acceleration
// structure work correctly regardless of what the abandoned recording did.
async fn run_iterations_with_abandoned_producer(ctx: TestingContext, discard_encoder: bool) {
    let pipeline = create_ray_query_pipeline(&ctx);

    for i in 0..ITERATIONS {
        let setup = create_trace_setup(&ctx, &pipeline);

        let mut producer = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let blas_entry = blas_build_entry(&setup);
        producer.build_acceleration_structures([&blas_entry], []);

        let mut consumer = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let blas_entry = blas_build_entry(&setup);
        consumer.build_acceleration_structures([&blas_entry], []);
        record_tlas_trace_and_copy(&setup, &pipeline, &mut consumer);

        if discard_encoder {
            drop(producer);
        } else {
            drop(producer.finish());
        }

        ctx.queue.submit([consumer.finish()]);

        let hit_kind = read_hit_kind(&ctx, &setup).await;
        assert_eq!(
            hit_kind, TRIANGLE_HIT_KIND,
            "iteration {i}: ray query did not commit a triangle hit"
        );
    }
}

// A BLAS build followed by a blit is recorded and then thrown away: either by
// discarding the encoder, or by finishing and dropping the command buffer
// without submitting it. The blit forces the published close of the build's
// acceleration structure encoder, so the abandoned recording leaves behind
// encoder state saying a fence was updated and acceleration structure
// commands are present, with no consumer left in the recording. The consumer
// is only recorded after the producer is gone, so it records on the
// producer's recycled encoder: leftover state would make its first
// acceleration structure encoder wait on a fence whose update was never
// committed. The submission must still produce a triangle hit.
async fn run_iterations_with_published_then_abandoned_producer(
    ctx: TestingContext,
    discard_encoder: bool,
) {
    let pipeline = create_ray_query_pipeline(&ctx);

    let blit_src = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("producer blit source"),
        contents: &[0; 16],
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let blit_dst = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("producer blit destination"),
        size: blit_src.size(),
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for i in 0..ITERATIONS {
        let setup = create_trace_setup(&ctx, &pipeline);

        let mut producer = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let blas_entry = blas_build_entry(&setup);
        producer.build_acceleration_structures([&blas_entry], []);
        producer.copy_buffer_to_buffer(&blit_src, 0, &blit_dst, 0, blit_src.size());

        if discard_encoder {
            drop(producer);
        } else {
            drop(producer.finish());
        }

        // Recorded only after the producer is gone, on the producer's
        // recycled encoder.
        let mut consumer = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let blas_entry = blas_build_entry(&setup);
        consumer.build_acceleration_structures([&blas_entry], []);
        record_tlas_trace_and_copy(&setup, &pipeline, &mut consumer);

        ctx.queue.submit([consumer.finish()]);

        let hit_kind = read_hit_kind(&ctx, &setup).await;
        assert_eq!(
            hit_kind, TRIANGLE_HIT_KIND,
            "iteration {i}: ray query did not commit a triangle hit"
        );
    }
}

// A blit encoded between two builds closes the acceleration structure encoder
// they would otherwise share; the builds must still be ordered.
async fn run_iterations_with_blit_between_builds(ctx: TestingContext) {
    let pipeline = create_ray_query_pipeline(&ctx);

    let blit_src = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("blit source"),
        contents: &[0; 16],
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let blit_dst = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blit destination"),
        size: blit_src.size(),
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for i in 0..ITERATIONS {
        let setup = create_trace_setup(&ctx, &pipeline);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let blas_entry = blas_build_entry(&setup);
        encoder.build_acceleration_structures([&blas_entry], []);

        encoder.copy_buffer_to_buffer(&blit_src, 0, &blit_dst, 0, blit_src.size());

        encoder.build_acceleration_structures([], [&setup.tlas]);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &setup.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &setup.hit_buffer,
            0,
            &setup.read_back,
            0,
            setup.hit_buffer.size(),
        );

        ctx.queue.submit([encoder.finish()]);

        let hit_kind = read_hit_kind(&ctx, &setup).await;
        assert_eq!(
            hit_kind, TRIANGLE_HIT_KIND,
            "iteration {i}: ray query did not commit a triangle hit"
        );
    }
}

// A compute pass encoded between two builds closes the acceleration structure
// encoder they would otherwise share; the builds must still be ordered.
async fn run_iterations_with_compute_pass_between_builds(ctx: TestingContext) {
    let pipeline = create_ray_query_pipeline(&ctx);

    let filler_pipeline = {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("issue_9215_fill.wgsl"));
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    let scratch = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("filler scratch"),
        size: size_of::<u32>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let filler_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &filler_pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: scratch.as_entire_binding(),
        }],
    });

    for i in 0..ITERATIONS {
        let setup = create_trace_setup(&ctx, &pipeline);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let blas_entry = blas_build_entry(&setup);
        encoder.build_acceleration_structures([&blas_entry], []);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&filler_pipeline);
            pass.set_bind_group(0, &filler_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.build_acceleration_structures([], [&setup.tlas]);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &setup.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &setup.hit_buffer,
            0,
            &setup.read_back,
            0,
            setup.hit_buffer.size(),
        );

        ctx.queue.submit([encoder.finish()]);

        let hit_kind = read_hit_kind(&ctx, &setup).await;
        assert_eq!(
            hit_kind, TRIANGLE_HIT_KIND,
            "iteration {i}: ray query did not commit a triangle hit"
        );
    }
}

// A compacted BLAS is produced through the queue's pending writes: the
// compaction copy is encoded onto the internal pending-writes encoder, which
// is committed ahead of the command buffers of the next submission. A queue
// write encoded onto the same encoder closes the acceleration structure
// encoder the compaction copy leaves open, and the consuming TLAS build in
// the next submission must be ordered after the copy.
async fn run_iterations_with_compacted_blas_in_pending_writes(ctx: TestingContext) {
    let pipeline = create_ray_query_pipeline(&ctx);

    let scratch = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("queue write scratch"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for i in 0..ITERATIONS {
        let mut setup = create_trace_setup(&ctx, &pipeline);

        // The BLAS must allow compaction for `compact_blas` to accept it.
        let blas = ctx.device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: None,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD
                    | wgpu::AccelerationStructureFlags::ALLOW_COMPACTION,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![setup.blas_size.clone()],
            },
        );
        setup.blas = blas;
        setup.tlas[0] = Some(wgpu::TlasInstance::new(
            &setup.blas,
            [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            0,
            0xff,
        ));

        // Build the BLAS and wait for it so it can be prepared for compaction.
        let mut blas_encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let blas_entry = blas_build_entry(&setup);
        blas_encoder.build_acceleration_structures([&blas_entry], []);
        ctx.queue.submit([blas_encoder.finish()]);

        let (send, recv) = std::sync::mpsc::channel();
        setup.blas.prepare_compaction_async(move |res| {
            res.unwrap();
            send.send(()).unwrap();
        });
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        recv.recv().unwrap();
        assert!(setup.blas.ready_for_compaction());

        // Encodes the compaction copy onto the pending-writes encoder.
        let compacted = ctx.queue.compact_blas(&setup.blas);

        // Encodes a blit onto the same pending-writes encoder, closing the
        // acceleration structure encoder the compaction copy left open.
        ctx.queue
            .write_buffer(&scratch, 0, bytemuck::cast_slice(&[0_u32; 4]));

        let mut tlas = ctx.device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            max_instances: 1,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        tlas[0] = Some(wgpu::TlasInstance::new(
            &compacted,
            [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            0,
            0xff,
        ));

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: setup.hit_buffer.as_entire_binding(),
                },
            ],
        });

        let mut consumer = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        consumer.build_acceleration_structures([], [&tlas]);
        {
            let mut pass = consumer.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        consumer.copy_buffer_to_buffer(
            &setup.hit_buffer,
            0,
            &setup.read_back,
            0,
            setup.hit_buffer.size(),
        );

        // Commits the pending writes (compaction copy and queue write) ahead
        // of the consumer command buffer.
        ctx.queue.submit([consumer.finish()]);

        let hit_kind = read_hit_kind(&ctx, &setup).await;
        assert_eq!(
            hit_kind, TRIANGLE_HIT_KIND,
            "iteration {i}: ray query did not commit a triangle hit"
        );
    }
}

#[gpu_test]
static BLAS_AND_TLAS_IN_SAME_COMMAND_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(|ctx| async move { run_iterations(ctx, false).await });

#[gpu_test]
static BLAS_AND_TLAS_IN_SEPARATE_COMMAND_BUFFERS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .limits(acceleration_structure_limits())
                .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
        )
        .run_async(|ctx| async move { run_iterations(ctx, true).await });

#[gpu_test]
static ENCODER_DISCARDED_AFTER_BLAS_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(|ctx| async move { run_iterations_with_abandoned_producer(ctx, true).await });

#[gpu_test]
static COMMAND_BUFFER_DROPPED_WITHOUT_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(|ctx| async move { run_iterations_with_abandoned_producer(ctx, false).await });

#[gpu_test]
static ENCODER_DISCARDED_AFTER_PUBLISHED_CLOSE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(|ctx| async move {
        run_iterations_with_published_then_abandoned_producer(ctx, true).await
    });

#[gpu_test]
static COMMAND_BUFFER_DROPPED_AFTER_PUBLISHED_CLOSE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .limits(acceleration_structure_limits())
                .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
        )
        .run_async(|ctx| async move {
            run_iterations_with_published_then_abandoned_producer(ctx, false).await
        });

#[gpu_test]
static BLIT_BETWEEN_BUILDS_IN_SAME_COMMAND_BUFFER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .limits(acceleration_structure_limits())
                .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
        )
        .run_async(|ctx| async move { run_iterations_with_blit_between_builds(ctx).await });

#[gpu_test]
static COMPUTE_PASS_BETWEEN_BUILDS_IN_SAME_COMMAND_BUFFER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .limits(acceleration_structure_limits())
                .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
        )
        .run_async(|ctx| async move { run_iterations_with_compute_pass_between_builds(ctx).await });

#[gpu_test]
static COMPACTED_BLAS_IN_PENDING_WRITES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(
        |ctx| async move { run_iterations_with_compacted_blas_in_pending_writes(ctx).await },
    );
