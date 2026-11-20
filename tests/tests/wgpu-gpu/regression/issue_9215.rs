//! Acceleration structure commands must be ordered after the builds they
//! consume. The BLAS and TLAS are built by two `build_acceleration_structures`
//! calls. Before the TLAS build, wgpu-core uploads the TLAS instance buffer
//! with `copy_buffer_to_buffer`, which on Metal ends the BLAS's acceleration
//! structure encoder, so the two builds land in different acceleration
//! structure encoders of the same command buffer. Metal does not order them,
//! because the TLAS build references the BLAS only through instance resource
//! IDs and never binds it. A TLAS build could therefore consume a BLAS that is
//! still building, and the ray query below would commit no intersection
//! (kind 0) instead of the triangle (kind 1).
//!
//! Each iteration builds a fresh BLAS and a TLAS referencing it, traces a ray
//! straight up through the triangle, and reads back the committed
//! intersection.
//!
//! See <https://github.com/gfx-rs/wgpu/issues/9215>.

use std::time::Duration;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        BLAS_AND_TLAS_IN_SAME_COMMAND_BUFFER,
        COMPACT_SIZE_READ_AFTER_BUILD,
        FIRST_BUILD_AFTER_COMPUTE_UPLOAD,
    ]);
}

// The failures are races, so a pass is evidence rather than proof. On the
// hardware this bug reproduces on (Apple M4 Max), the unfixed failure rate of
// the affected configurations is tens of percent per iteration.
const ITERATIONS: u32 = 10;

// `rayQueryGetCommittedIntersection().kind` for a committed triangle hit.
const TRIANGLE_HIT_KIND: u32 = 1;

const IDENTITY_TRANSFORM: [f32; 12] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0,
];

fn acceleration_structure_parameters() -> TestParameters {
    TestParameters::default()
        .test_features_limits()
        .limits(wgpu::Limits::default().using_minimum_supported_acceleration_structure_values())
        .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

struct TraceSetup {
    vertex_buffer: wgpu::Buffer,
    blas_size: wgpu::BlasTriangleGeometrySizeDescriptor,
    blas: wgpu::Blas,
    tlas: wgpu::Tlas,
    hit_buffer: wgpu::Buffer,
    read_back: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn create_trace_setup(
    ctx: &TestingContext,
    pipeline: &wgpu::ComputePipeline,
    blas_flags: wgpu::AccelerationStructureFlags,
) -> TraceSetup {
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

    let blas = ctx.device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD | blas_flags,
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
        IDENTITY_TRANSFORM,
        0,
        0xff,
        0,
    ));

    let hit_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hit kind"),
        size: 64,
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

/// Records the TLAS build, the ray query, and the readback copy onto `encoder`.
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

async fn wait_idle(ctx: &TestingContext) {
    ctx.async_poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    })
    .await
    .unwrap();
}

/// Reads back the trace and asserts that the ray through the triangle hit it.
async fn assert_triangle_hit(ctx: &TestingContext, setup: &TraceSetup, iteration: u32) {
    let slice = setup.read_back.slice(..);
    slice.map_async(wgpu::MapMode::Read, Result::unwrap);
    wait_idle(ctx).await;

    {
        let view = slice.get_mapped_range().unwrap();
        let words: &[u32] = bytemuck::cast_slice(&view);
        assert_eq!(words[8], 0, "iteration {iteration}: known-miss control");
        assert_eq!(
            words[0], TRIANGLE_HIT_KIND,
            "iteration {iteration}: ray query did not commit a triangle hit"
        );
        assert!((f32::from_bits(words[1]) - 1.0).abs() < 1e-5);
        assert_eq!(&words[2..6], &[0, 0, 0, 0], "hit identifiers");
        for &barycentric in &words[6..8] {
            assert!((f32::from_bits(barycentric) - 0.25).abs() < 1e-5);
        }
    }
    setup.read_back.unmap();
}

fn create_compute_pipeline(
    ctx: &TestingContext,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
}

fn create_ray_query_pipeline(ctx: &TestingContext) -> wgpu::ComputePipeline {
    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("issue_9215.wgsl"));
    create_compute_pipeline(ctx, &shader, "trace")
}

async fn build_and_trace(ctx: TestingContext) {
    let pipeline = create_ray_query_pipeline(&ctx);

    for i in 0..ITERATIONS {
        let setup = create_trace_setup(&ctx, &pipeline, wgpu::AccelerationStructureFlags::empty());

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.build_acceleration_structures([&blas_build_entry(&setup)], []);
        record_tlas_trace_and_copy(&setup, &pipeline, &mut encoder);
        ctx.queue.submit([encoder.finish()]);

        assert_triangle_hit(&ctx, &setup, i).await;
    }
}

#[apply(gpu_test!)]
static BLAS_AND_TLAS_IN_SAME_COMMAND_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(acceleration_structure_parameters())
    .run_async(build_and_trace);

// wgpu-core encodes the compacted size query of a BLAS built with
// `ALLOW_COMPACTION` right after the build, in the same command buffer. If the
// query runs before the build completes, the size used for compaction is wrong
// and the compacted BLAS does not contain the triangle.
//
// This test passes on unfixed trunk, so it does not reproduce a failure. It is
// regression coverage for the compacted size read being ordered after the
// build.
#[apply(gpu_test!)]
static COMPACT_SIZE_READ_AFTER_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(acceleration_structure_parameters())
    .run_async(|ctx| async move {
        let pipeline = create_ray_query_pipeline(&ctx);

        for i in 0..ITERATIONS {
            let mut setup = create_trace_setup(
                &ctx,
                &pipeline,
                wgpu::AccelerationStructureFlags::ALLOW_COMPACTION,
            );

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.build_acceleration_structures([&blas_build_entry(&setup)], []);
            ctx.queue.submit([encoder.finish()]);

            let (send, recv) = std::sync::mpsc::channel();
            setup.blas.prepare_compaction_async(move |res| {
                res.unwrap();
                send.send(()).unwrap();
            });
            wait_idle(&ctx).await;
            recv.recv_timeout(Duration::from_secs(30)).unwrap();
            assert!(setup.blas.ready_for_compaction());

            let compacted = ctx.queue.compact_blas(&setup.blas);
            setup.tlas[0] = Some(wgpu::TlasInstance::new(
                &compacted,
                IDENTITY_TRANSFORM,
                0,
                0xff,
                0,
            ));

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            record_tlas_trace_and_copy(&setup, &pipeline, &mut encoder);
            ctx.queue.submit([encoder.finish()]);

            assert_triangle_hit(&ctx, &setup, i).await;
        }
    });

// Same BLAS/TLAS build ordering as `BLAS_AND_TLAS_IN_SAME_COMMAND_BUFFER`, but
// with BLAS vertices generated by a compute pass in an earlier submission, so
// the build input is a storage buffer rather than an initialized one. The
// compute write itself is already ordered by Metal's tracking of the vertex
// buffer; on unfixed trunk this test fails because the TLAS build, which is in
// a different acceleration structure encoder of the same command buffer, races
// the BLAS build.
#[apply(gpu_test!)]
static FIRST_BUILD_AFTER_COMPUTE_UPLOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(acceleration_structure_parameters())
    .run_async(|ctx| async move {
        let pipeline = create_ray_query_pipeline(&ctx);
        let generate = create_compute_pipeline(
            &ctx,
            &ctx.device
                .create_shader_module(wgpu::include_wgsl!("issue_9215_vertices.wgsl")),
            "generate_vertices",
        );

        for i in 0..ITERATIONS {
            let mut setup =
                create_trace_setup(&ctx, &pipeline, wgpu::AccelerationStructureFlags::empty());
            setup.vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute generated BLAS vertices"),
                size: 36,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
                mapped_at_creation: false,
            });
            let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &generate.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: setup.vertex_buffer.as_entire_binding(),
                }],
            });

            let mut upload = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = upload.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&generate);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit([upload.finish()]);

            let mut build = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            build.build_acceleration_structures([&blas_build_entry(&setup)], []);
            record_tlas_trace_and_copy(&setup, &pipeline, &mut build);
            ctx.queue.submit([build.finish()]);

            assert_triangle_hit(&ctx, &setup, i).await;
        }
    });
