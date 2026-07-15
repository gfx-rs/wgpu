use crate::ray_tracing::acceleration_structure_limits;
use wgpu::include_wgsl;
use wgpu::util::{BufferInitDescriptor, DeviceExt, TlasInstancePackParams, TlasInstancePacker};
use wgpu::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, Backends, BindGroupDescriptor, BindGroupEntry,
    BindingResource, Blas, BlasBuildEntry, BlasGeometries, BlasGeometrySizeDescriptors,
    BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, BufferAddress, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    CreateBlasDescriptor, CreateTlasDescriptor, MapMode, PollType, RawTlasInstance, Tlas,
    TlasInstancesBuffer, VertexFormat,
};
use wgpu_macros::gpu_test;
use wgpu_test::{
    fail, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(TLAS_FROM_INSTANCES_BUFFER);
    tests.push(TLAS_FROM_PACKED_INSTANCES_BUFFER);
    tests.push(TLAS_FROM_BUFFER_MULTI_INSTANCE);
    tests.push(TLAS_FROM_BUFFER_REBUILD);
    tests.push(TLAS_FROM_BUFFER_OFFSET);
    tests.push(TLAS_FROM_BUFFER_EMPTY);
    tests.push(TLAS_FROM_BUFFER_TOO_MANY);
    tests.push(TLAS_FROM_BUFFER_MISSING_USAGE);
    tests.push(TLAS_FROM_BUFFER_TOO_SMALL);
    tests.push(TLAS_FROM_BUFFER_UNALIGNED_OFFSET);
}

fn parameters() -> TestParameters {
    TestParameters::default()
        .test_features_limits()
        .limits(acceleration_structure_limits())
        .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        .skip(FailureCase::backend(Backends::METAL))
}

/// Row-major 3x4 affine that only translates along x.
fn translate_x(tx: f32) -> [f32; 12] {
    [1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// Build (and submit) a BLAS containing a single triangle centred on x=0, y=0 at the given `z`.
/// The device address is valid after creation; contents after the submitted build.
fn build_triangle_blas_at_z(ctx: &TestingContext, z: f32) -> Blas {
    let vertices: [[f32; 3]; 3] = [[-0.5, -0.5, z], [0.5, -0.5, z], [0.0, 0.5, z]];
    let vertex_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: BufferUsages::BLAS_INPUT,
    });
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("blas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build blas"),
        });
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &vertex_buf,
                first_vertex: 0,
                vertex_stride: core::mem::size_of::<[f32; 3]>() as BufferAddress,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [],
    );
    ctx.queue.submit([encoder.finish()]);
    blas
}

fn new_tlas(ctx: &TestingContext, max_instances: u32) -> Tlas {
    ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("tlas"),
        max_instances,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    })
}

/// A committed intersection read back from the trace shader (`as_build_from_buffer.wgsl`).
struct Hit {
    kind: u32,
    instance_index: u32,
    custom_data: u32,
    t: f32,
}

impl Hit {
    fn is_miss(&self) -> bool {
        self.kind == 0
    }
}

/// Trace one ray from `origin` (along +z) at `tlas` and read back the committed intersection.
async fn trace_ray(ctx: &TestingContext, tlas: &Tlas, origin: [f32; 3]) -> Hit {
    let out_buf = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("hit"),
        size: 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let origin_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("ray_origin"),
        contents: bytemuck::cast_slice(&[origin[0], origin[1], origin[2], 0.0]),
        usage: BufferUsages::UNIFORM,
    });
    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("as_build_from_buffer.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::AccelerationStructure(tlas),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(out_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(origin_buf.as_entire_buffer_binding()),
            },
        ],
    });
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: 16,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("trace"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, 16);
    ctx.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(MapMode::Read, Result::unwrap);
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    let data = slice.get_mapped_range().unwrap();
    let [kind, instance_index, custom_data, t_bits]: [u32; 4] = bytemuck::pod_read_unaligned(&data);
    Hit {
        kind,
        instance_index,
        custom_data,
        t: f32::from_bits(t_bits),
    }
}

fn instances_buffer(ctx: &TestingContext, instances: &[RawTlasInstance]) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("instances"),
        contents: bytemuck::cast_slice(instances),
        usage: BufferUsages::TLAS_INPUT,
    })
}

/// Build a TLAS directly from a user-filled instance buffer, then trace a ray and confirm the
/// instance is present and its packed fields round-trip.
#[gpu_test]
static TLAS_FROM_INSTANCES_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(tlas_from_instances_buffer);

async fn tlas_from_instances_buffer(ctx: TestingContext) {
    const CUSTOM_DATA: u32 = 42;
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let instance =
        RawTlasInstance::new(translate_x(0.0), CUSTOM_DATA, 0xFF, blas.handle().unwrap());
    let instance_buf = instances_buffer(&ctx, &[instance]);

    let tlas = new_tlas(&ctx, 1);
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build tlas"),
        });
    encoder.build_tlas_from_instances_buffer(
        &tlas,
        TlasInstancesBuffer {
            buffer: &instance_buf,
            offset: 0,
            count: 1,
            dependencies: &[&blas],
        },
    );
    ctx.queue.submit([encoder.finish()]);

    let hit = trace_ray(&ctx, &tlas, [0.0, 0.0, 0.0]).await;
    assert!(!hit.is_miss(), "ray should hit the triangle");
    assert_eq!(hit.instance_index, 0);
    assert_eq!(
        hit.custom_data, CUSTOM_DATA,
        "custom_data should round-trip"
    );
    assert!((hit.t - 1.0).abs() < 1e-3, "t ≈ 1, got {}", hit.t);
}

/// Same, but the instance buffer is produced on the GPU by `wgpu::util::TlasInstancePacker`
/// exercising the packer and the automatic storage-write -> TLAS-input barrier.
#[gpu_test]
static TLAS_FROM_PACKED_INSTANCES_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(tlas_from_packed_instances_buffer);

async fn tlas_from_packed_instances_buffer(ctx: TestingContext) {
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let tlas = pack_and_build(&ctx, &[&blas], &[translate_x(0.0)], &[0], &[7], 1);
    let hit = trace_ray(&ctx, &tlas, [0.0, 0.0, 0.0]).await;
    assert!(!hit.is_miss());
    assert_eq!(hit.instance_index, 0);
    assert_eq!(
        hit.custom_data, 7,
        "packer should write the supplied custom_data"
    );
    assert!((hit.t - 1.0).abs() < 1e-3);
}

/// Multiple instances referencing two different BLASes at different depths, each placed at a
/// different world position by its transform. Verifies per-instance transform, ordering
/// (`instance_index`), per-instance `custom_data`, and, because BLAS B sits at a different depth,
/// that each instance's `blas_index` resolves to the correct BLAS device address.
#[gpu_test]
static TLAS_FROM_BUFFER_MULTI_INSTANCE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(tlas_from_buffer_multi_instance);

async fn tlas_from_buffer_multi_instance(ctx: TestingContext) {
    let blas_a = build_triangle_blas_at_z(&ctx, 1.0);
    let blas_b = build_triangle_blas_at_z(&ctx, 2.0);

    // instance 0 -> BLAS A at x=-2 (t=1), 1 -> BLAS B at x=0 (t=2), 2 -> BLAS A at x=+2 (t=1).
    let tlas = pack_and_build(
        &ctx,
        &[&blas_a, &blas_b],
        &[translate_x(-2.0), translate_x(0.0), translate_x(2.0)],
        &[0, 1, 0],
        &[10, 20, 30],
        3,
    );

    for (i, (x, expected_t, expected_custom)) in
        [(-2.0, 1.0, 10u32), (0.0, 2.0, 20), (2.0, 1.0, 30)]
            .into_iter()
            .enumerate()
    {
        let hit = trace_ray(&ctx, &tlas, [x, 0.0, 0.0]).await;
        assert!(!hit.is_miss(), "instance {i} at x={x} should be hit");
        assert_eq!(hit.instance_index, i as u32, "instance order");
        assert_eq!(
            hit.custom_data, expected_custom,
            "custom_data of instance {i}"
        );
        assert!(
            (hit.t - expected_t).abs() < 1e-3,
            "instance {i} should resolve to the right BLAS (t={expected_t}), got {}",
            hit.t
        );
    }

    // A ray through the gap between instances hits nothing.
    let miss = trace_ray(&ctx, &tlas, [1.0, 0.0, 0.0]).await;
    assert!(
        miss.is_miss(),
        "ray through the gap should miss, got kind {}",
        miss.kind
    );
}

/// Rebuild the *same* TLAS from a smaller instance buffer (removing the middle instance) and
/// confirm the removed instance is gone, the survivors are re-indexed, and `count < max_instances`
/// is honoured.
#[gpu_test]
static TLAS_FROM_BUFFER_REBUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(tlas_from_buffer_rebuild);

async fn tlas_from_buffer_rebuild(ctx: TestingContext) {
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let addr = blas.handle().unwrap();
    let tlas = new_tlas(&ctx, 3);

    // Build 1: three instances at x=-2, 0, +2.
    let three = instances_buffer(
        &ctx,
        &[
            RawTlasInstance::new(translate_x(-2.0), 100, 0xFF, addr),
            RawTlasInstance::new(translate_x(0.0), 200, 0xFF, addr),
            RawTlasInstance::new(translate_x(2.0), 300, 0xFF, addr),
        ],
    );
    build_from(&ctx, &tlas, &three, 3, &[&blas]);

    for (x, custom, idx) in [(-2.0, 100u32, 0u32), (0.0, 200, 1), (2.0, 300, 2)] {
        let hit = trace_ray(&ctx, &tlas, [x, 0.0, 0.0]).await;
        assert!(!hit.is_miss(), "x={x} should hit before removal");
        assert_eq!(hit.instance_index, idx);
        assert_eq!(hit.custom_data, custom);
    }

    // Build 2: rebuild the SAME TLAS with the middle instance removed (count 2 < max 3).
    let two = instances_buffer(
        &ctx,
        &[
            RawTlasInstance::new(translate_x(-2.0), 100, 0xFF, addr),
            RawTlasInstance::new(translate_x(2.0), 300, 0xFF, addr),
        ],
    );
    build_from(&ctx, &tlas, &two, 2, &[&blas]);

    let removed = trace_ray(&ctx, &tlas, [0.0, 0.0, 0.0]).await;
    assert!(
        removed.is_miss(),
        "removed instance should no longer be hit"
    );

    let left = trace_ray(&ctx, &tlas, [-2.0, 0.0, 0.0]).await;
    assert_eq!(left.custom_data, 100);
    assert_eq!(left.instance_index, 0, "survivor keeps index 0");

    let right = trace_ray(&ctx, &tlas, [2.0, 0.0, 0.0]).await;
    assert_eq!(right.custom_data, 300);
    assert_eq!(
        right.instance_index, 1,
        "survivor is re-indexed from 2 to 1"
    );
}

/// A non-zero `offset` selects a later instance record in the buffer: the TLAS should contain the
/// instance at `offset`, not the one before it.
#[gpu_test]
static TLAS_FROM_BUFFER_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(tlas_from_buffer_offset);

async fn tlas_from_buffer_offset(ctx: TestingContext) {
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let addr = blas.handle().unwrap();
    // Two records; build the TLAS from the second one only.
    let buffer = instances_buffer(
        &ctx,
        &[
            RawTlasInstance::new(translate_x(-2.0), 111, 0xFF, addr),
            RawTlasInstance::new(translate_x(0.0), 222, 0xFF, addr),
        ],
    );
    let tlas = new_tlas(&ctx, 1);
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build tlas"),
        });
    encoder.build_tlas_from_instances_buffer(
        &tlas,
        TlasInstancesBuffer {
            buffer: &buffer,
            offset: core::mem::size_of::<RawTlasInstance>() as u64,
            count: 1,
            dependencies: &[&blas],
        },
    );
    ctx.queue.submit([encoder.finish()]);

    let at_second = trace_ray(&ctx, &tlas, [0.0, 0.0, 0.0]).await;
    assert!(
        !at_second.is_miss(),
        "the instance at `offset` should be present"
    );
    assert_eq!(
        at_second.custom_data, 222,
        "offset should select the second record"
    );

    let at_first = trace_ray(&ctx, &tlas, [-2.0, 0.0, 0.0]).await;
    assert!(
        at_first.is_miss(),
        "the record before `offset` should be excluded"
    );
}

/// Rebuilding a TLAS with `count == 0` empties it: nothing is hit afterwards.
#[gpu_test]
static TLAS_FROM_BUFFER_EMPTY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(tlas_from_buffer_empty);

async fn tlas_from_buffer_empty(ctx: TestingContext) {
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let addr = blas.handle().unwrap();
    let tlas = new_tlas(&ctx, 1);

    let buffer = instances_buffer(
        &ctx,
        &[RawTlasInstance::new(translate_x(0.0), 5, 0xFF, addr)],
    );
    build_from(&ctx, &tlas, &buffer, 1, &[&blas]);
    assert!(
        !trace_ray(&ctx, &tlas, [0.0, 0.0, 0.0]).await.is_miss(),
        "should hit before the TLAS is emptied"
    );

    // Rebuild the same TLAS with no instances.
    build_from(&ctx, &tlas, &buffer, 0, &[]);
    assert!(
        trace_ray(&ctx, &tlas, [0.0, 0.0, 0.0]).await.is_miss(),
        "an empty TLAS should hit nothing"
    );
}

/// Building with more instances than the TLAS's `max_instances` is a validation error.
#[gpu_test]
static TLAS_FROM_BUFFER_TOO_MANY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_sync(tlas_from_buffer_too_many);

fn tlas_from_buffer_too_many(ctx: TestingContext) {
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let addr = blas.handle().unwrap();
    let tlas = new_tlas(&ctx, 1); // capacity for one instance
    let buffer = instances_buffer(
        &ctx,
        &[
            RawTlasInstance::new(translate_x(0.0), 1, 0xFF, addr),
            RawTlasInstance::new(translate_x(2.0), 2, 0xFF, addr),
        ],
    );
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build tlas"),
        });
    encoder.build_tlas_from_instances_buffer(
        &tlas,
        TlasInstancesBuffer {
            buffer: &buffer,
            offset: 0,
            count: 2, // exceeds max_instances
            dependencies: &[&blas],
        },
    );
    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("active instances"),
    );
}

/// Building with an instance buffer that was not created with `TLAS_INPUT` usage is a validation
/// error.
#[gpu_test]
static TLAS_FROM_BUFFER_MISSING_USAGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_sync(tlas_from_buffer_missing_usage);

fn tlas_from_buffer_missing_usage(ctx: TestingContext) {
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let addr = blas.handle().unwrap();
    let tlas = new_tlas(&ctx, 1);
    // Created with STORAGE instead of TLAS_INPUT.
    let buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("instances (wrong usage)"),
        contents: bytemuck::cast_slice(&[RawTlasInstance::new(translate_x(0.0), 1, 0xFF, addr)]),
        usage: BufferUsages::STORAGE,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build tlas"),
        });
    encoder.build_tlas_from_instances_buffer(
        &tlas,
        TlasInstancesBuffer {
            buffer: &buffer,
            offset: 0,
            count: 1,
            dependencies: &[&blas],
        },
    );
    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("TLAS_INPUT"),
    );
}

/// Building from a buffer too small to hold `count` instance records is a validation error.
#[gpu_test]
static TLAS_FROM_BUFFER_TOO_SMALL: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_sync(tlas_from_buffer_too_small);

fn tlas_from_buffer_too_small(ctx: TestingContext) {
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let tlas = new_tlas(&ctx, 1);
    // Room for less than a single 64-byte `RawTlasInstance` record.
    let buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("instances (too small)"),
        size: 32,
        usage: BufferUsages::TLAS_INPUT,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build tlas"),
        });
    encoder.build_tlas_from_instances_buffer(
        &tlas,
        TlasInstancesBuffer {
            buffer: &buffer,
            offset: 0,
            count: 1,
            dependencies: &[&blas],
        },
    );
    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("overrun"),
    );
}

/// A non-16-aligned `offset` is a validation error: the acceleration structure instances data
/// address must be 16-byte aligned on Vulkan and DX12.
#[gpu_test]
static TLAS_FROM_BUFFER_UNALIGNED_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_sync(tlas_from_buffer_unaligned_offset);

fn tlas_from_buffer_unaligned_offset(ctx: TestingContext) {
    let blas = build_triangle_blas_at_z(&ctx, 1.0);
    let addr = blas.handle().unwrap();
    let tlas = new_tlas(&ctx, 1);
    // Two records' worth of space so the (misaligned) region is still in bounds and the failure is
    // unambiguously the alignment check.
    let buffer = instances_buffer(
        &ctx,
        &[
            RawTlasInstance::new(translate_x(0.0), 1, 0xFF, addr),
            RawTlasInstance::new(translate_x(2.0), 2, 0xFF, addr),
        ],
    );
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build tlas"),
        });
    encoder.build_tlas_from_instances_buffer(
        &tlas,
        TlasInstancesBuffer {
            buffer: &buffer,
            offset: 8, // not a multiple of 16
            count: 1,
            dependencies: &[&blas],
        },
    );
    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("multiple of 16"),
    );
}

/// Submit a single `build_tlas_from_instances_buffer` for `tlas` from `buffer`.
fn build_from(
    ctx: &TestingContext,
    tlas: &Tlas,
    buffer: &wgpu::Buffer,
    count: u32,
    deps: &[&Blas],
) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build tlas"),
        });
    encoder.build_tlas_from_instances_buffer(
        tlas,
        TlasInstancesBuffer {
            buffer,
            offset: 0,
            count,
            dependencies: deps,
        },
    );
    ctx.queue.submit([encoder.finish()]);
}

/// GPU-pack `count` instances (via `TlasInstancePacker`) from the given transforms / BLAS indices /
/// custom data + a BLAS address table, then build a fresh TLAS from the packed buffer in the same
/// encoder.
fn pack_and_build(
    ctx: &TestingContext,
    blases: &[&Blas],
    transforms: &[[f32; 12]],
    blas_indices: &[u32],
    custom_data: &[u32],
    count: u32,
) -> Tlas {
    let transforms_flat: Vec<f32> = transforms.iter().flatten().copied().collect();
    let addresses: Vec<u64> = blases.iter().map(|b| b.handle().unwrap()).collect();

    let transforms_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("transforms"),
        contents: bytemuck::cast_slice(&transforms_flat),
        usage: BufferUsages::STORAGE,
    });
    let indices_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("blas_indices"),
        contents: bytemuck::cast_slice(blas_indices),
        usage: BufferUsages::STORAGE,
    });
    let addresses_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("blas_addresses"),
        contents: bytemuck::cast_slice(&addresses),
        usage: BufferUsages::STORAGE,
    });
    let custom_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("custom_data"),
        contents: bytemuck::cast_slice(custom_data),
        usage: BufferUsages::STORAGE,
    });
    let instance_buf = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("instances"),
        size: count as u64 * core::mem::size_of::<RawTlasInstance>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::TLAS_INPUT,
        mapped_at_creation: false,
    });

    let packer = TlasInstancePacker::new(&ctx.device);
    let tlas = new_tlas(ctx, count);
    let deps: Vec<&Blas> = blases.to_vec();

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("pack + build"),
        });
    packer.pack(
        &ctx.device,
        &mut encoder,
        &TlasInstancePackParams {
            transforms: &transforms_buf,
            blas_indices: &indices_buf,
            blas_addresses: &addresses_buf,
            custom_data: &custom_buf,
            instances: &instance_buf,
            count,
        },
    );
    encoder.build_tlas_from_instances_buffer(
        &tlas,
        TlasInstancesBuffer {
            buffer: &instance_buf,
            offset: 0,
            count,
            dependencies: &deps,
        },
    );
    ctx.queue.submit([encoder.finish()]);
    tlas
}
