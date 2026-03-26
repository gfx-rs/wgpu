use std::mem::size_of;

use crate::ray_tracing::acceleration_structure_limits;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    include_wgsl, AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BlasAABBGeometrySizeDescriptor, BlasAabbGeometry,
    BlasBuildEntry, BlasGeometries, BlasGeometrySizeDescriptors, BlasTriangleGeometry,
    BlasTriangleGeometrySizeDescriptor, BufferAddress, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, CreateBlasDescriptor, CreateTlasDescriptor,
    TlasInstance, VertexFormat, AABB_GEOMETRY_MIN_STRIDE,
};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindingResource, BufferDescriptor};
use wgpu_test::{
    fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AabbPrimitive {
    min: [f32; 3],
    max: [f32; 3],
}

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        AABB_BLAS_BUILD_AND_TRACE,
        AABB_INVALID_STRIDE_CREATE,
        AABB_DIFFERENT_STRIDE_BUILD,
        AABB_UNALIGNED_PRIMITIVE_OFFSET,
        AABB_GEOMETRY_KIND_MISMATCH,
        AABB_INSUFFICIENT_BUFFER,
        AABB_PRIMITIVE_COUNT_EXCEEDS_CREATION,
        AABB_FLAGS_MISMATCH,
    ]);
}

// OPAQUE geometry so BLAS primitives are visible to `ray_query` in WGSL.
fn aabb_size_desc(stride: BufferAddress, primitive_count: u32) -> BlasAABBGeometrySizeDescriptor {
    BlasAABBGeometrySizeDescriptor {
        primitive_count,
        stride,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    }
}

#[gpu_test]
static AABB_BLAS_BUILD_AND_TRACE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_sync(aabb_blas_build_and_trace);

fn aabb_blas_build_and_trace(ctx: TestingContext) {
    let stride = AABB_GEOMETRY_MIN_STRIDE;
    let aabb_data = AabbPrimitive {
        min: [-1.0, -1.0, 2.0],
        max: [1.0, 1.0, 4.0],
    };

    let aabb_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&aabb_data),
        usage: BufferUsages::BLAS_INPUT,
    });

    let blas_size = aabb_size_desc(stride, 1);
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("AABB BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![blas_size.clone()],
        },
    );

    let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("TLAS"),
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    tlas[0] = Some(TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        0,
        0xFF,
    ));

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build"),
        });
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                size: &blas_size,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0,
            }]),
        }],
        [&tlas],
    );
    ctx.queue.submit([encoder.finish()]);

    let out_buf = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 176,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("shader.wgsl"));
    let compute_pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("all_of_struct"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::AccelerationStructure(&tlas),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(out_buf.as_entire_buffer_binding()),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);
}

#[gpu_test]
static AABB_INVALID_STRIDE_CREATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(aabb_invalid_stride_create);

/// Create a BLAS with AABB stride below the minimum.
fn aabb_invalid_stride_create(ctx: TestingContext) {
    let bad_stride = 16u64;
    fail(
        &ctx.device,
        || {
            let _ = ctx.device.create_blas(
                &CreateBlasDescriptor {
                    label: Some("bad stride"),
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                },
                BlasGeometrySizeDescriptors::AABBs {
                    descriptors: vec![BlasAABBGeometrySizeDescriptor {
                        primitive_count: 1,
                        stride: bad_stride,
                        flags: AccelerationStructureGeometryFlags::OPAQUE,
                    }],
                },
            );
        },
        None,
    );
}

#[gpu_test]
static AABB_DIFFERENT_STRIDE_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(aabb_different_stride_build);

/// Build stride must match the stride given at BLAS creation.
fn aabb_different_stride_build(ctx: TestingContext) {
    let create_stride = AABB_GEOMETRY_MIN_STRIDE;
    let build_stride = create_stride + 8;
    let aabb_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0u8; 64],
        usage: BufferUsages::BLAS_INPUT,
    });

    let create_desc = aabb_size_desc(create_stride, 1);
    let build_desc = aabb_size_desc(build_stride, 1);

    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![create_desc],
        },
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                size: &build_desc,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0,
            }]),
        }],
        [],
    );
    fail(&ctx.device, || encoder.finish(), None);
}

#[gpu_test]
static AABB_UNALIGNED_PRIMITIVE_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(aabb_unaligned_primitive_offset);

/// `primitive_offset` must be a multiple of 8.
fn aabb_unaligned_primitive_offset(ctx: TestingContext) {
    let stride = AABB_GEOMETRY_MIN_STRIDE;
    let aabb_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0u8; 64],
        usage: BufferUsages::BLAS_INPUT,
    });
    let blas_size = aabb_size_desc(stride, 1);
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![blas_size.clone()],
        },
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                size: &blas_size,
                aabb_buffer: &aabb_buf,
                primitive_offset: 4,
            }]),
        }],
        [],
    );
    fail(&ctx.device, || encoder.finish(), None);
}

#[gpu_test]
static AABB_GEOMETRY_KIND_MISMATCH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(aabb_geometry_kind_mismatch);

/// Triangle build entries are invalid for an AABB BLAS and vice versa.
fn aabb_geometry_kind_mismatch(ctx: TestingContext) {
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; size_of::<[[f32; 3]; 3]>()],
        usage: BufferUsages::BLAS_INPUT,
    });
    let stride = AABB_GEOMETRY_MIN_STRIDE;
    let aabb_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0u8; 32],
        usage: BufferUsages::BLAS_INPUT,
    });

    let aabb_blas_size = aabb_size_desc(stride, 1);
    let aabb_blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("AABB BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![aabb_blas_size.clone()],
        },
    );

    let tri_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    // AABB BLAS built with triangle geometry
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &aabb_blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &tri_size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: size_of::<[f32; 3]>() as BufferAddress,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [],
    );
    fail(&ctx.device, || encoder.finish(), None);

    // Triangle BLAS built with AABB geometry
    let tri_blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("tri BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![tri_size.clone()],
        },
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &tri_blas,
            geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                size: &aabb_blas_size,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0,
            }]),
        }],
        [],
    );
    fail(&ctx.device, || encoder.finish(), None);
}

#[gpu_test]
static AABB_INSUFFICIENT_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(aabb_insufficient_buffer);

/// The AABB buffer must cover `primitive_offset + primitive_count * stride`.
fn aabb_insufficient_buffer(ctx: TestingContext) {
    let stride = AABB_GEOMETRY_MIN_STRIDE;
    let small_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0u8; 16],
        usage: BufferUsages::BLAS_INPUT,
    });
    let blas_size = aabb_size_desc(stride, 1);
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![blas_size.clone()],
        },
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                size: &blas_size,
                aabb_buffer: &small_buf,
                primitive_offset: 0,
            }]),
        }],
        [],
    );
    fail(&ctx.device, || encoder.finish(), None);
}

#[gpu_test]
static AABB_PRIMITIVE_COUNT_EXCEEDS_CREATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(aabb_primitive_count_exceeds_creation);

/// Build cannot request more AABB primitives than the BLAS was created for.
fn aabb_primitive_count_exceeds_creation(ctx: TestingContext) {
    let stride = AABB_GEOMETRY_MIN_STRIDE;
    let aabb_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &vec![0u8; (stride * 2) as usize],
        usage: BufferUsages::BLAS_INPUT,
    });
    let create_desc = aabb_size_desc(stride, 1);
    let build_desc = aabb_size_desc(stride, 2);

    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![create_desc],
        },
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                size: &build_desc,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0,
            }]),
        }],
        [],
    );
    fail(&ctx.device, || encoder.finish(), None);
}

#[gpu_test]
static AABB_FLAGS_MISMATCH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(aabb_flags_mismatch);

/// Per-geometry flags at build time must match those at creation.
fn aabb_flags_mismatch(ctx: TestingContext) {
    let stride = AABB_GEOMETRY_MIN_STRIDE;
    let aabb_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0u8; 32],
        usage: BufferUsages::BLAS_INPUT,
    });
    let create_desc = BlasAABBGeometrySizeDescriptor {
        primitive_count: 1,
        stride,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let build_desc = BlasAABBGeometrySizeDescriptor {
        primitive_count: 1,
        stride,
        flags: AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION,
    };

    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![create_desc.clone()],
        },
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                size: &build_desc,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0,
            }]),
        }],
        [],
    );
    fail(&ctx.device, || encoder.finish(), None);
}
