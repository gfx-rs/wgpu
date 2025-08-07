use crate::ray_tracing::acceleration_structure_limits;
use wgpu::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BlasGeometrySizeDescriptors,
    BlasTriangleGeometrySizeDescriptor, CreateBlasDescriptor,
};
use wgpu::{IndexFormat, VertexFormat};
use wgpu_macros::gpu_test;
use wgpu_test::{fail, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(tests: &mut Vec<wgpu_test::GpuTestInitializer>) {
    tests.extend([
        BLAS_INVALID_VERTEX_FORMAT,
        BLAS_MISMATCHED_INDEX,
        UNSUPPORTED_ACCELERATION_STRUCTURE_RESOURCES,
    ]);
}

#[gpu_test]
static BLAS_INVALID_VERTEX_FORMAT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(invalid_vertex_format_blas_create);

fn invalid_vertex_format_blas_create(ctx: TestingContext) {
    //
    // Create a BLAS with a format that is not allowed
    //

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x4,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::empty(),
    };

    fail(
        &ctx.device,
        || {
            let _ = ctx.device.create_blas(
                &CreateBlasDescriptor {
                    label: Some("BLAS"),
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                },
                BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![blas_size.clone()],
                },
            );
        },
        None,
    );
}

#[gpu_test]
static BLAS_MISMATCHED_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .enable_noop(),
    )
    .run_sync(mismatched_index_blas_create);

fn mismatched_index_blas_create(ctx: TestingContext) {
    //
    // Create a BLAS with just an index format
    //

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: Some(IndexFormat::Uint32),
        index_count: None,
        flags: AccelerationStructureGeometryFlags::empty(),
    };

    fail(
        &ctx.device,
        || {
            let _ = ctx.device.create_blas(
                &CreateBlasDescriptor {
                    label: Some("BLAS1"),
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                },
                BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![blas_size.clone()],
                },
            );
        },
        None,
    );

    //
    // Create a BLAS with just an index count
    //

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: Some(3),
        flags: AccelerationStructureGeometryFlags::empty(),
    };

    fail(
        &ctx.device,
        || {
            let _ = ctx.device.create_blas(
                &CreateBlasDescriptor {
                    label: Some("BLAS2"),
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                },
                BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![blas_size.clone()],
                },
            );
        },
        None,
    );
}

#[gpu_test]
static UNSUPPORTED_ACCELERATION_STRUCTURE_RESOURCES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits())
        .run_sync(unsupported_acceleration_structure_resources);

fn unsupported_acceleration_structure_resources(ctx: TestingContext) {
    fail(
        &ctx.device,
        || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::BLAS_INPUT,
                mapped_at_creation: false,
            })
        },
        None,
    );
    fail(
        &ctx.device,
        || {
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    }],
                })
        },
        None,
    );
}
