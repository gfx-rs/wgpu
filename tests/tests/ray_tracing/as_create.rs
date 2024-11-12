use wgpu::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BlasGeometrySizeDescriptors,
    BlasTriangleGeometrySizeDescriptor, CreateBlasDescriptor,
};
use wgpu_macros::gpu_test;
use wgpu_test::{fail, GpuTestConfiguration, TestParameters, TestingContext};
use wgt::{IndexFormat, VertexFormat};

#[gpu_test]
static BLAS_INVALID_VERTEX_FORMAT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE),
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
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE),
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
