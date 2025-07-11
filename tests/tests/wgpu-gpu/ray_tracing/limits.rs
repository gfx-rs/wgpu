use wgpu::wgt::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BlasGeometrySizeDescriptors,
};
use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BlasTriangleGeometrySizeDescriptor, CreateBlasDescriptor, CreateTlasDescriptor, Limits,
    ShaderStages, VertexFormat,
};
use wgpu_macros::gpu_test;
use wgpu_test::{fail, GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static LIMITS_HIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(Limits {
                max_blas_primitive_count: 3,
                max_blas_geometry_count: 1,
                max_tlas_instance_count: 1,
                max_acceleration_structures_per_shader_stage: 1,
                ..Limits::default()
            })
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_sync(hit_limits);

fn hit_limits(ctx: TestingContext) {
    fail(
        &ctx.device,
        || {
            let _ = ctx.device.create_blas(
                &CreateBlasDescriptor {
                    label: None,
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                },
                BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![
                        BlasTriangleGeometrySizeDescriptor {
                            vertex_format: VertexFormat::Float32x3,
                            vertex_count: 3,
                            index_format: None,
                            index_count: None,
                            flags: AccelerationStructureGeometryFlags::empty(),
                        };
                        2
                    ],
                },
            );
        },
        None,
    );
    fail(
        &ctx.device,
        || {
            let _ = ctx.device.create_blas(
                &CreateBlasDescriptor {
                    label: None,
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                },
                BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![BlasTriangleGeometrySizeDescriptor {
                        vertex_format: VertexFormat::Float32x3,
                        vertex_count: 6,
                        index_format: None,
                        index_count: None,
                        flags: AccelerationStructureGeometryFlags::empty(),
                    }],
                },
            );
        },
        None,
    );
    fail(
        &ctx.device,
        || {
            let _ = ctx.device.create_tlas(&CreateTlasDescriptor {
                label: None,
                max_instances: 2,
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            });
        },
        None,
    );
    fail(
        &ctx.device,
        || {
            let _ = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::AccelerationStructure {
                                vertex_return: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::AccelerationStructure {
                                vertex_return: false,
                            },
                            count: None,
                        },
                    ],
                });
        },
        None,
    );
}
