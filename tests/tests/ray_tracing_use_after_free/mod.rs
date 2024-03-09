use std::iter;
use std::mem::size_of;
use wgpu::include_wgsl;
use wgpu::ray_tracing::{
    AccelerationStructureUpdateMode, BlasBuildEntry, BlasGeometries, BlasTriangleGeometry,
    CommandEncoderRayTracing, CreateBlasDescriptor, CreateTlasDescriptor, DeviceRayTracing,
    TlasInstance, TlasPackage,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, TestParameters, TestingContext};
use wgt::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags, BlasGeometrySizeDescriptors,
    BlasTriangleGeometrySizeDescriptor, BufferUsages, CommandEncoderDescriptor, VertexFormat,
};

fn required_features() -> wgpu::Features {
    wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
        | wgpu::Features::VERTEX_WRITABLE_STORAGE
        | wgpu::Features::RAY_QUERY
        | wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
}

fn execute(ctx: TestingContext) {
    let size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 1,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::empty(),
    };
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("Use after free blas"),
            flags: AccelerationStructureFlags::empty(),
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles { desc: vec![size] },
    );
    let vertex_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; 3 * size_of::<f32>()],
        usage: BufferUsages::BLAS_INPUT,
    });
    let tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("Use after free tlas"),
        max_instances: 1,
        flags: AccelerationStructureFlags::empty(),
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    let mut tlas_package = TlasPackage::new(tlas, 1);
    *tlas_package.get_mut_single(0) = Some(TlasInstance::new(&blas, [0.0; 12], 0, 0));
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        iter::once(&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vertex_buf,
                first_vertex: 0,
                vertex_stride: (3 * size_of::<f32>()) as wgt::BufferAddress,
                index_buffer: None,
                index_buffer_offset: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::empty(),
    );
    ctx.queue.submit(Some(encoder.finish()));
    drop(blas);
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(iter::empty(), iter::once(&tlas_package));
    ctx.queue.submit(Some(encoder.finish()));
    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("compute_usage.wgsl"));
}

#[gpu_test]
static RAY_TRACING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(required_features()),
    )
    .run_sync(execute);
