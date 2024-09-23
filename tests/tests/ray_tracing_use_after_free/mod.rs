use std::{iter, mem};
use wgpu::{
    AccelerationStructureUpdateMode, BlasBuildEntry, BlasGeometries, BlasTriangleGeometry, TlasInstance, TlasPackage,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    include_wgsl, BindGroupDescriptor, BindGroupEntry, BindingResource, ComputePassDescriptor,
    ComputePipelineDescriptor,
};
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, TestParameters, TestingContext};
use wgt::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags, BlasGeometrySizeDescriptors,
    BlasTriangleGeometrySizeDescriptor, BufferAddress, BufferUsages, CommandEncoderDescriptor,
    CreateBlasDescriptor, CreateTlasDescriptor, Maintain, VertexFormat,
};

fn required_features() -> wgpu::Features {
    wgpu::Features::RAY_QUERY | wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
}

/// This test creates a blas, puts a reference to it in a tlas instance inside a tlas package,
/// drops the blas, and ensures it gets kept alive by the tlas instance. Then it uses the built
/// package in a bindgroup, drops it, and checks that it is kept alive by the bindgroup by
/// executing a shader using that bindgroup.
fn execute(ctx: TestingContext) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::empty(),
    };
    // create the blas
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("blas use after free"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );
    // create the tlas and put it in a package
    let tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("tlas use after free"),
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; mem::size_of::<[[f32; 3]; 3]>()],
        usage: BufferUsages::BLAS_INPUT,
    });
    let mut tlas_package = TlasPackage::new(tlas);
    // place blas in tlas instance, then put tlas instance in a tlas package
    *tlas_package.get_mut_single(0).unwrap() = Some(TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        0,
        0xFF,
    ));
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    // build blas to make sure tlas doesn't error when building it
    encoder.build_acceleration_structures(
        iter::once(&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: mem::size_of::<[f32; 3]>() as BufferAddress,
                index_buffer: None,
                index_buffer_offset: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::empty(),
    );
    ctx.queue.submit(Some(encoder.finish()));
    // drop the blas
    drop(blas);
    ctx.device.poll(Maintain::Wait);
    // build the tlas package to ensure the blas is dropped
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(iter::empty(), iter::once(&tlas_package));
    ctx.queue.submit(Some(encoder.finish()));
    // create shader to execute
    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("shader.wgsl"));
    let compute_pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("comp_main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::AccelerationStructure(tlas_package.tlas()),
        }],
    });
    // drop tlas_package
    drop(tlas_package);
    ctx.device.poll(Maintain::Wait);
    // run pass with bindgroup to ensure the tlas was kept alive
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
        pass.dispatch_workgroups(1, 1, 1)
    }
    ctx.queue.submit(Some(encoder.finish()));
}

#[gpu_test]
static RAY_TRACING_USE_AFTER_FREE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(required_features()),
    )
    .run_sync(execute);
