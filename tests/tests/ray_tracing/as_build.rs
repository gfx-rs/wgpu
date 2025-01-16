use std::iter;

use crate::ray_tracing::AsBuildContext;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_test::{
    fail, gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext,
};

#[gpu_test]
static UNBUILT_BLAS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(unbuilt_blas);

fn unbuilt_blas(ctx: TestingContext) {
    let as_ctx = AsBuildContext::new(&ctx);

    // Build the TLAS package with an unbuilt BLAS.
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());

    encoder.build_acceleration_structures([], [&as_ctx.tlas_package]);

    fail(
        &ctx.device,
        || {
            ctx.queue.submit([encoder.finish()]);
        },
        None,
    );
}

#[gpu_test]
static OUT_OF_ORDER_AS_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(out_of_order_as_build);

fn out_of_order_as_build(ctx: TestingContext) {
    let as_ctx = AsBuildContext::new(&ctx);

    //
    // Encode the TLAS build before the BLAS build, but submit them in the right order.
    //

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 1"),
        });

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas_package]);

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        });

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []);

    ctx.queue
        .submit([encoder_blas.finish(), encoder_tlas.finish()]);

    drop(as_ctx);

    //
    // Create a clean `AsBuildContext`
    //

    let as_ctx = AsBuildContext::new(&ctx);

    //
    // Encode the BLAS build before the TLAS build, but submit them in the wrong order.
    //

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 2"),
        });

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []);

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 2"),
        });

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas_package]);

    fail(
        &ctx.device,
        || {
            ctx.queue
                .submit([encoder_tlas.finish(), encoder_blas.finish()]);
        },
        None,
    );
}

#[gpu_test]
static OUT_OF_ORDER_AS_BUILD_USE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
                    | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            )
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(out_of_order_as_build_use);

fn out_of_order_as_build_use(ctx: TestingContext) {
    //
    // Create a clean `AsBuildContext`
    //

    let as_ctx = AsBuildContext::new(&ctx);

    //
    // Build in the right order, then rebuild the BLAS so the TLAS is invalid, then use the TLAS.
    //

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        });

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []);

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 1"),
        });

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas_package]);

    let mut encoder_blas2 = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 2"),
        });

    encoder_blas2.build_acceleration_structures([&as_ctx.blas_build_entry()], []);

    ctx.queue.submit([
        encoder_blas.finish(),
        encoder_tlas.finish(),
        encoder_blas2.finish(),
    ]);

    //
    // Create shader to use tlas with
    //

    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("shader.wgsl"));
    let compute_pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("basic_usage"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::AccelerationStructure(as_ctx.tlas_package.tlas()),
        }],
    });

    //
    // Use TLAS
    //

    let mut encoder_compute = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder_compute.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1)
    }

    fail(
        &ctx.device,
        || {
            ctx.queue.submit(Some(encoder_compute.finish()));
        },
        None,
    );
}

#[gpu_test]
static EMPTY_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE),
    )
    .run_sync(empty_build);
fn empty_build(ctx: TestingContext) {
    let mut encoder_safe = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        });

    encoder_safe.build_acceleration_structures(iter::empty(), iter::empty());

    let mut encoder_unsafe = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        });

    // # SAFETY:
    // we don't actually do anything so all the requirements are satisfied
    unsafe {
        encoder_unsafe.build_acceleration_structures_unsafe_tlas(iter::empty(), iter::empty());
    }

    ctx.queue
        .submit([encoder_safe.finish(), encoder_unsafe.finish()]);
}

#[gpu_test]
static BUILD_WITH_TRANSFORM: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(build_with_transform);

fn build_with_transform(ctx: TestingContext) {
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; size_of::<[[f32; 3]; 3]>()],
        usage: BufferUsages::BLAS_INPUT,
    });

    let transform = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
        });

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::empty(),
    };

    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    let tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("TLAS"),
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });

    let mut tlas_package = TlasPackage::new(tlas);
    tlas_package[0] = Some(TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        0,
        0xFF,
    ));

    let mut encoder_build = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BUILD 1"),
        });

    encoder_build.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: size_of::<[f32; 3]>() as BufferAddress,
                index_buffer: None,
                first_index: None,
                transform_buffer: Some(&transform),
                transform_buffer_offset: Some(0),
            }]),
        }],
        [&tlas_package],
    );
    ctx.queue.submit([encoder_build.finish()]);
}
