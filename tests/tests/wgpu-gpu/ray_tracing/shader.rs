use crate::ray_tracing::{acceleration_structure_limits, AsBuildContext};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    include_wgsl, BindGroupDescriptor, BindGroupEntry, BindingResource, BufferDescriptor,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, InstanceFlags,
};
use wgpu::{AccelerationStructureFlags, BufferUsages};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

const STRUCT_SIZE: wgpu::BufferAddress = 176;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(ACCESS_ALL_STRUCT_MEMBERS);
    tests.push(PREVENT_INVALID_RAY_QUERY_CALLS);
    tests.push(SPARSE_INSTANCES_INDEX_CORRECT);
}

#[apply(gpu_test!)]
static ACCESS_ALL_STRUCT_MEMBERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_sync(access_all_struct_members);

fn access_all_struct_members(ctx: TestingContext) {
    let buf = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: STRUCT_SIZE,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    //
    // Create a clean `AsBuildContext`
    //

    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    let mut encoder_build = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Build"),
        });

    encoder_build.build_acceleration_structures([&as_ctx.blas_build_entry()], [&as_ctx.tlas]);

    ctx.queue.submit([encoder_build.finish()]);

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
                resource: BindingResource::AccelerationStructure(&as_ctx.tlas),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(buf.as_entire_buffer_binding()),
            },
        ],
    });

    //
    // Submit once to check for no issues
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

    ctx.queue.submit([encoder_compute.finish()]);
}

#[apply(gpu_test!)]
static PREVENT_INVALID_RAY_QUERY_CALLS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        // Don't disable metal shader validation in `TestParameters` because
        // the shader validation is important here and the results from the
        // ray query don't matter.
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            // Otherwise, mistakes in the generated code won't be caught.
            .instance_flags(InstanceFlags::GPU_BASED_VALIDATION),
    )
    .run_sync(prevent_invalid_ray_query_calls);

fn prevent_invalid_ray_query_calls(ctx: TestingContext) {
    let invalid_values_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("invalid values buffer"),
        contents: bytemuck::cast_slice(&[f32::NAN, f32::INFINITY]),
        usage: BufferUsages::STORAGE,
    });

    //
    // Create a clean `AsBuildContext`
    //

    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    let mut encoder_build = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Build"),
        });

    encoder_build.build_acceleration_structures([&as_ctx.blas_build_entry()], [&as_ctx.tlas]);

    ctx.queue.submit([encoder_build.finish()]);

    //
    // Create shader
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
            entry_point: Some("invalid_usages"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::AccelerationStructure(&as_ctx.tlas),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(invalid_values_buffer.as_entire_buffer_binding()),
            },
        ],
    });

    //
    // Submit once to check for no issues
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

    ctx.queue.submit([encoder_compute.finish()]);
}

#[apply(gpu_test!)]
static SPARSE_INSTANCES_INDEX_CORRECT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .disable_mtl_shader_validation(),
    )
    .run_sync(sparse_instances_index_correct);

fn sparse_instances_index_correct(ctx: TestingContext) {
    // create the shader
    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("shader.wgsl"));
    // Reuse the `all_of_struct` entry point as it outputs what we need
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

    let out_buf = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("shader output buffer"),
        size: STRUCT_SIZE,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // create the blas and tlas
    let tri_sizes = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let sizes = wgpu::BlasGeometrySizeDescriptors::Triangles {
        descriptors: vec![tri_sizes.clone()],
    };

    let blas = ctx.device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_BUILD,
            update_mode: wgpu::wgt::AccelerationStructureUpdateMode::Build,
        },
        sizes.clone(),
    );

    // Create a buffer with a triangle that the shader will hit.
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Blas vertex buf"),
        contents: bytemuck::cast_slice(&[[1.0f32, 0.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]),
        usage: BufferUsages::BLAS_INPUT,
    });

    let mut tlas = ctx.device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        max_instances: 2,
        flags: AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::wgt::AccelerationStructureUpdateMode::Build,
    });

    // Set the second instance, leaving the first blank.
    tlas[1] = Some(wgpu::TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xFF,
    ));

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("base encoder"),
        });

    encoder.build_acceleration_structures(
        [&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &tri_sizes,
                vertex_buffer: &vertex_buffer,
                first_vertex: 0,
                vertex_stride: size_of::<[f32; 3]>() as _,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [&tlas],
    );

    let bind = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: tlas.as_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });


    // Run the shader to get back the instance index

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());

        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let read_back = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("readback buffer"),
        size: size_of::<u32>() as _,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // read back the instance index from the shader.
    encoder.copy_buffer_to_buffer(
        &out_buf,
        12,
        &read_back,
        0,
        size_of::<u32>() as wgpu::BufferAddress,
    );

    ctx.queue.submit([encoder.finish()]);
    read_back.map_async(wgpu::MapMode::Read, .., |res| res.unwrap());
    ctx.device
        .poll(wgpu::wgt::PollType::wait_indefinitely())
        .unwrap();

    let range = read_back.get_mapped_range(..).unwrap();

    let res = u32::from_ne_bytes(*bytemuck::from_bytes(&range));

    drop(range);

    read_back.unmap();

    // instance index should be one as that is the input index.
    assert_eq!(res, 1)
}
