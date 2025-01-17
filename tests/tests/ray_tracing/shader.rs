use crate::ray_tracing::AsBuildContext;
use wgpu::{
    include_wgsl, BindGroupDescriptor, BindGroupEntry, BindingResource, BufferDescriptor,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
};
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, TestParameters, TestingContext};
use wgt::BufferUsages;

const STRUCT_SIZE: wgt::BufferAddress = 176;

#[gpu_test]
static ACCESS_ALL_STRUCT_MEMBERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits().features(
        wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
            | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
    ))
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

    let as_ctx = AsBuildContext::new(&ctx);

    let mut encoder_build = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Build"),
        });

    encoder_build
        .build_acceleration_structures([&as_ctx.blas_build_entry()], [&as_ctx.tlas_package]);

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
                resource: BindingResource::AccelerationStructure(as_ctx.tlas_package.tlas()),
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
