use wgpu::{
    BindGroupDescriptor, BindGroupEntry, CommandEncoderDescriptor, Features, Limits,
    RayTracingIntersectionDescriptor, RayTracingPassDescriptor, RayTracingPipelineDescriptor,
    RayTracingStage, ShaderModuleDescriptor,
};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};
use wgpu_types::AccelerationStructureFlags;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(PIPELINE_CREATE_USE);
}

#[apply(gpu_test!)]
static PIPELINE_CREATE_USE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .limits(Limits::defaults().using_minimum_supported_acceleration_structure_values()),
    )
    .run_sync(pipeline_create_use);

fn pipeline_create_use(ctx: TestingContext) {
    let as_ctx = super::AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
        true,
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());

    // Build the BLAS to be compacted (so compaction is valid).
    encoder.build_acceleration_structures([&as_ctx.blas_build_entry()], [&as_ctx.tlas]);

    ctx.queue.submit([encoder.finish()]);

    let ray_gen_source = "
        enable wgpu_ray_tracing_pipeline;

        @group(0) @binding(0) var acc_struct: acceleration_structure;

        var<ray_payload> payload: u32;

        @ray_generation
        fn gen() {
            traceRay(acc_struct, RayDesc(), &payload);
        }
    ";

    let ray_closest_source = "
        enable wgpu_ray_tracing_pipeline;

        var<incoming_ray_payload> payload: u32;

        @closest_hit
        @incoming_payload(payload)
        fn closest() {
            
        }
    ";

    let ray_any_source = "
        enable wgpu_ray_tracing_pipeline;

        var<incoming_ray_payload> payload: u32;

        @any_hit
        @incoming_payload(payload)
        fn any() {
            
        }
    ";

    let ray_miss_source = "
        enable wgpu_ray_tracing_pipeline;

        var<incoming_ray_payload> payload: u32;

        @miss
        @incoming_payload(payload)
        fn miss() {
            
        }
    ";

    let ray_gen = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray generation shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ray_gen_source)),
    });

    let ray_closest = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray closest hit shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ray_closest_source)),
    });

    let ray_any = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray any hit shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ray_any_source)),
    });

    let ray_miss = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray miss shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(ray_miss_source)),
    });

    let pipeline = ctx
        .device
        .create_ray_tracing_pipeline(&RayTracingPipelineDescriptor {
            label: None,
            layout: None,
            ray_generation: RayTracingStage {
                module: &ray_gen,
                entry_point: None,
                compilation_options: Default::default(),
            },
            miss: RayTracingStage {
                module: &ray_miss,
                entry_point: None,
                compilation_options: Default::default(),
            },
            intersection_descs: &[RayTracingIntersectionDescriptor::Triangle {
                closest_hit: RayTracingStage {
                    module: &ray_closest,
                    entry_point: None,
                    compilation_options: Default::default(),
                },
                any_hit: Some(RayTracingStage {
                    module: &ray_any,
                    entry_point: None,
                    compilation_options: Default::default(),
                }),
            }],
            max_recersion_depth: 1,
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("ray tracing pipeline bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: as_ctx.tlas.as_binding(),
        }],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.trace_rays(1, 1, 2);
    }

    ctx.queue.submit([encoder.finish()]);
}
