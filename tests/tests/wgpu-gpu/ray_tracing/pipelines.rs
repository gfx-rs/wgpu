use std::sync::mpsc;

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, CommandEncoderDescriptor, Features, Limits, PipelineLayoutDescriptor, RayTracingIntersectionDescriptor, RayTracingPassDescriptor, RayTracingPipelineDescriptor, RayTracingStage, ShaderModuleDescriptor, ShaderStages
};
use wgpu_test::{
    apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};
use wgpu_types::AccelerationStructureFlags;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(PIPELINE_CREATE_USE);
    tests.push(RAY_TRACING_PASS_NO_FEATURE);
    tests.push(PIPELINE_OUTPUT);
    tests.push(PIPELINE_SWAP);
}

#[apply(gpu_test!)]
static PIPELINE_CREATE_USE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .limits(
                Limits::defaults()
                    .using_minimum_supported_acceleration_structure_values()
                    .using_minimum_supported_ray_tracing_pipeline_values(),
            ),
    )
    .run_sync(pipeline_create_use);

fn pipeline_create_use(ctx: TestingContext) {
    let mut as_ctx = super::AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());

    // Build the BLAS and the TLAS.
    encoder.build_acceleration_structures([&as_ctx.blas_build_entry()], [&as_ctx.tlas]);

    ctx.queue.submit([encoder.finish()]);

    let ray_gen_source = "
        enable wgpu_ray_tracing_pipeline;

        @group(0) @binding(0) var acc_struct: acceleration_structure;

        var<ray_payload> payload: u32;

        @ray_generation
        fn gen() {
            traceRay(acc_struct, RayDesc(0u, 0xFFu, 0.001, 100.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)), &payload);
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
            max_recursion_depth: 1,
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

    // Change the intersection index to the other valid one
    as_ctx.tlas[0].as_mut().unwrap().intersection_index = 1;

    // Build the TLAS with the other index.
    encoder.build_acceleration_structures([], [&as_ctx.tlas]);

    // Rerun with the new intersection index.
    {
        let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.trace_rays(1, 1, 2);
    }

    ctx.queue.submit([encoder.finish()]);

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    // Change the intersection index to be invalid.
    as_ctx.tlas[0].as_mut().unwrap().intersection_index = 2;

    // Build the TLAS with the invalid index.
    encoder.build_acceleration_structures([], [&as_ctx.tlas]);

    // Rerun with the new intersection index (should fail).
    {
        let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.trace_rays(1, 1, 2);
    }

    fail(
        &ctx.device,
        || {
            ctx.queue.submit([encoder.finish()]);
        },
        None,
    );
}

#[apply(gpu_test!)]
static RAY_TRACING_PASS_NO_FEATURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(ray_tracing_pass_no_feature);

fn ray_tracing_pass_no_feature(ctx: TestingContext) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());

    let pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
    drop(pass);

    fail(&ctx.device, || encoder.finish(), None);
}

#[apply(gpu_test!)]
static PIPELINE_OUTPUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .limits(
                Limits::defaults()
                    .using_minimum_supported_acceleration_structure_values()
                    .using_minimum_supported_ray_tracing_pipeline_values(),
            ),
    )
    .run_sync(pipeline_output);

fn pipeline_output(ctx: TestingContext) {
    let ray_gen_source = "
        enable wgpu_ray_tracing_pipeline;

        var<ray_payload> payload: u32;

        @group(0) @binding(0)
        var<storage, read_write> out: u32;

        @ray_generation
        fn gen() {
            out = 1;
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
            intersection_descs: &[],
            max_recursion_depth: 1,
            cache: None,
        });

    let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_of::<u32>() as _,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_of::<u32>() as _,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("ray tracing pipeline bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: out_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.trace_rays(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&out_buffer, 0, &readback, 0, out_buffer.size());

    ctx.queue.submit([encoder.finish()]);

    let (send, recv) = mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |res| {
        res.unwrap();
        send.send(()).unwrap()
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    recv.recv().unwrap();

    let range = readback.get_mapped_range(..).unwrap();

    assert_eq!(range[0], 1)
}

#[apply(gpu_test!)]
static PIPELINE_SWAP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .limits(
                Limits::defaults()
                    .using_minimum_supported_acceleration_structure_values()
                    .using_minimum_supported_ray_tracing_pipeline_values(),
            ),
    )
    .run_sync(pipeline_swap);

fn pipeline_swap(ctx: TestingContext) {
    let as_ctx = super::AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());

    // Build the BLAS and the TLAS.
    encoder.build_acceleration_structures([&as_ctx.blas_build_entry()], [&as_ctx.tlas]);

    ctx.queue.submit([encoder.finish()]);

    let ray_gen_source = "
        enable wgpu_ray_tracing_pipeline;

        @group(0) @binding(0) var acc_struct: acceleration_structure;

        var<ray_payload> payload: u32;

        @ray_generation
        fn gen() {
            traceRay(acc_struct, RayDesc(0u, 0xFFu, 0.001, 100.0, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0)), &payload);
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

    let bgl = ctx.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::RAY_GENERATION,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            Some(&bgl),
        ],
        immediate_size: 0,
    });

    let pipeline = ctx
        .device
        .create_ray_tracing_pipeline(&RayTracingPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
            max_recursion_depth: 1,
            cache: None,
        });

    let pipeline_to_swap = ctx
        .device
        .create_ray_tracing_pipeline(&RayTracingPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
            intersection_descs: &[],
            max_recursion_depth: 1,
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("ray tracing pipeline bind group"),
        layout: &bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: as_ctx.tlas.as_binding(),
        }],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_pipeline(&pipeline_to_swap);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.trace_rays(1, 1, 2);
    }

    fail(
        &ctx.device,
        || {
            ctx.queue.submit([encoder.finish()]);
        },
        None,
    );
}
