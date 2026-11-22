use std::sync::mpsc;

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BufferBinding, BufferUsages, Features, InstanceFlags,
    Limits, RayTracingPassDescriptor, RayTracingPipeline, RayTracingPipelineDescriptor,
    RayTracingStage, ShaderModuleDescriptor,
};
use wgpu_test::{
    apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(TRACE_RAYS_INDIRECT);
    tests.push(TRACE_RAYS_INDIRECT_DISCARD);
    tests.push(TRACE_RAYS_INDIRECT_NO_VALIDATION);
    tests.push(TRACE_RAYS_INDIRECT_STATE_RESTORE);
    tests.push(TRACE_RAYS_INDIRECT_INVALID);
}

fn ray_tracing_limits() -> Limits {
    Limits::defaults()
        .using_minimum_supported_acceleration_structure_values()
        .using_minimum_supported_ray_tracing_pipeline_values()
}

/// Creates a pipeline whose ray generation shader counts its invocations.
fn counting_pipeline(ctx: &TestingContext) -> RayTracingPipeline {
    let ray_gen = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray generation shader"),
        source: wgpu::ShaderSource::Wgsl(
            "
            enable wgpu_ray_tracing_pipeline;

            var<ray_payload> payload: u32;

            @group(0) @binding(0)
            var<storage, read_write> count: atomic<u32>;

            @ray_generation
            fn gen() {
                atomicAdd(&count, 1u);
            }
            "
            .into(),
        ),
    });

    let ray_miss = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray miss shader"),
        source: wgpu::ShaderSource::Wgsl(
            "
            enable wgpu_ray_tracing_pipeline;

            var<incoming_ray_payload> payload: u32;

            @miss
            @incoming_payload(payload)
            fn miss() {}
            "
            .into(),
        ),
    });

    ctx.device
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
            intersections: &[],
            max_recursion_depth: 1,
            cache: None,
        })
}

/// Runs one indirect trace rays call per entry of `args` in a single pass, each counting into
/// its own counter, followed by a direct `trace_rays(1, 1, 1)` that reuses the last bind group.
///
/// Returns the counters.
fn run_indirect(ctx: &TestingContext, args: &[[u32; 3]]) -> Vec<u32> {
    let pipeline = counting_pipeline(ctx);

    // Offsets that are multiples of 4 but not of 12 or 256, some of which are past 256.
    let offset_of = |i: usize| 4 + 100 * i as u64;
    let mut indirect_data = vec![0u8; offset_of(args.len()) as usize];
    for (i, arg) in args.iter().enumerate() {
        let offset = offset_of(i) as usize;
        indirect_data[offset..offset + 12].copy_from_slice(bytemuck::cast_slice(arg));
    }
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    let indirect_buffer = write_indirect_buffer(ctx, &mut encoder, &indirect_data);

    let counter_stride = ctx.device.limits().min_storage_buffer_offset_alignment as u64;
    let counter_size = counter_stride * args.len() as u64;
    let counters = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("counters"),
        size: counter_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: counter_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_groups: Vec<_> = (0..args.len())
        .map(|i| {
            ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(BufferBinding {
                        buffer: &counters,
                        offset: counter_stride * i as u64,
                        size: wgpu::BufferSize::new(4),
                    }),
                }],
            })
        })
        .collect();

    {
        let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor {
            label: Some("indirect pass"),
        });
        pass.push_debug_group("indirect trace rays");
        pass.set_pipeline(&pipeline);
        for (i, bind_group) in bind_groups.iter().enumerate() {
            pass.set_bind_group(0, bind_group, &[]);
            pass.trace_rays_indirect(&indirect_buffer, offset_of(i));
        }
        pass.trace_rays(1, 1, 1);
        pass.pop_debug_group();
    }
    encoder.copy_buffer_to_buffer(&counters, 0, &readback, 0, counter_size);
    ctx.queue.submit([encoder.finish()]);

    read_counters(ctx, &readback, counter_stride, args.len())
}

/// Creates an indirect buffer whose contents are copied from `data` on the GPU by `encoder`, so
/// that the indirect read depends on a barrier after a GPU write.
fn write_indirect_buffer(
    ctx: &TestingContext,
    encoder: &mut wgpu::CommandEncoder,
    data: &[u8],
) -> wgpu::Buffer {
    let staging = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("indirect staging buffer"),
        contents: data,
        usage: BufferUsages::COPY_SRC,
    });
    let indirect_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indirect buffer"),
        size: staging.size(),
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&staging, 0, &indirect_buffer, 0, staging.size());
    indirect_buffer
}

fn read_counters(
    ctx: &TestingContext,
    readback: &wgpu::Buffer,
    counter_stride: u64,
    count: usize,
) -> Vec<u32> {
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
    (0..count)
        .map(|i| {
            let start = counter_stride as usize * i;
            u32::from_le_bytes(range[start..start + 4].try_into().unwrap())
        })
        .collect()
}

#[apply(gpu_test!)]
static TRACE_RAYS_INDIRECT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .instance_flags(InstanceFlags::VALIDATION_INDIRECT_CALL)
            .limits(ray_tracing_limits()),
    )
    .run_sync(|ctx| {
        let counts = run_indirect(
            &ctx,
            &[
                [2, 3, 1],
                [0, 5, 5],
                [1, 1, 1],
                // Within the per-dimension limits, but the product overflows `u32`.
                [65536, 65536, 1],
                [65536, 65536, 16],
                [4, 4, 4],
            ],
        );
        // The last counter also includes the direct `trace_rays(1, 1, 1)`.
        assert_eq!(counts, [6, 0, 1, 0, 0, 65]);
    });

/// Make sure that indirect trace rays calls above the device limits don't run.
#[apply(gpu_test!)]
static TRACE_RAYS_INDIRECT_DISCARD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .instance_flags(InstanceFlags::VALIDATION_INDIRECT_CALL)
            .limits(Limits {
                // Limits each dimension to 8 rays.
                max_compute_workgroup_size_x: 4,
                max_compute_workgroup_size_y: 4,
                max_compute_workgroup_size_z: 4,
                max_compute_invocations_per_workgroup: 64,
                max_compute_workgroups_per_dimension: 2,
                max_ray_dispatch_count: 100,
                ..ray_tracing_limits()
            }),
    )
    .run_sync(|ctx| {
        let counts = run_indirect(
            &ctx,
            &[
                [8, 8, 1],
                [9, 1, 1],
                [1, 9, 1],
                [1, 1, 9],
                [5, 5, 4],
                [8, 8, 2],
                [8, 8, 8],
                [1, 1, 1],
            ],
        );
        // The last counter also includes the direct `trace_rays(1, 1, 1)`.
        assert_eq!(counts, [64, 0, 0, 0, 100, 0, 0, 2]);
    });

/// Indirect trace rays without indirect validation reads the user's buffer directly.
#[apply(gpu_test!)]
static TRACE_RAYS_INDIRECT_NO_VALIDATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .remove_instance_flags(InstanceFlags::VALIDATION_INDIRECT_CALL)
            .limits(ray_tracing_limits()),
    )
    .run_sync(|ctx| {
        let counts = run_indirect(&ctx, &[[2, 3, 1], [3, 1, 2]]);
        assert_eq!(counts, [6, 7]);
    });

/// Make sure the pipeline, bind groups with dynamic offsets and immediates survive the indirect
/// validation between consecutive indirect calls.
#[apply(gpu_test!)]
static TRACE_RAYS_INDIRECT_STATE_RESTORE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES | Features::IMMEDIATES)
            .instance_flags(InstanceFlags::VALIDATION_INDIRECT_CALL)
            .limits(Limits {
                max_immediate_size: 4,
                ..ray_tracing_limits()
            }),
    )
    .run_sync(|ctx| {
        let ray_gen = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ray generation shader"),
            source: wgpu::ShaderSource::Wgsl(
                "
                enable wgpu_ray_tracing_pipeline;

                var<ray_payload> payload: u32;
                var<immediate> amount: u32;

                @group(0) @binding(0)
                var<storage, read_write> count: atomic<u32>;

                @ray_generation
                fn gen() {
                    atomicAdd(&count, amount);
                }
                "
                .into(),
            ),
        });
        let ray_miss = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ray miss shader"),
            source: wgpu::ShaderSource::Wgsl(
                "
                enable wgpu_ray_tracing_pipeline;

                var<incoming_ray_payload> payload: u32;

                @miss
                @incoming_payload(payload)
                fn miss() {}
                "
                .into(),
            ),
        });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::RAY_GENERATION,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(4),
                        },
                        count: None,
                    }],
                });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 4,
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
                intersections: &[],
                max_recursion_depth: 1,
                cache: None,
            });

        let counter_stride = ctx.device.limits().min_storage_buffer_offset_alignment as u64;
        let counters = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counters"),
            size: counter_stride * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: counter_stride * 2,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(BufferBinding {
                    buffer: &counters,
                    offset: 0,
                    size: wgpu::BufferSize::new(4),
                }),
            }],
        });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        let indirect_buffer =
            write_indirect_buffer(&ctx, &mut encoder, bytemuck::cast_slice(&[1u32, 1, 1]));
        {
            let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_immediates(0, &3u32.to_le_bytes());
            pass.set_bind_group(0, &bind_group, &[0]);
            pass.trace_rays_indirect(&indirect_buffer, 0);
            pass.trace_rays_indirect(&indirect_buffer, 0);
            pass.set_bind_group(0, &bind_group, &[counter_stride as u32]);
            pass.trace_rays_indirect(&indirect_buffer, 0);
            pass.trace_rays_indirect(&indirect_buffer, 0);
            pass.trace_rays(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&counters, 0, &readback, 0, counter_stride * 2);
        ctx.queue.submit([encoder.finish()]);

        assert_eq!(read_counters(&ctx, &readback, counter_stride, 2), [6, 9]);
    });

#[apply(gpu_test!)]
static TRACE_RAYS_INDIRECT_INVALID: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)
            .limits(ray_tracing_limits()),
    )
    .run_sync(|ctx| {
        let pipeline = counting_pipeline(&ctx);
        let counter = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counter"),
            size: 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: counter.as_entire_binding(),
            }],
        });

        let trace = |buffer: &wgpu::Buffer, offset: u64| {
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.trace_rays_indirect(buffer, offset);
            }
            encoder.finish()
        };

        let indirect_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect buffer"),
            size: 16,
            usage: BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let storage_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("storage buffer"),
            size: 16,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Sanity check that the valid call succeeds.
        ctx.queue.submit([trace(&indirect_buffer, 4)]);

        fail(
            &ctx.device,
            || trace(&storage_buffer, 0),
            Some("do not contain required usage flags BufferUsages { buffer_usages_webgpu: BufferUsagesWebGPU(INDIRECT)"),
        );
        fail(
            &ctx.device,
            || trace(&indirect_buffer, 2),
            Some("Indirect buffer offset 2 is not a multiple of 4"),
        );
        fail(
            &ctx.device,
            || trace(&indirect_buffer, 8),
            Some("Indirect buffer uses bytes 8..20 which overruns indirect buffer of size 16"),
        );
        fail(
            &ctx.device,
            || trace(&indirect_buffer, u64::MAX - 3),
            Some("overruns indirect buffer of size 16"),
        );

        // The indirect buffer must not also be written by the same trace.
        let storage_indirect_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("storage indirect buffer"),
            size: 16,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let conflicting_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: storage_indirect_buffer.as_entire_binding(),
            }],
        });
        fail(
            &ctx.device,
            || {
                let mut encoder = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass =
                        encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &conflicting_bind_group, &[]);
                    pass.trace_rays_indirect(&storage_indirect_buffer, 0);
                }
                encoder.finish()
            },
            Some("Attempted to use Buffer with 'storage indirect buffer' label with"),
        );
    });
