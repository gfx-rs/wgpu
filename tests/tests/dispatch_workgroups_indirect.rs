use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

/// Make sure that the num_workgroups builtin works properly (it requires a workaround on D3D12).
#[gpu_test]
static NUM_WORKGROUPS_BUILTIN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::PUSH_CONSTANTS)
            .downlevel_flags(
                wgpu::DownlevelFlags::COMPUTE_SHADERS | wgpu::DownlevelFlags::INDIRECT_EXECUTION,
            )
            .limits(wgpu::Limits {
                max_push_constant_size: 4,
                ..wgpu::Limits::downlevel_defaults()
            }),
    )
    .run_async(|ctx| async move {
        let num_workgroups = [1, 2, 3];
        let res = run_test(&ctx, &num_workgroups).await;
        assert_eq!(res, num_workgroups);
    });

/// Make sure that we discard (don't run) the dispatch if its size exceeds the device limit.
#[gpu_test]
static DISCARD_DISPATCH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::PUSH_CONSTANTS)
            .downlevel_flags(
                wgpu::DownlevelFlags::COMPUTE_SHADERS | wgpu::DownlevelFlags::INDIRECT_EXECUTION,
            )
            .limits(wgpu::Limits {
                max_compute_workgroups_per_dimension: 10,
                max_push_constant_size: 4,
                ..wgpu::Limits::downlevel_defaults()
            }),
    )
    .run_async(|ctx| async move {
        let max = ctx.device.limits().max_compute_workgroups_per_dimension;

        let res = run_test(&ctx, &[max, max, max]).await;
        assert_eq!(res, [max; 3]);

        let res = run_test(&ctx, &[max + 1, 1, 1]).await;
        assert_eq!(res, [0; 3]);

        let res = run_test(&ctx, &[1, max + 1, 1]).await;
        assert_eq!(res, [0; 3]);

        let res = run_test(&ctx, &[1, 1, max + 1]).await;
        assert_eq!(res, [0; 3]);
    });

/// Make sure that resetting the bind groups set by the validation code works properly.
#[gpu_test]
static RESET_BIND_GROUPS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::PUSH_CONSTANTS)
            .downlevel_flags(
                wgpu::DownlevelFlags::COMPUTE_SHADERS | wgpu::DownlevelFlags::INDIRECT_EXECUTION,
            )
            .limits(wgpu::Limits {
                max_push_constant_size: 4,
                ..wgpu::Limits::downlevel_defaults()
            }),
    )
    .run_async(|ctx| async move {
        ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let test_resources = TestResources::new(&ctx);

        let indirect_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&test_resources.pipeline);
            compute_pass.set_push_constants(0, &[0, 0, 0, 0]);
            // compute_pass.set_bind_group(0, Some(&test_resources.bind_group), &[]);
            compute_pass.dispatch_workgroups_indirect(&indirect_buffer, 0);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let error = pollster::block_on(ctx.device.pop_error_scope());
        assert!(error.map_or(false, |error| {
            format!("{error}").contains("The current set ComputePipeline with '' label expects a BindGroup to be set at index 0")
        }));
    });

/// Make sure that zero sized buffer validation is raised.
#[gpu_test]
static ZERO_SIZED_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::PUSH_CONSTANTS)
            .downlevel_flags(
                wgpu::DownlevelFlags::COMPUTE_SHADERS | wgpu::DownlevelFlags::INDIRECT_EXECUTION,
            )
            .limits(wgpu::Limits {
                max_push_constant_size: 4,
                ..wgpu::Limits::downlevel_defaults()
            }),
    )
    .run_async(|ctx| async move {
        ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let test_resources = TestResources::new(&ctx);

        let indirect_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 0,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&test_resources.pipeline);
            compute_pass.set_push_constants(0, &[0, 0, 0, 0]);
            compute_pass.set_bind_group(0, Some(&test_resources.bind_group), &[]);
            compute_pass.dispatch_workgroups_indirect(&indirect_buffer, 0);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let error = pollster::block_on(ctx.device.pop_error_scope());
        assert!(error.map_or(false, |error| {
            format!("{error}").contains(
                "Indirect buffer uses bytes 0..12 which overruns indirect buffer of size 0",
            )
        }));
    });

struct TestResources {
    pipeline: wgpu::ComputePipeline,
    out_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl TestResources {
    fn new(ctx: &TestingContext) -> Self {
        const SHADER_SRC: &str = "
            struct TestOffsetPc {
                inner: u32,
            }

            // `test_offset.inner` should always be 0; we test that resetting the push constant set by the validation code works properly.
            var<push_constant> test_offset: TestOffsetPc;

            @group(0) @binding(0)
            var<storage, read_write> out: array<u32, 3>;

            @compute @workgroup_size(1)
            fn main(@builtin(num_workgroups) num_workgroups: vec3u, @builtin(workgroup_id) workgroup_id: vec3u) {
                if (all(workgroup_id == vec3u())) {
                    out[0] = num_workgroups.x + test_offset.inner;
                    out[1] = num_workgroups.y + test_offset.inner;
                    out[2] = num_workgroups.z + test_offset.inner;
                }
            }
        ";

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgt::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[wgt::PushConstantRange {
                    stages: wgt::ShaderStages::COMPUTE,
                    range: 0..4,
                }],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            out_buffer,
            readback_buffer,
            bind_group,
        }
    }
}

async fn run_test(ctx: &TestingContext, num_workgroups: &[u32; 3]) -> [u32; 3] {
    let test_resources = TestResources::new(ctx);

    let mut res = None;

    for (indirect_offset, indirect_buffer_size) in [
        // internal src buffer binding size will be buffer.size
        (0, 12),
        (4, 4 + 12),
        (4, 8 + 12),
        (256 * 2 - 4 - 12, 256 * 2 - 4),
        // internal src buffer binding size will be 256 * 2 + x
        (0, 256 * 2 * 2 + 4),
        (256, 256 * 2 * 2 + 8),
        (256 + 4, 256 * 2 * 2 + 12),
        (256 * 2 + 16, 256 * 2 * 2 + 16),
        (256 * 2 * 2, 256 * 2 * 2 + 32),
        (256 + 12, 256 * 2 * 2 + 64),
    ] {
        let indirect_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: indirect_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        ctx.queue.write_buffer(
            &indirect_buffer,
            indirect_offset,
            bytemuck::bytes_of(num_workgroups),
        );

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&test_resources.pipeline);
            compute_pass.set_push_constants(0, &[0, 0, 0, 0]);
            compute_pass.set_bind_group(0, Some(&test_resources.bind_group), &[]);
            compute_pass.dispatch_workgroups_indirect(&indirect_buffer, indirect_offset);
        }

        encoder.copy_buffer_to_buffer(
            &test_resources.out_buffer,
            0,
            &test_resources.readback_buffer,
            0,
            12,
        );

        ctx.queue.submit(Some(encoder.finish()));

        test_resources
            .readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});

        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();

        let view = test_resources.readback_buffer.slice(..).get_mapped_range();

        let current_res = *bytemuck::from_bytes(&view);
        drop(view);
        test_resources.readback_buffer.unmap();

        if let Some(past_res) = res {
            assert_eq!(past_res, current_res);
        } else {
            res = Some(current_res);
        }
    }

    res.unwrap()
}
