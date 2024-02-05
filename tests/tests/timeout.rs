use std::{num::NonZeroU64, time::Duration};

use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static POLL_TIMEOUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            // At best, this test is meaningless on the GL backend, as there are no consequences for
            // early deleting resources before the timeout is hit.
            //
            // At worst, on lavapipe the test will run forever, as GL implementations are allowed to
            // flush whenever they want, meaning that there is never any asynchronicity in the first place.
            .skip(FailureCase::backend(wgpu::Backends::GL)),
    )
    .run_sync(poll_timeout);

const WORKGROUP_SIZE: u32 = 256;
const DISPATCHES: u32 = (1 << 16) - 1;
const BUFFER_SIZE: u64 = (DISPATCHES as u64) * (WORKGROUP_SIZE as u64) * 4;
const TIMEOUT: Duration = Duration::from_millis(250);

const SHADER: &str = r#"
    @group(0) @binding(0) var<storage, read_write> buffer_a: array<f32>;
    @group(0) @binding(1) var<storage, read> buffer_b: array<f32>;

    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) index: vec3<u32>) {
        // arctan famously blows up to a ton of alu code.
        buffer_a[index.x] = atan(atan(atan(buffer_b[index.x])));
    }
"#;

/// The given work is fairly memory and alu intensive, so higher iterations should increase in time quickly.
struct HeavyWork {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl HeavyWork {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cs_module"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: "main",
        });

        Self {
            bind_group_layout,
            pipeline,
        }
    }

    fn perform_work(&self, ctx: &TestingContext, iterations: u32) {
        // We intentionally create new buffers every time so that the destruction
        // of each buffer/texture is dependant on the given submission being finished.
        //
        // We do not keep our handles to the resources open, so they can be destroyed as soon as
        // the submission is finished.
        let buffer_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("buffer_a iterations {iterations}")),
            size: BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buffer_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("buffer_b iterations {iterations}")),
            size: BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group_a = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bind_group_a iterations {iterations}")),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
            ],
        });

        let bind_group_b = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bind_group_b iterations {iterations}")),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_a.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("encoder iterations {iterations}")),
            });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("cpass iterations {iterations}")),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);

        for _ in 0..iterations {
            cpass.set_bind_group(0, &bind_group_a, &[]);
            cpass.dispatch_workgroups(DISPATCHES, 1, 1);

            cpass.set_bind_group(0, &bind_group_b, &[]);
            cpass.dispatch_workgroups(DISPATCHES, 1, 1);
        }

        drop(cpass);

        ctx.queue.submit([encoder.finish()]);
    }
}

fn poll_timeout(ctx: TestingContext) {
    let work = HeavyWork::new(&ctx.device);

    // We keep doubling the amount of iterations until we hit a timeout.
    //
    // We then keep increasing at least two more times, to make sure that the handling
    // of the timeout works correctly, and does not cause early deletes, even after multiple
    // potential timeouts.
    let mut iterations = 1;
    let mut timeouts = 0;
    loop {
        eprintln!("{iterations}: starting");
        work.perform_work(&ctx, iterations);
        ctx.queue.on_submitted_work_done(move || {
            eprintln!("{iterations}: done");
        });
        let maintain_result = ctx.device.poll(wgpu::PollInfo {
            submission_index: None,
            wait_duration: wgpu::WaitDuration::Duration(TIMEOUT),
        });
        eprintln!("{iterations}: maintain result {maintain_result:?}");
        if maintain_result.is_incomplete() {
            timeouts += 1;
            if timeouts == 3 {
                eprintln!("three timeouts detected!");
                break;
            }
        }
        iterations *= 2;
    }
}
