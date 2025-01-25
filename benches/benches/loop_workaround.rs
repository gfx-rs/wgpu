use std::{collections::VecDeque, time::Duration};

use criterion::{criterion_group, Criterion};
use std::sync::LazyLock;
use wgpu::{ComputePassTimestampWrites, ComputePipeline, PipelineCompilationOptions};

use crate::DeviceState;

const ITERATIONS_IN_FLIGHT: usize = 5;
const WORKGROUPS_PER_DISPATCH: u32 = 1024;
const INVOCATIONS_PER_DISPATCH: u32 = 64 * WORKGROUPS_PER_DISPATCH;

struct LoopWorkaroundState {
    device_state: DeviceState,
    pipeline: ComputePipeline,
    bg: wgpu::BindGroup,
    query_sets: Vec<wgpu::QuerySet>,
    resolve_buffers: Vec<wgpu::Buffer>,
    readback_buffers: Vec<wgpu::Buffer>,
}

impl LoopWorkaroundState {
    /// Create and prepare all the resources needed for the renderpass benchmark.
    fn new() -> Self {
        let device_state = DeviceState::new();

        let shader_module = unsafe {
            device_state.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("loop_workaround.wgsl"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(
                        std::fs::read_to_string(concat!(
                            env!("CARGO_MANIFEST_DIR"),
                            "/benches/loop_workaround.wgsl"
                        ))
                        .unwrap(),
                    )),
                },
                wgpu::ShaderRuntimeChecks {
                    bounds_checks: true,
                    force_loop_bounding: true,
                },
            )
        };

        let pipeline =
            device_state
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Loop Workaround Pipeline"),
                    layout: None,
                    module: &shader_module,
                    entry_point: None,
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let buffer = device_state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loop Workaround Buffer"),
            size: INVOCATIONS_PER_DISPATCH as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bg = device_state
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Loop Workaround Bind Group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

        let query_sets = (0..ITERATIONS_IN_FLIGHT)
            .map(|_| {
                device_state
                    .device
                    .create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("Loop Workaround Query Set"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 2,
                    })
            })
            .collect();

        let resolve_buffers = (0..ITERATIONS_IN_FLIGHT)
            .map(|_| {
                device_state.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Loop Workaround Resolve Buffer"),
                    size: 2 * std::mem::size_of::<u64>() as u64,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let readback_buffers = (0..ITERATIONS_IN_FLIGHT)
            .map(|_| {
                device_state.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Loop Workaround Readback Buffer"),
                    size: 2 * std::mem::size_of::<u64>() as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            device_state,
            pipeline,
            bg,
            query_sets,
            resolve_buffers,
            readback_buffers,
        }
    }
}

fn run_bench(ctx: &mut Criterion) {
    let state = LazyLock::new(LoopWorkaroundState::new);

    if !std::env::var("NEXTEST").is_ok() {
        LazyLock::force(&state);
    }

    ctx.bench_function("Loop Workaround", |b| {
        b.iter_custom(|iters| {
            let queue_period = state.device_state.queue.get_timestamp_period() as f64;
            let mut in_flight_submissions = VecDeque::new();

            let mut total_duration_spent = Duration::ZERO;

            for iter in 0..iters {
                let iter_in_flight = iter % ITERATIONS_IN_FLIGHT as u64;

                let query_set = &state.query_sets[iter_in_flight as usize];
                let resolve_buffer = &state.resolve_buffers[iter_in_flight as usize];
                let readback_buffer = &state.readback_buffers[iter_in_flight as usize];

                let mut encoder = state
                    .device_state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });

                cpass.set_pipeline(&state.pipeline);
                cpass.set_bind_group(0, &state.bg, &[]);
                cpass.dispatch_workgroups(WORKGROUPS_PER_DISPATCH, 1, 1);

                drop(cpass);

                encoder.resolve_query_set(&query_set, 0..2, &resolve_buffer, 0);

                encoder.copy_buffer_to_buffer(
                    &resolve_buffer,
                    0,
                    &readback_buffer,
                    0,
                    2 * std::mem::size_of::<u64>() as u64,
                );

                let submission_index = state.device_state.queue.submit(Some(encoder.finish()));
                in_flight_submissions.push_back((iter_in_flight, submission_index));

                readback_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |_| {});

                let last_iteration = iter as u64 == iters - 1;
                let five_iterations_in_flight = in_flight_submissions.len() == ITERATIONS_IN_FLIGHT;

                if five_iterations_in_flight || last_iteration {
                    let iterations_to_purge = if last_iteration {
                        in_flight_submissions.len()
                    } else {
                        1
                    };

                    for _ in 0..iterations_to_purge {
                        let (buffer_idx, submission) = in_flight_submissions.pop_front().unwrap();

                        state
                            .device_state
                            .device
                            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

                        let readback_buffer = &state.readback_buffers[buffer_idx as usize];

                        let query_range = readback_buffer.slice(..).get_mapped_range();
                        let query_data: &[u64] = bytemuck::cast_slice(&*query_range);

                        let diff = query_data[1] - query_data[0];
                        let time = diff as f64 * queue_period;

                        total_duration_spent += Duration::from_secs_f64(time / 1_000_000_000.0);

                        drop(query_range);
                        readback_buffer.unmap();
                    }
                }
            }

            println!(
                "{:?}: {} {:?} per",
                total_duration_spent,
                iters,
                total_duration_spent / iters as u32
            );

            total_duration_spent
        });
    });
}

criterion_group! {
    name = loop_workaround;
    config = Criterion::default().measurement_time(Duration::from_secs(20)).sample_size(10);
    targets = run_bench,
}
