use std::time::Instant;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use wgpu_benchmark::{iter, BenchmarkContext, SubBenchResult};

use crate::DeviceState;

fn thread_count_list(ctx: &BenchmarkContext) -> &'static [usize] {
    if ctx.is_test() {
        &[2]
    } else {
        &[1, 2, 4, 8]
    }
}

pub fn run_bench(ctx: BenchmarkContext) -> anyhow::Result<Vec<SubBenchResult>> {
    let state = DeviceState::new();

    const RESOURCES_TO_CREATE: usize = 8;

    let mut results = Vec::new();
    for &threads in thread_count_list(&ctx) {
        let resources_per_thread = RESOURCES_TO_CREATE / threads;

        results.push(iter(
            &ctx,
            &format!("{threads} threads"),
            "buffers",
            RESOURCES_TO_CREATE as u32,
            || {
                let start = Instant::now();
                let buffers = (0..threads)
                    .into_par_iter()
                    .map(|_| {
                        (0..resources_per_thread)
                            .map(|_| {
                                state.device.create_buffer(&wgpu::BufferDescriptor {
                                    label: None,
                                    size: 256 * 1024 * 1024,
                                    usage: wgpu::BufferUsages::COPY_DST,
                                    mapped_at_creation: false,
                                })
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let duration = start.elapsed();

                drop(buffers);

                state.queue.submit([]);
                state
                    .device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();

                duration
            },
        ));
    }
    Ok(results)
}
