use std::time::{Duration, Instant};

use criterion::{criterion_group, Criterion, Throughput};
use once_cell::sync::Lazy;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::DeviceState;

fn run_bench(ctx: &mut Criterion) {
    let state = Lazy::new(DeviceState::new);

    const RESOURCES_TO_CREATE: usize = 8;

    let mut group = ctx.benchmark_group("Resource Creation: Large Buffer");
    group.throughput(Throughput::Elements(RESOURCES_TO_CREATE as _));

    for threads in [1, 2, 4, 8] {
        let resources_per_thread = RESOURCES_TO_CREATE / threads;
        group.bench_function(
            format!("{threads} threads x {resources_per_thread} resource"),
            |b| {
                Lazy::force(&state);

                b.iter_custom(|iters| {
                    profiling::scope!("benchmark invocation");

                    let mut duration = Duration::ZERO;

                    for _ in 0..iters {
                        profiling::scope!("benchmark iteration");

                        // We can't create too many resources at once, so we do it 8 resources at a time.
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

                        duration += start.elapsed();

                        drop(buffers);

                        state.queue.submit([]);
                        state.device.poll(wgpu::Maintain::Wait);
                    }

                    duration
                })
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = resource_creation;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = run_bench,
}
