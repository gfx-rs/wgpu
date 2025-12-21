use std::time::Duration;

use crate::{BenchmarkContext, LoopControl, SubBenchResult};

pub fn iter(
    ctx: &BenchmarkContext,
    name: &str,
    throughput_unit: &str,
    throughput_count_per_iteration: u32,
    mut f: impl FnMut() -> Duration,
) -> SubBenchResult {
    profiling::scope!("iter", name);

    let mut iterations = 0_u32;
    let mut duration = Duration::ZERO;

    let control = if let Some(override_control) = ctx.override_iters {
        override_control
    } else {
        ctx.default_iterations
    };

    while !control.finished(iterations, duration) {
        duration += f();
        iterations += 1;
    }

    SubBenchResult {
        name: name.to_string(),
        avg_duration_per_iteration: duration / iterations,
        iterations,
        throughput_unit: throughput_unit.to_string(),
        throughput_count_per_iteration,
    }
}

pub fn iter_auto(
    ctx: &BenchmarkContext,
    name: &str,
    throughput_unit: &str,
    throughput_count_per_iteration: u32,
    mut f: impl FnMut(),
) -> SubBenchResult {
    iter(
        ctx,
        name,
        throughput_unit,
        throughput_count_per_iteration,
        || {
            let start = std::time::Instant::now();
            f();
            start.elapsed()
        },
    )
}

pub fn iter_many(
    ctx: &BenchmarkContext,
    names: Vec<String>,
    throughput_unit: &str,
    throughput_count_per_iteration: u32,
    mut f: impl FnMut() -> Vec<Duration>,
) -> Vec<SubBenchResult> {
    profiling::scope!("iter", &*names[0]);

    let mut iterations = 0_u32;
    let mut durations = vec![Duration::ZERO; names.len()];

    let control = if let Some(override_control) = ctx.override_iters {
        override_control
    } else {
        LoopControl::Time(Duration::from_secs(1))
    };

    // We use the first duration to determine whether to stop. This means the other sub-benchmarks
    // could have run for longer or shorter than intended, but that's acceptable.
    while !control.finished(iterations, *durations.first().unwrap_or(&Duration::ZERO)) {
        let iteration_durations = f();
        assert_eq!(iteration_durations.len(), names.len());
        for (i, dur) in iteration_durations.into_iter().enumerate() {
            durations[i] += dur;
        }
        iterations += 1;
    }

    durations
        .into_iter()
        .enumerate()
        .map(|(i, d)| SubBenchResult {
            name: names[i].to_string(),
            avg_duration_per_iteration: d / iterations,
            iterations,
            throughput_unit: throughput_unit.to_string(),
            throughput_count_per_iteration,
        })
        .collect()
}
