#![cfg(not(target_arch = "wasm32"))]
#![expect(
    clippy::disallowed_types,
    reason = "We're outside of the main wgpu codebase"
)]

//! Benchmarking framework for `wgpu`.
//!
//! This crate is a basic framework for benchmarking. Its design is guided
//! by a few goals:
//!
//! - Enumerating tests should be extremely cheap. `criterion` needs
//!   to run all of your benchmark functions to enumerate them during
//!   testing. This requires your code to contort itself to avoid doing
//!   any work until you enter a benchmark callback. This framework
//!   avoids that by having an explicit list of benchmark function.
//! - It must be compatible with `cargo-nextest` and have a compatible
//!   "test" mode that runs each benchmark exactly once.
//! - It should be able to have intuitive test grouping, allowing for
//!   allowing for quick execution of a reasonable baseline set of benchmarks
//!   during development, while still allowing for a more exhaustive
//!   benchmark suite to be run if desired.
//!
//! By default all tests run for 2 seconds, but this can be overridden
//! by individual tests.

use std::{collections::HashMap, io::IsTerminal, time::Duration};

use anyhow::Result;
use pico_args::Arguments;
use serde::{Deserialize, Serialize};
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

mod context;
mod file;
mod iter;
mod print;

pub use context::*;
pub use iter::*;

use crate::file::PREVIOUS;

#[derive(Serialize, Deserialize, Default)]
pub struct BenchmarkFile {
    pub results: HashMap<String, Vec<SubBenchResult>>,
}

impl BenchmarkFile {
    pub fn get_result(
        &self,
        benchmark_name: &str,
        sub_benchmark_name: &str,
    ) -> Option<&SubBenchResult> {
        self.results
            .get(benchmark_name)?
            .iter()
            .find(|r| r.name == sub_benchmark_name)
    }
}

#[derive(Serialize, Deserialize)]
pub struct SubBenchResult {
    /// Name of the subbenchmark.
    pub name: String,
    /// Average duration per iteration of the subbenchmark.
    pub avg_duration_per_iteration: Duration,
    /// Total number of iterations executed.
    pub iterations: u32,
    /// Throughput unit description. e.g., "bytes", "elements", etc.
    pub throughput_unit: String,
    /// Number of throughput units processed per iteration.
    pub throughput_count_per_iteration: u32,
}

impl SubBenchResult {
    pub fn throughput_per_second(&self) -> f64 {
        let secs_f64 = self.avg_duration_per_iteration.as_secs_f64();
        if secs_f64 == 0.0 {
            return 0.0;
        }
        self.throughput_count_per_iteration as f64 / secs_f64
    }
}

pub struct Benchmark {
    pub name: &'static str,
    pub func: fn(BenchmarkContext) -> Result<Vec<SubBenchResult>>,
}

const HELP: &str = "\
Usage: wgpu-benchmark [OPTIONS] [BENCHMARK_NAME]

Modes:
    --bench                     Run in benchmark mode, comparing against previous results.
    --list                      List available benchmarks.
    <no flag>                   Run in test mode, executing each benchmark exactly once.

Test Matching:
    --exact                     When specifying BENCHMARK_NAME, only run exact matches.
    BENCHMARK_NAME              Only run benchmarks whose names contain this substring.

Comparison:
    -b, --baseline NAME         Specify a baseline file for comparison.
    -s, --save-baseline NAME    Save the results as a baseline file.

Timings:
    --iters N                   Override number of iterations per benchmark.
    --time SECONDS              Override time per benchmark in seconds.

Other:
    --color                     Set colored output (always,always-ansi,auto,never).
    --format terse              Specify --list output format (only 'terse' is supported).
    --no-capture                (Ignored)
";

pub fn main(benchmarks: Vec<Benchmark>) {
    let mut args = Arguments::from_env();

    let help = args.contains(["-h", "--help"]);

    if help {
        println!("{HELP}");
        return;
    }

    let mut color: ColorChoice = args
        .opt_value_from_str("--color")
        .unwrap_or(None)
        .unwrap_or(ColorChoice::Auto);
    if color == ColorChoice::Auto && !std::io::stdin().is_terminal() {
        color = ColorChoice::Never;
    }

    let exact = args.contains("--exact");
    // We don't actually need this flag, but cargo-nextest passes it in
    // test mode, so we need to accept it.
    let _no_capture = args.contains("--no-capture");

    #[expect(clippy::manual_map, reason = "So much clearer this way")]
    let mut override_iterations = if let Some(iters) = args.opt_value_from_str("--iters").unwrap() {
        Some(LoopControl::Iterations(iters))
    } else if let Some(seconds) = args.opt_value_from_str("--time").unwrap() {
        Some(LoopControl::Time(Duration::from_secs_f64(seconds)))
    } else {
        None
    };

    let baseline_name: Option<String> = args.opt_value_from_str(["-b", "--baseline"]).unwrap();
    let write_baseline: Option<String> =
        args.opt_value_from_str(["-s", "--save-baseline"]).unwrap();

    let is_bench = args.contains("--bench");
    let is_list = args.contains("--list");
    let is_test = !is_bench && !is_list;

    let format: Option<String> = args.opt_value_from_str("--format").unwrap();

    if let Some(fmt) = format {
        assert_eq!(fmt, "terse", "Only 'terse' format is supported.");
    }
    if let Some(ref baseline) = baseline_name {
        if baseline == PREVIOUS {
            eprintln!("Cannot use '{PREVIOUS}' as a baseline name.");
            return;
        }
    }
    if let Some(ref write_baseline) = write_baseline {
        if write_baseline == PREVIOUS {
            eprintln!("Cannot use '{PREVIOUS}' as a baseline name.");
            return;
        }
    }

    if override_iterations.is_none() && is_test {
        override_iterations = Some(LoopControl::Iterations(1));
    }

    let name = args.free_from_str::<String>().ok();

    let baseline = if is_bench {
        let res = file::get_comparison_file(baseline_name.as_deref());

        match (&res, baseline_name.as_deref()) {
            (Some(_), Some(baseline)) => {
                println!("Using baseline \"{baseline}\" for comparison.\n")
            }
            (None, Some(baseline)) => {
                eprintln!("Could not find baseline named {baseline:?}.\n");
                return;
            }
            (Some(_), None) => {
                println!("Using previous benchmark results for comparison.\n");
            }
            (None, None) => {
                println!("No previous benchmark results found for comparison.\n");
            }
        }

        res
    } else {
        None
    };

    let mut output_file = BenchmarkFile::default();

    let mut stdout = StandardStream::stdout(color);

    for bench in benchmarks {
        if let Some(ref bench_name) = name {
            if exact {
                if bench.name != bench_name {
                    continue;
                }
            } else if !bench.name.contains(bench_name) {
                continue;
            }
        }

        if is_list {
            println!("{}: benchmark", bench.name);
            continue;
        }

        let ctx = BenchmarkContext {
            override_iters: override_iterations,
            default_iterations: LoopControl::default(),
            is_test,
        };

        stdout
            .set_color(ColorSpec::new().set_fg(Some(Color::Blue)))
            .unwrap();
        println!("Running benchmark: {}", bench.name);
        stdout.reset().unwrap();

        let results = {
            profiling::scope!("bench", bench.name);
            let r = (bench.func)(ctx);
            match r {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  Error running benchmark '{}': {:?}", bench.name, e);
                    continue;
                }
            }
        };

        let previous_results = if let Some(ref baseline) = baseline {
            baseline.results.get(bench.name).map(|r| r.as_slice())
        } else {
            None
        };

        print::print_results(&mut stdout, &results, previous_results);

        output_file.results.insert(bench.name.to_string(), results);
    }

    file::write_results_file(PREVIOUS, &output_file).unwrap();
    if let Some(output_baseline) = write_baseline {
        file::write_results_file(&output_baseline, &output_file).unwrap();
    }
}
