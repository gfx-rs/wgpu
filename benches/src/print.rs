use std::collections::HashMap;
use std::io::Write;

use termcolor::{Color, ColorSpec, StandardStream, WriteColor};

use crate::SubBenchResult;

#[derive(Default, Clone)]
struct Delta {
    throughput_change_str: String,
    throughput_change: f64,
    time_change_str: String,
    time_change: f64,
}

impl Delta {
    fn new(previous: &SubBenchResult, current: &SubBenchResult) -> Self {
        let prev_throughput = previous.throughput_per_second();
        let curr_throughput = current.throughput_per_second();
        let delta_throughput = if prev_throughput != 0.0 {
            (curr_throughput - prev_throughput) / prev_throughput * 100.0
        } else {
            0.0
        };
        let throughput_change = format!(" ({delta_throughput:+.2}%)");

        let prev_time = previous.avg_duration_per_iteration;
        let curr_time = current.avg_duration_per_iteration;
        let delta_time = if prev_time.as_nanos() != 0 {
            (curr_time.as_secs_f64() - prev_time.as_secs_f64()) / prev_time.as_secs_f64() * 100.0
        } else {
            0.0
        };

        let time_change = format!("{delta_time:+.2}%; ");

        Delta {
            throughput_change_str: throughput_change,
            throughput_change: delta_throughput,
            time_change_str: time_change,
            time_change: delta_time,
        }
    }
}

/// Get a color spec for the given change percentage.
///
/// Positive changes are red (regression), negative changes are green (improvement).
/// This represents changes for time durations. For throughput changes, the sign should be inverted
/// before passing to this method.
fn get_change_color(percent_change: f64) -> ColorSpec {
    let mut color_spec = ColorSpec::new();
    if percent_change > 3.0 {
        color_spec.set_fg(Some(Color::Red));
    } else if percent_change < -3.0 {
        color_spec.set_fg(Some(Color::Green));
    } else {
        color_spec.set_fg(Some(Color::Yellow));
    }
    if percent_change.abs() > 15.0 {
        color_spec.set_intense(true);
    }
    color_spec
}

pub fn print_results(
    stdout: &mut StandardStream,
    results: &[SubBenchResult],
    previous_results: Option<&[SubBenchResult]>,
) {
    let mut deltas = HashMap::new();
    if let Some(previous_results) = previous_results {
        for result in results {
            if let Some(previous_result) = previous_results.iter().find(|r| r.name == result.name) {
                deltas.insert(result.name.clone(), Delta::new(previous_result, result));
            }
        }
    }

    let longest_throughput_change_len = deltas
        .values()
        .map(|d| d.throughput_change_str.len())
        .max()
        .unwrap_or(0);
    let longest_time_change_len = deltas
        .values()
        .map(|d| d.time_change_str.len())
        .max()
        .unwrap_or(0);

    let longest_name_len = results.iter().map(|r| r.name.len()).max().unwrap_or(0);
    let duration_strings: Vec<String> = results
        .iter()
        .map(|r| format!("{:.3?}", r.avg_duration_per_iteration))
        .collect();
    let longest_duration_len = duration_strings.iter().map(|s| s.len()).max().unwrap_or(0);

    let iterations_strings: Vec<String> = results
        .iter()
        .map(|r| format!("{}", r.iterations))
        .collect();
    let longest_iterations_len = iterations_strings
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(0);

    let throughput_strings: Vec<String> = results
        .iter()
        .map(|r| {
            let throughput_per_second = r.throughput_count_per_iteration as f64
                / r.avg_duration_per_iteration.as_secs_f64();
            human_scale(throughput_per_second)
        })
        .collect();
    let longest_throughput_len = throughput_strings
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(0);

    let longest_throughput_unit_len = results
        .iter()
        .map(|r| r.throughput_unit.len())
        .max()
        .unwrap_or(0);

    for (i, result) in results.iter().enumerate() {
        let delta = deltas.get(&result.name).cloned().unwrap_or_default();
        let time_color = get_change_color(delta.time_change);
        let throughput_color = get_change_color(-delta.throughput_change);

        stdout
            .set_color(ColorSpec::new().set_fg(Some(Color::Cyan)))
            .unwrap();
        write!(stdout, "    {:>longest_name_len$}: ", result.name).unwrap();

        stdout.set_color(&time_color).unwrap();
        write!(stdout, "{:>longest_duration_len$} ", duration_strings[i],).unwrap();
        stdout.reset().unwrap();
        write!(stdout, "(").unwrap();
        stdout.set_color(&time_color).unwrap();
        write!(
            stdout,
            "{:>longest_time_change_len$}",
            delta.time_change_str
        )
        .unwrap();
        stdout.reset().unwrap();

        write!(
            stdout,
            "over {:>longest_iterations_len$} iter) ",
            result.iterations,
        )
        .unwrap();

        stdout.set_color(&throughput_color).unwrap();
        write!(stdout, "{:>longest_throughput_len$}", throughput_strings[i]).unwrap();
        stdout.reset().unwrap();
        write!(
            stdout,
            " {:>longest_throughput_unit_len$}/s",
            result.throughput_unit,
        )
        .unwrap();
        stdout.set_color(&throughput_color).unwrap();
        writeln!(
            stdout,
            "{:>longest_throughput_change_len$}",
            delta.throughput_change_str
        )
        .unwrap();
        stdout.reset().unwrap();
    }
    println!();
}

fn human_scale(value: f64) -> String {
    const PREFIXES: &[&str] = &["", "K", "M", "G", "T", "P"];

    if value == 0.0 {
        return "0".to_string();
    }

    let abs_value = value.abs();
    let exponent = (abs_value.log10() / 3.0).floor() as usize;
    let prefix_index = exponent.min(PREFIXES.len() - 1);

    let scaled = value / 10_f64.powi((prefix_index * 3) as i32);

    // Determine decimal places for 3 significant figures
    let decimal_places = if scaled.abs() >= 100.0 {
        0
    } else if scaled.abs() >= 10.0 {
        1
    } else {
        2
    };

    format!(
        "{:.prec$}{}",
        scaled,
        PREFIXES[prefix_index],
        prec = decimal_places
    )
}
