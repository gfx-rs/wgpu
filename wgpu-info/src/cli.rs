use std::{io, process::exit};

use anyhow::Context;

use crate::human::PrintingVerbosity;

const HELP: &str = "\
Usage: wgpu-info [--input <PATH>] [--output <PATH>] [-q/-v/-vv/--json]

Information Density:
  These settings have no effect on the JSON output. The highest verbosity
  provided will be used if multiple are passed.

  -q                  Quiet mode, only print the names and backends of the adapters.
  [default]           Print the adapter info.
  -v                  Additionally print all features, limits, and downlevel capabilities.
  -vv                 Additionally print all texture capabilities and flags.

Options:
  -h, --help          Print this help message.
  -i, --input <PATH>  Read a json report to make it human readable. (\"-\" reads from stdin)
  -o, --output <PATH> Destination to write output to. (\"-\" writes to stdout)
  -j, --json          Output JSON information instead of human-readable text.
";

fn exit_with_help() {
    eprintln!("{HELP}");
    exit(101);
}

pub fn main() -> anyhow::Result<()> {
    let mut args = pico_args::Arguments::from_env();
    // Check for help flag before parsing arguments
    let help = args.contains(["-h", "--help"]);

    if help {
        exit_with_help();
    }

    // Argument parsing
    let input_path: Option<String> = args.opt_value_from_str(["-i", "--input"]).unwrap();
    let output_path: Option<String> = args.opt_value_from_str(["-o", "--output"]).unwrap();
    let json = args.contains(["-j", "--json"]);

    let verbosity = if args.contains("-vv") {
        PrintingVerbosity::InformationFeaturesLimitsTexture
    } else if args.contains("-v") {
        PrintingVerbosity::InformationFeaturesLimits
    } else if args.contains("-q") {
        PrintingVerbosity::NameOnly
    } else {
        PrintingVerbosity::Information
    };

    // Binary OR is intentional, we want all flags to be consumed every iteration.
    while args.contains("-vv") | args.contains("-v") | args.contains("-q") {
        eprintln!(
            "Warning: More than one verbosity flag was passed. Using the most verbose option."
        );
    }

    let remaining = args.finish();
    if !remaining.is_empty() {
        eprint!("Unknown argument(s): ");
        for arg in remaining {
            eprint!("\"{}\" ", arg.to_string_lossy());
        }
        eprint!("\n\n");
        exit_with_help();
    }

    env_logger::init();

    // Generate or load report
    let report = match input_path.as_deref() {
        // Pull report from stdin or file
        Some(path) => {
            let json = if "-" == path {
                std::io::read_to_string(std::io::stdin()).context("Could not read from stdin")?
            } else {
                std::fs::read_to_string(path)
                    .with_context(|| format!("Could not read from file \"{path}\""))?
            };
            crate::report::GpuReport::from_json(&json).context("Could not parse JSON")?
        }
        // Generate the report natively
        None => crate::report::GpuReport::generate(),
    };

    // Setup output writer
    let mut file_handle;
    let mut std_handle;
    let output: &mut dyn io::Write = match output_path.as_deref() {
        None | Some("-") => {
            std_handle = io::stdout();
            &mut std_handle
        }
        Some(path) => {
            file_handle = std::fs::File::create(path)
                .with_context(|| format!("Could not create file \"{path}\""))?;
            &mut file_handle
        }
    };
    let mut output = io::BufWriter::new(output);

    let output_name = output_path.as_deref().unwrap_or("stdout");

    if json {
        report
            .into_json(output)
            .with_context(|| format!("Failed to write to output: {output_name}"))?;
    } else {
        crate::human::print_adapters(&mut output, &report, verbosity)
            .with_context(|| format!("Failed to write to output: {output_name}"))?;
    }

    Ok(())
}
