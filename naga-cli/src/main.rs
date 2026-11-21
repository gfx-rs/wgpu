mod cli;
mod config;
mod core;
mod error;
mod hooks;
mod output;
mod params;

use anyhow::Context as _;
use clap::{CommandFactory as _, FromArgMatches as _};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let color = cli::ColorArg::from_raw_args(std::env::args_os());
    error::init_color(color);
    let matches = cli::Args::command().color(color.to_clap()).get_matches();
    let args = cli::Args::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());

    match real_main(&args) {
        Ok(true) => {}
        Ok(false) => {
            // Handled failure already emitted (JSON or text diagnostics).
            std::process::exit(1);
        }
        Err(e) => {
            error::print_err(e.as_ref());
            std::process::exit(1);
        }
    }
}

fn real_main(args: &cli::Args) -> anyhow::Result<bool> {
    if args.print_config_schema {
        let schema = schemars::schema_for!(crate::config::Config);
        println!("{}", serde_json::to_string_pretty(&schema)?);
        return Ok(true);
    }
    if args.print_output_schema {
        let schema = schemars::schema_for!(crate::output::JsonOutput);
        println!("{}", serde_json::to_string_pretty(&schema)?);
        return Ok(true);
    }

    let mut params = params::build_parameters(args)?;

    // Load config from file or inline JSON and apply on top of the defaults.
    let json = if let Some(ref path) = args.config {
        Some((
            std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read config file: {path}"))?,
            format!("Failed to parse config file: {path}"),
        ))
    } else {
        args.config_json
            .clone()
            .map(|json| (json, "Failed to parse --config-json value".to_owned()))
    };
    if let Some((json, context)) = json {
        let cfg: config::Config = serde_json::from_str(&json).context(context)?;
        let raw: serde_json::Value = serde_json::from_str(&json)?;
        config::apply_config(cfg, &raw, &mut params);
    }

    core::run(args, &mut params)
}
