mod cli;
mod config;
mod core;
mod error;
mod hooks;
mod output;
mod params;

use anyhow::Context as _;
use clap::Parser as _;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = cli::Args::parse();

    match real_main(&args) {
        Ok(true) => {}
        Ok(false) => {
            // Handled failure already emitted (JSON to stdout or text to stderr).
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

    let mut params = params::build_parameters(args)?;

    // Load config from file or inline JSON and apply on top of the defaults.
    if let Some(ref path) = args.config {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {path}"))?;
        let cfg: config::Config = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse config file: {path}"))?;
        config::apply_config(cfg, &mut params);
    } else if let Some(ref json) = args.config_json {
        let cfg: config::Config =
            serde_json::from_str(json).context("Failed to parse --config-json value")?;
        config::apply_config(cfg, &mut params);
    }

    core::run(args, &mut params)
}
