use std::process::ExitCode;

use anyhow::Context;
use cli::{Args, Subcommand};

mod cli;

fn main() -> ExitCode {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .format_indent(Some(0))
        .init();

    let args = Args::parse();

    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("{e:?}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> anyhow::Result<()> {
    let Args { subcommand } = args;
    match subcommand {
        Subcommand::RunWasm { args } => {
            cargo_run_wasm::run_wasm_with_css_and_args(
                "body { margin: 0px; }",
                cargo_run_wasm::Args::from_args(args)
                    .map_err(anyhow::Error::msg)
                    .context("failed to parse arguments for `cargo-run-wasm`")?,
            );
            Ok(())
        }
    }
}
