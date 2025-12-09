#![cfg_attr(target_arch = "wasm32", no_main)]
#![cfg(not(target_arch = "wasm32"))]

use std::process::ExitCode;

use cli::Args;

use crate::{
    cli::Subcommand,
    process::{which, EasyCommand},
};

mod cli;
mod fs;
mod glob;
mod jobserver;
mod path;
mod process;
mod validate;

fn main() -> ExitCode {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .format_indent(Some(0))
        .init();

    jobserver::init();

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

    assert!(which("cargo").is_ok());

    match subcommand {
        Subcommand::All => {
            EasyCommand::simple("cargo", ["fmt"]).success()?;
            EasyCommand::simple("cargo", ["test", "--all-features", "--workspace"]).success()?;
            EasyCommand::simple(
                "cargo",
                [
                    "clippy",
                    "--all-features",
                    "--workspace",
                    "--",
                    "-D",
                    "warnings",
                ],
            )
            .success()?;
            Ok(())
        }
        Subcommand::Validate(cmd) => validate::validate(cmd),
    }
}
