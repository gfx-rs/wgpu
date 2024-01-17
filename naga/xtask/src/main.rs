use std::process::ExitCode;

use anyhow::Context;
use cli::Args;

use crate::{
    cli::Subcommand,
    fs::remove_dir_all,
    path::join_path,
    process::{which, EasyCommand},
};

mod cli;
mod fs;
mod glob;
mod jobserver;
mod path;
mod process;
mod result;
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
        Subcommand::Bench { clean } => {
            if clean {
                let criterion_artifact_dir = join_path(["target", "criterion"]);
                log::info!("removing {}", criterion_artifact_dir.display());
                remove_dir_all(&criterion_artifact_dir)
                    .with_context(|| format!("failed to remove {criterion_artifact_dir:?}"))?;
            }
            EasyCommand::simple("cargo", ["bench"]).success()
        }
        Subcommand::Validate(cmd) => validate::validate(cmd),
    }
}
