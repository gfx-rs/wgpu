use std::process::exit;

use anyhow::{anyhow, bail, Context};
use pico_args::Arguments;

const HELP: &str = "\
Usage: xtask <COMMAND>

Commands:
  run-wasm

Options:
  -h, --help  Print help
";

pub(crate) struct Args {
    pub(crate) subcommand: Subcommand,
}

impl Args {
    pub fn parse() -> Self {
        let mut args = Arguments::from_env();
        log::debug!("parsing args: {args:?}");
        if args.contains("--help") {
            eprint!("{HELP}");
            // Emulate Cargo exit status:
            // <https://doc.rust-lang.org/cargo/commands/cargo.html#exit-status>
            let cargo_like_exit_code = 101;
            exit(cargo_like_exit_code);
        }
        match Subcommand::parse(args).map(|subcommand| Self { subcommand }) {
            Ok(this) => this,
            Err(e) => {
                eprintln!("{:?}", anyhow!(e));
                exit(1)
            }
        }
    }
}

pub(crate) enum Subcommand {
    RunWasm { args: Arguments },
}

impl Subcommand {
    fn parse(mut args: Arguments) -> anyhow::Result<Subcommand> {
        let subcmd = args
            .subcommand()
            .context("failed to parse subcommand")?
            .context("no subcommand specified; see `--help` for more details")?;
        match &*subcmd {
            "run-wasm" => Ok(Self::RunWasm { args }),
            other => {
                bail!("unrecognized subcommand {other:?}; see `--help` for more details")
            }
        }
    }
}
