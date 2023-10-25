use std::process::exit;

use anyhow::{anyhow, bail, Context};
use pico_args::Arguments;

const HELP: &str = "\
Usage: xtask <COMMAND>

Commands:
  run-wasm
  test
    --llvm-cov  Run tests with LLVM code coverage using the llvm-cov tool

Options:
  -h, --help  Print help
";

pub struct Args {
    pub subcommand: Subcommand,
    pub command_args: Arguments,
}

impl Args {
    pub fn parse() -> Self {
        let mut args = Arguments::from_env();
        if args.contains("--help") {
            eprint!("{HELP}");
            // Emulate Cargo exit status:
            // <https://doc.rust-lang.org/cargo/commands/cargo.html#exit-status>
            let cargo_like_exit_code = 101;
            exit(cargo_like_exit_code);
        }
        match Subcommand::parse(&mut args) {
            Ok(subcommand) => Self {
                subcommand,
                command_args: args,
            },
            Err(e) => {
                eprintln!("{:?}", anyhow!(e));
                exit(1)
            }
        }
    }
}

pub enum Subcommand {
    RunWasm,
    Test,
}

impl Subcommand {
    /// Returns the name of the subcommand as a string.
    ///
    /// Opposite of [`Self::parse`].
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::RunWasm => "run-wasm",
            Self::Test => "test",
        }
    }

    /// Returns true if all required features are enabled for this subcommand.
    pub fn required_features_enabled(&self) -> bool {
        match self {
            Self::RunWasm => cfg!(feature = "run-wasm"),
            Self::Test => true,
        }
    }

    /// Comma separated list of features required by this subcommand.
    pub fn features(&self) -> &'static str {
        match self {
            Self::RunWasm => "run-wasm",
            // We will never ask for the features if required_features_enabled always returns true.
            Self::Test => unreachable!(),
        }
    }

    fn parse(args: &mut Arguments) -> anyhow::Result<Subcommand> {
        let subcmd = args
            .subcommand()
            .context("failed to parse subcommand")?
            .context("no subcommand specified; see `--help` for more details")?;
        match &*subcmd {
            "run-wasm" => Ok(Self::RunWasm),
            "test" => Ok(Self::Test),
            other => {
                bail!("unrecognized subcommand {other:?}; see `--help` for more details")
            }
        }
    }
}
