use std::process::exit;

use anyhow::{anyhow, bail, ensure, Context};
use pico_args::Arguments;

const HELP: &str = "\
Usage: xtask <COMMAND>

Commands:
  all
  bench [--clean]
  validate
    dot
    glsl
    hlsl
      dxc
      fxc
    msl
    spv

Options:
  -h, --help  Print help
";

#[derive(Debug)]
pub(crate) struct Args {
    pub subcommand: Subcommand,
}

impl Args {
    pub fn parse() -> Self {
        let mut args = Arguments::from_env();
        log::debug!("parsing args: {args:?}");
        if args.contains("--help") {
            eprint!("{HELP}");
            exit(101);
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

#[derive(Debug)]
pub(crate) enum Subcommand {
    All,
    Bench { clean: bool },
    Validate(ValidateSubcommand),
}

impl Subcommand {
    fn parse(mut args: Arguments) -> anyhow::Result<Subcommand> {
        args.subcommand()
            .context("failed to parse subcommand")
            .and_then(|parsed| match parsed.as_deref() {
                None => bail!("no subcommand specified; see `--help` for more details"),
                Some("all") => {
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::All)
                }
                Some("bench") => {
                    let clean = args.contains("--clean");
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::Bench { clean })
                }
                Some("validate") => Ok(Self::Validate(ValidateSubcommand::parse(args)?)),
                Some(other) => {
                    bail!("unrecognized subcommand {other:?}; see `--help` for more details")
                }
            })
    }
}

#[derive(Debug)]
pub(crate) enum ValidateSubcommand {
    Spirv,
    Metal,
    Glsl,
    Dot,
    Wgsl,
    Hlsl(ValidateHlslCommand),
}

impl ValidateSubcommand {
    fn parse(mut args: Arguments) -> Result<Self, anyhow::Error> {
        args.subcommand()
            .context("failed to parse `validate` subcommand")
            .and_then(|parsed| match parsed.as_deref() {
                None => bail!("no `validate` subcommand specified; see `--help` for more details"),
                Some("spv") => {
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::Spirv)
                }
                Some("msl") => {
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::Metal)
                }
                Some("glsl") => {
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::Glsl)
                }
                Some("dot") => {
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::Dot)
                }
                Some("wgsl") => {
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::Wgsl)
                }
                Some("hlsl") => Ok(Self::Hlsl(ValidateHlslCommand::parse(args)?)),
                Some(other) => bail!(
                    "unrecognized `validate` subcommand {other:?}; see `--help` for more details"
                ),
            })
    }
}

#[derive(Debug)]
pub(crate) enum ValidateHlslCommand {
    Dxc,
    Fxc,
}

impl ValidateHlslCommand {
    fn parse(mut args: Arguments) -> anyhow::Result<Self> {
        args.subcommand()
            .context("failed to parse `hlsl` subcommand")
            .and_then(|parsed| match parsed.as_deref() {
                None => bail!("no `hlsl` subcommand specified; see `--help` for more details"),
                Some("dxc") => {
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::Dxc)
                }
                Some("fxc") => {
                    ensure_remaining_args_empty(args)?;
                    Ok(Self::Fxc)
                }
                Some(other) => {
                    bail!("unrecognized `hlsl` subcommand {other:?}; see `--help` for more details")
                }
            })
    }
}

fn ensure_remaining_args_empty(args: Arguments) -> anyhow::Result<()> {
    let remaining_args = args.finish();
    ensure!(
        remaining_args.is_empty(),
        "not all arguments were parsed (remaining: {remaining_args:?}); fix your invocation, \
        please!"
    );
    Ok(())
}
