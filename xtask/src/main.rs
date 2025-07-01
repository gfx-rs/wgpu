#![cfg_attr(target_arch = "wasm32", no_main)]
#![cfg(not(target_arch = "wasm32"))]

use std::process::ExitCode;

use anyhow::Context;
use pico_args::Arguments;

mod cts;
mod run_wasm;
mod test;
mod util;
mod vendor_web_sys;

const HELP: &str = "\
Usage: xtask <COMMAND>

Commands:
  cts [--skip-checkout] [<test selector> | -f <test list file>]...
    Check out, build, and run CTS tests

    --skip-checkout     Don't check out the pinned CTS version, use whatever is
                        already checked out.

  run-wasm
    Build and run web examples

    --release   Build in release mode
    --no-serve  Just build the generated files, don't serve them

  test
    Run tests

    --llvm-cov  Run tests with LLVM code coverage using the llvm-cov tool
    --list      List all of the tests and their executables without running them
    --retries   Number of times to retry failing tests

    All extra arguments will be forwarded to cargo-nextest (NOT wgpu-info)

  vendor-web-sys
    Re-vendor the WebGPU web-sys bindings.

    --no-cleanup        Don't clean up temporary checkout of wasm-bindgen
    One of:
        --path-to-checkout  Path to a local checkout of wasm-bindgen to generate bindings from.
                            This is useful for testing changes to wasm-bindgen
        --version           String that can be passed to `git checkout` to checkout the wasm-bindgen repository.

Options:
  -h, --help  Print help
";

/// Helper macro for printing the help message, then bailing with an error message.
#[macro_export]
macro_rules! bad_arguments {
    ($($arg:tt)*) => {{
        eprintln!("{}", $crate::HELP);
        anyhow::bail!($($arg)*)
    }};
}

fn main() -> anyhow::Result<ExitCode> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .format_indent(Some(0))
        .init();

    let mut args = Arguments::from_env();

    if args.contains(["-h", "--help"]) {
        eprint!("{HELP}");
        return Ok(ExitCode::FAILURE);
    }

    let subcommand = args
        .subcommand()
        .context("Expected subcommand to be UTF-8")?;

    // -- Shell Creation --

    let shell = xshell::Shell::new().context("Couldn't create xshell shell")?;
    shell.change_dir(String::from(env!("CARGO_MANIFEST_DIR")) + "/..");

    match subcommand.as_deref() {
        Some("cts") => cts::run_cts(shell, args)?,
        Some("run-wasm") => run_wasm::run_wasm(shell, args)?,
        Some("test") => test::run_tests(shell, args)?,
        Some("vendor-web-sys") => vendor_web_sys::run_vendor_web_sys(shell, args)?,
        Some(subcommand) => {
            bad_arguments!("Unknown subcommand: {}", subcommand)
        }
        None => {
            bad_arguments!("Expected subcommand")
        }
    }

    Ok(ExitCode::SUCCESS)
}
