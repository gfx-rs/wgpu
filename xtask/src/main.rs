#![cfg_attr(target_arch = "wasm32", no_main)]
#![cfg(not(target_arch = "wasm32"))]

use std::process::ExitCode;

use anyhow::Context;
use pico_args::Arguments;

mod changelog;
mod cts;
mod install_warp;
mod miri;
mod run_wasm;
mod test;
mod util;
mod vendor_web_sys;

const HELP: &str = "\
Usage: xtask <COMMAND>

Commands:
  cts [<options>] [<test selector...> | -f <test list file...> | -- <args...>]
    Check out, build, and run CTS tests

    If no command-line arguments are specified, runs the default test list
    in `cts_runner/test.lst`, with quiet mode enabled.

    --skip-checkout         Don't check out the pinned CTS version, use whatever
                            is already checked out.
    --release               Build and run in release mode
    --llvm-cov              Run with LLVM code coverage
    --backend <backend>     Specify the backend (metal, dx12, or vulkan). Used
                            to evaluate `fails-if` conditions in the test list.
    --filter <regex>        Filter tests by selector using a regex pattern.
                            Prefix with '!' to invert (exclude matching tests).
                            Applied after all tests are collected.
    --quiet                 Only show test counts for suites without failures.
    --verbose               Show full output when running the default test list.

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

  changelog [from_branch] [to_commit]
    Audit changes in the `CHANGELOG.md` at the root of the repo. Ensure that:

    1. All changes are in an `Unreleased` section.

        `<from_branch>` is used to determine the base of the diff to be performed. The base is set to fork point between `<to_commit>` and this branch.

        `<to_commit>` is the tip of the `git diff` that will be used for checking (1).

    --allow-released-changes  Only reports issues as warnings, rather than reporting errors and forcing a non-zero exit code.

  miri
    Run all miri-compatible tests under miri. Requires a nightly toolchain
    with the x86_64-unknown-linux-gnu target and miri component installed.

    --toolchain <toolchain>   The toolchain to use for miri tests.
                              Must be a nightly toolchain.
                              Defaults to `nightly`.

  vendor-web-sys
    Re-vendor the WebGPU web-sys bindings.

    --no-cleanup        Don't clean up temporary checkout of wasm-bindgen
    One of:
        --path-to-checkout  Path to a local checkout of wasm-bindgen to generate bindings from.
                            This is useful for testing changes to wasm-bindgen
        --version           String that can be passed to `git checkout` to checkout the wasm-bindgen repository.

  install-warp
    Download and install the WARP (D3D12 software implementation) DLL for D3D12 testing.

    --target-dir <dir>    The target directory to install WARP into.
    --profile <profile>   The cargo profile to install WARP for (default: debug)

    Note: Cannot specify both --target-dir and --profile

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

    let mut args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let passthrough_args = args
        .iter()
        .position(|arg| arg == "--")
        .map(|pos| args.drain(pos..).skip(1).collect());
    let mut args = Arguments::from_vec(args);

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
        Some("changelog") => changelog::check_changelog(shell, args)?,
        Some("cts") => cts::run_cts(shell, args, passthrough_args)?,
        Some("run-wasm") => run_wasm::run_wasm(shell, args, passthrough_args)?,
        Some("miri") => miri::run_miri(shell, args)?,
        Some("test") => test::run_tests(shell, args, passthrough_args)?,
        Some("vendor-web-sys") => vendor_web_sys::run_vendor_web_sys(shell, args)?,
        Some("install-warp") => install_warp::run_install_warp(shell, args)?,
        Some(subcommand) => {
            bad_arguments!("Unknown subcommand: {}", subcommand)
        }
        None => {
            bad_arguments!("Expected subcommand")
        }
    }

    Ok(ExitCode::SUCCESS)
}
