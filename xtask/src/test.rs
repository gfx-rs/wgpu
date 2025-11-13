use std::ffi::OsString;

use anyhow::Context;
use pico_args::Arguments;
use xshell::Shell;

use crate::{install_warp, util::flatten_args};

pub fn run_tests(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
) -> anyhow::Result<()> {
    let llvm_cov = args.contains("--llvm-cov");
    let list = args.contains("--list");

    // Determine the build profile from arguments
    let is_release = args.contains("--release");
    let custom_profile = args
        .opt_value_from_str::<_, String>("--cargo-profile")
        .ok()
        .flatten();
    let profile = if is_release {
        "release"
    } else if let Some(ref p) = custom_profile {
        p.as_str()
    } else {
        "debug"
    };

    let mut cargo_args = flatten_args(args, passthrough_args);

    // Re-add profile flags that were consumed during argument parsing
    if is_release {
        cargo_args.insert(0, OsString::from("--release"));
    } else if let Some(ref p) = custom_profile {
        cargo_args.insert(0, OsString::from(format!("--cargo-profile={p}")));
    }

    // Retries handled by cargo nextest natively

    // Install WARP on Windows for D3D12 testing
    if cfg!(target_os = "windows") {
        let llvm_cov_dir = if llvm_cov {
            "target/llvm-cov-target"
        } else {
            "target"
        };
        let target_dir = format!("{llvm_cov_dir}/{profile}");
        install_warp::install_warp(&shell, &target_dir)?;
    }

    // These needs to match the command in "run wgpu-info" in `.github/workflows/ci.yml`
    let llvm_cov_flags: &[_] = if llvm_cov {
        &["llvm-cov", "--no-cfg-coverage", "--no-report"]
    } else {
        &[]
    };
    let llvm_cov_nextest_flags: &[_] = if llvm_cov {
        &["llvm-cov", "--no-cfg-coverage", "--no-report", "nextest"]
    } else if list {
        &["nextest", "list"]
    } else {
        &["nextest", "run"]
    };

    log::info!("Generating .gpuconfig file based on gpus on the system");

    shell
        .cmd("cargo")
        .args(llvm_cov_flags)
        .args([
            "run",
            "--bin",
            "wgpu-info",
            "--",
            "--json",
            "-o",
            ".gpuconfig",
        ])
        .quiet()
        .run()
        .context("Failed to run wgpu-info to generate .gpuconfig")?;

    let gpu_count = shell
        .read_file(".gpuconfig")
        .unwrap()
        .lines()
        .filter(|line| line.contains("name"))
        .count();

    log::info!(
        "Found {} gpu{}",
        gpu_count,
        if gpu_count == 1 { "" } else { "s" }
    );

    if list {
        log::info!("Listing tests");
        shell
            .cmd("cargo")
            .args(llvm_cov_nextest_flags)
            .args(["-v", "--benches", "--tests", "--all-features"])
            .args(cargo_args)
            .run()
            .context("Failed to list tests")?;
        return Ok(());
    }
    log::info!("Running cargo tests");

    shell
        .cmd("cargo")
        .args(llvm_cov_nextest_flags)
        .args(["--benches", "--tests", "--all-features"])
        .args(cargo_args)
        .quiet()
        .run()
        .context("Tests failed")?;

    log::info!("Finished tests");

    Ok(())
}
