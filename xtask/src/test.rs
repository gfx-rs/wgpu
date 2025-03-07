use anyhow::Context;
use pico_args::Arguments;
use xshell::Shell;

pub fn run_tests(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    let llvm_cov = args.contains("--llvm-cov");
    let list = args.contains("--list");
    // Retries handled by cargo nextest natively

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
            .args(args.finish())
            .run()
            .context("Failed to list tests")?;
        return Ok(());
    }
    log::info!("Running cargo tests");

    shell
        .cmd("cargo")
        .args(llvm_cov_nextest_flags)
        .args(["--benches", "--tests", "--all-features"])
        .args(args.finish())
        .quiet()
        .run()
        .context("Tests failed")?;

    log::info!("Finished tests");

    Ok(())
}
