use anyhow::Context;
use pico_args::Arguments;

pub fn run_tests(mut args: Arguments) -> anyhow::Result<()> {
    let llvm_cov = args.contains("--llvm-cov");
    // These needs to match the command in "run wgpu-info" in `.github/workflows/ci.yml`
    let llvm_cov_flags: &[_] = if llvm_cov {
        &["llvm-cov", "--no-cfg-coverage", "--no-report"]
    } else {
        &[]
    };
    let llvm_cov_nextest_flags: &[_] = if llvm_cov {
        &["llvm-cov", "--no-cfg-coverage", "--no-report", "nextest"]
    } else {
        &["nextest", "run"]
    };

    let shell = xshell::Shell::new().context("Couldn't create xshell shell")?;

    shell.change_dir(String::from(env!("CARGO_MANIFEST_DIR")) + "/..");

    log::info!("Generating .gpuconfig file based on gpus on the system");

    xshell::cmd!(
        shell,
        "cargo {llvm_cov_flags...} run --bin wgpu-info -- --json -o .gpuconfig"
    )
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

    log::info!("Running cargo tests");

    xshell::cmd!(
        shell,
        "cargo {llvm_cov_nextest_flags...} --all-features --no-fail-fast --retries 2"
    )
    .args(args.finish())
    .quiet()
    .run()
    .context("Tests failed")?;

    log::info!("Finished tests");

    Ok(())
}
