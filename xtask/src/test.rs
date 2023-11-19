use std::ffi::OsStr;

use anyhow::Context;
use pico_args::Arguments;
use walkdir::WalkDir;

pub fn run_tests(mut args: Arguments) -> anyhow::Result<()> {
    let llvm_cov = args.contains("--llvm-cov");
    let partition: Option<u32> = args.value_from_str("--partition").ok();
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

    if let Some(partition) = partition {
        log::info!("Running tests over {} partitions", partition);

        let mut failed = false;
        for i in 1..=partition {
            let partition_arg = format!("hash:{i}/{partition}");
            let output_path_arg = format!("coverage-partition-{i}.info");

            log::info!("Running partition {} of {}", i, partition);

            // Execute partition
            failed |= xshell::cmd!(
                shell,
                "cargo llvm-cov --no-cfg-coverage --no-report nextest --all-features --no-fail-fast --retries 2 --partition {partition_arg}"
            )
            .args(args.clone().finish())
            .quiet()
            .run()
            .is_err();

            log::info!("Building report to {output_path_arg}");

            // Build report
            xshell::cmd!(
                shell,
                "cargo llvm-cov report --lcov --output-path {output_path_arg}"
            )
            .quiet()
            .run()
            .context("Report failed")?;

            log::info!("Clearing profraw files");

            // Clear profraw

            let mut output_bytes = 0_u64;

            for entry in WalkDir::new(".").into_iter() {
                if let Ok(entry) = entry {
                    if entry.path().extension().and_then(OsStr::to_str) == Some("profraw") {
                        output_bytes += entry.metadata().expect("Could not find metadata").len();
                        std::fs::remove_file(entry.path()).expect("Could not remove file");
                    }
                }
            }

            log::info!(
                "Partition {} of {} finished, {:.0} megabytes of profraw files removed",
                i,
                partition,
                output_bytes as f32 / (1024.0 * 1024.0)
            );
        }
        if failed {
            anyhow::bail!("Some partitions failed");
        }
    } else {
        log::info!("Running tests");

        xshell::cmd!(
            shell,
            "cargo {llvm_cov_nextest_flags...} --all-features --no-fail-fast --retries 2"
        )
        .args(args.finish())
        .quiet()
        .run()
        .context("Running tests failed")?;
    };

    log::info!("Finished tests");

    Ok(())
}
