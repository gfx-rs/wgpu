use std::{
    collections::BTreeSet,
    ffi::OsString,
    fs,
    io::BufReader,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result};
use nextest_metadata::{ListCommand, TestListSummary};
use pico_args::Arguments;
use quick_junit::Report;
use termcolor::Color;
use xshell::Shell;

use crate::{install_warp, util::flatten_args};

mod analyze;
mod baselines;
mod report;

use analyze::analyze_report;
use baselines::{
    clear_baselines_on_disk, load_baseline, opt_value_alias, print_baseline_list, save_baseline,
};
use report::{print_general, print_outcome, print_section_line};

// Entry point for `cargo xtask test`.
//
// The flow is intentionally linear and easy to follow:
// 1) run setup (gpuconfig),
// 2) run nextest with JUnit output,
// 3) reconcile results against expectations/baseline,
// 4) render a contributor-focused summary.
pub fn run_tests(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
) -> Result<()> {
    let llvm_cov = args.contains("--llvm-cov");
    let list = args.contains("--list");
    let skip_analysis = args.contains("--skip-analysis");
    let verbose = args.contains("--verbose");
    let list_baselines = args.contains("--list-baselines");
    let clear_baselines = args.contains("--clear-baselines");
    let baseline_name = opt_value_alias(&mut args, "--baseline", "-b", "baseline")?;
    let save_baseline_name = opt_value_alias(&mut args, "--save-baseline", "-s", "save-baseline")?;

    if clear_baselines {
        let cleared = clear_baselines_on_disk()?;
        print_section_line(format!("Cleared {} baseline files.", cleared), Color::Green);
    }
    if list_baselines {
        print_baseline_list()?;
        return Ok(());
    }
    if clear_baselines {
        return Ok(());
    }

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

    // Re-add profile flags that were consumed during argument parsing.
    // We need to reconstruct the original flags and add them back to cargo_args
    // so that cargo nextest receives the correct profile selection.
    // Using explicit if-else is clearer than map() since we're selecting from
    // three different states (release flag, custom profile, or neither).
    #[expect(clippy::manual_map)]
    let profile_arg = if is_release {
        Some(OsString::from("--release"))
    } else if let Some(ref p) = custom_profile {
        Some(OsString::from(format!("--cargo-profile={p}")))
    } else {
        None
    };

    if let Some(ref profile_arg) = profile_arg {
        cargo_args.insert(0, profile_arg.clone());
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

    let test_suite_run_flags: &[_] = if llvm_cov {
        &["llvm-cov", "--no-cfg-coverage", "--no-report", "nextest"]
    } else {
        &["nextest", "run"]
    };

    print_general("Generating .gpuconfig file based on gpus on the system");

    // We use a test to generate the .gpuconfig file instead of using the cli directly
    // as `cargo run --bin wgpu-info` would build a different set of dependencies, causing
    // incremental changes to need to rebuild the wgpu stack twice, one for the tests
    // and once for the cli binary.
    //
    // Needs to be kept in sync with the test in wgpu-info/src/tests.rs
    shell
        .cmd("cargo")
        .args(test_suite_run_flags)
        // Use the same build configuration as the main tests, so that we only build once.
        .args(["--benches", "--tests", "--all-features"])
        // Use the same cargo profile as the main tests.
        .args(profile_arg)
        // We need to tell nextest to filter by binary too, so it doesn't try to enumerate
        // tests on any of the gpu enabled test binaries, as that will fail due to
        // old or missing .gpuconfig files.
        .args(["-E", "binary(wgpu-info)", "generate_gpuconfig_report"])
        // Turn on the env var for saving the .gpuconfig files
        .env("WGPU_INFO_SAVE_GPUCONFIG_REPORT", "1")
        .quiet()
        .run()
        .context("Failed to run tests to generate .gpuconfig")?;

    let gpu_count = shell
        .read_file(".gpuconfig")
        .unwrap()
        .lines()
        .filter(|line| line.contains("name"))
        .count();

    // Manually pluralize "gpu" since count_phrase helper is in the report module.
    print_general(format!(
        "Found {} gpu{}",
        gpu_count,
        if gpu_count == 1 { "" } else { "s" }
    ));

    if list {
        print_general("Listing tests");
        shell
            .cmd("cargo")
            .args(["nextest", "list"])
            .args(["-v", "--benches", "--tests", "--all-features"])
            .args(cargo_args)
            .run()
            .context("Failed to list tests")?;
        return Ok(());
    }

    if skip_analysis {
        print_general("Running cargo tests");
        shell
            .cmd("cargo")
            .args(test_suite_run_flags)
            .args(["--benches", "--tests", "--all-features"])
            .args(cargo_args)
            .quiet()
            .run()
            .context("Tests failed")?;
        print_section_line("Finished tests.", Color::Green);
        return Ok(());
    }

    // Resolve baseline early so a typo in `--baseline` fails fast before running tests.
    let baseline = load_baseline(baseline_name.as_deref())?;

    let inventory = collect_test_inventory()?;

    let repo_root = repository_root()?;
    let (tool_config_path, junit_path) = make_nextest_tool_config(&cargo_args)?;
    let tool_config_arg = format!("wgpu-xtask:{}", tool_config_path.display());

    print_general("Running cargo tests with expectation post-processing");
    let mut run_command = Command::new("cargo");
    run_command.current_dir(&repo_root);
    if llvm_cov {
        run_command.args(["llvm-cov", "--no-cfg-coverage", "--no-report", "nextest"]);
    } else {
        run_command.args(["nextest", "run"]);
    }
    run_command.args(["--benches", "--tests", "--all-features"]);
    run_command.args(&cargo_args);
    run_command.args(["--tool-config-file", &tool_config_arg]);
    run_command.env("NEXTEST_MAX_PROGRESS_RUNNING", "0");

    let run_status = run_command
        .status()
        .context("Failed to invoke cargo nextest run")?;

    let report = read_junit_report(&junit_path)?;
    let outcome = analyze_report(report, &baseline, inventory);

    if let Some(save_baseline_name) = save_baseline_name.as_deref() {
        save_baseline(save_baseline_name, &outcome.current_baseline)?;
    }
    print_outcome(&outcome, verbose);

    if outcome.success() {
        if !run_status.success() {
            if !outcome.known_failure_tests.is_empty() {
                print_section_line(
                    "nextest reported failures, and expectation reconciliation classified them as known failures.",
                    Color::Yellow,
                );
            } else {
                anyhow::bail!(
                    "nextest reported failures, but no reconciled failures were recorded; treating this as unexpected infra failure"
                );
            }
        }
        print_section_line("Finished tests.", Color::Green);
        return Ok(());
    }

    anyhow::bail!("Tests failed after expectation reconciliation")
}

fn collect_test_inventory() -> Result<BTreeSet<String>> {
    let mut list_command = ListCommand::new();
    list_command.add_args(["--benches", "--tests", "--all-features"]);
    let summary = list_command
        .exec()
        .context("Failed to gather nextest test inventory")?;
    Ok(inventory_from_summary(&summary))
}

fn inventory_from_summary(summary: &TestListSummary) -> BTreeSet<String> {
    // Intentionally include all listed test cases from nextest metadata, even if
    // they don't match the current run filters. This keeps add/remove reporting
    // meaningful when users run focused subsets.
    let mut inventory = BTreeSet::new();
    for suite in summary.rust_suites.values() {
        for test_case_name in suite.test_cases.keys() {
            inventory.insert(format!(
                "{}::{}",
                suite.binary.binary_id,
                test_case_name.as_str()
            ));
        }
    }
    inventory
}

fn repository_root() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(|path| path.to_path_buf())
        .with_context(|| {
            format!(
                "Failed to resolve repository root from CARGO_MANIFEST_DIR: {}",
                manifest_dir.display()
            )
        })
}

fn detect_nextest_profile(cargo_args: &[OsString]) -> String {
    for (index, arg) in cargo_args.iter().enumerate() {
        let arg = arg.to_string_lossy();
        if arg == "--profile" || arg == "-P" {
            if let Some(next_arg) = cargo_args.get(index + 1) {
                return next_arg.to_string_lossy().into_owned();
            }
        } else if let Some(profile) = arg.strip_prefix("--profile=") {
            return profile.to_owned();
        } else if let Some(profile) = arg.strip_prefix("-P") {
            if !profile.is_empty() {
                return profile.to_owned();
            }
        }
    }
    "default".to_owned()
}

fn make_nextest_tool_config(cargo_args: &[OsString]) -> Result<(PathBuf, PathBuf)> {
    let root = repository_root()?;
    let profile = detect_nextest_profile(cargo_args);
    let output_dir = root.join("target").join("nextest").join(&profile);
    fs::create_dir_all(&output_dir).with_context(|| {
        format!(
            "Failed to create nextest output directory {}",
            output_dir.display()
        )
    })?;

    // Kept in sync with xtask/src/test/nextest-junit.toml.
    let junit_path = output_dir.join("nextest-junit.xml");
    let _ = fs::remove_file(&junit_path);

    let config_path = root
        .join("xtask")
        .join("src")
        .join("test")
        .join("nextest-junit.toml");
    if !config_path.exists() {
        anyhow::bail!(
            "Missing nextest JUnit config at {}; expected xtask\\src\\test\\nextest-junit.toml",
            config_path.display()
        );
    }

    Ok((config_path, junit_path))
}

fn read_junit_report(junit_path: &Path) -> Result<Report> {
    let file = fs::File::open(junit_path).with_context(|| {
        format!(
            "Failed to open JUnit report at {}. Ensure nextest emitted a report.",
            junit_path.display()
        )
    })?;
    let reader = BufReader::new(file);
    Report::deserialize(reader).context("Failed to deserialize JUnit report")
}
