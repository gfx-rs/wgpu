//! Interface for running the WebGPU CTS (Conformance Test Suite) against wgpu.
//!
//! To run the default set of tests from `cts_runner/test.lst`:
//!
//! ```sh
//! cargo xtask cts
//! ```
//!
//! To run a specific test selector:
//!
//! ```sh
//! cargo xtask cts 'webgpu:api,operation,command_buffer,basic:*'
//! ```
//!
//! You can also supply your own test list in a file:
//!
//! ```sh
//! cargo xtask cts -f your_tests.lst
//! ```
//!
//! Each line in a test list file is a test selector that will be passed to the
//! CTS's own command line runner. Note that wildcards may only be used to specify
//! running all tests in a file, or all subtests in a test.
//!
//! A test line may optionally contain a `fails-if(backend)` clause. This
//! indicates that the test should be skipped on that backend, however, the
//! runner will only do so if the `--backend` flag is passed to tell it where
//! it is running.
//!
//! Lines starting with `//` or `#` in the test list are treated as comments and
//! ignored.

use anyhow::{bail, Context};
use pico_args::Arguments;
use regex_lite::{Regex, RegexBuilder};
use std::{ffi::OsString, sync::LazyLock};
use xshell::Shell;

use crate::util::git_version_at_least;

/// Path within the repository where the CTS will be checked out.
const CTS_CHECKOUT_PATH: &str = "cts";

/// Path within the repository to a file containing the git revision of the CTS to check out.
const CTS_REVISION_PATH: &str = "cts_runner/revision.txt";

/// URL of the CTS git repository.
const CTS_GIT_URL: &str = "https://github.com/gpuweb/cts.git";

/// Path to default CTS test list.
const CTS_DEFAULT_TEST_LIST: &str = "cts_runner/test.lst";

#[derive(Default)]
struct TestLine {
    pub selector: OsString,
    pub fails_if: Vec<String>,
}

pub fn run_cts(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
) -> anyhow::Result<()> {
    let skip_checkout = args.contains("--skip-checkout");
    let llvm_cov = args.contains("--llvm-cov");
    let release = args.contains("--release");
    let mut quiet = args.contains("--quiet");
    let verbose = args.contains("--verbose");
    let running_on_backend = args.opt_value_from_str::<_, String>("--backend")?;
    let mut filter_pattern = args.opt_value_from_str::<_, String>("--filter")?;
    let mut filter_invert = false;

    if let Some(filter) = filter_pattern.as_deref() {
        if let Some(filter) = filter.strip_prefix('!') {
            filter_pattern = Some(filter.to_owned());
            filter_invert = true;
        }
    }

    // Compile filter regex early to fail fast on invalid patterns
    let filter = if let Some(pattern) = filter_pattern {
        Some(
            Regex::new(&pattern)
                .context(format!("Invalid regex pattern '{pattern}' for --filter"))?,
        )
    } else {
        None
    };

    if running_on_backend.is_none() {
        log::warn!(
            "fails-if conditions are only evaluated if a backend is specified with --backend"
        );
    }

    let mut list_files = Vec::<OsString>::new();
    while let Some(file) = args.opt_value_from_str("-f")? {
        list_files.push(file);
    }

    let mut tests = args
        .finish()
        .into_iter()
        .map(|selector| TestLine {
            selector,
            ..Default::default()
        })
        .collect::<Vec<_>>();

    if tests.is_empty() && list_files.is_empty() {
        if passthrough_args.is_none() {
            log::info!("Reading default test list from {CTS_DEFAULT_TEST_LIST}");
            list_files.push(OsString::from(CTS_DEFAULT_TEST_LIST));

            // Reduce output, unless `--verbose` was specified.
            quiet = !verbose;
        }
    } else if passthrough_args.is_some() {
        bail!("Test(s) and test list(s) are incompatible with passthrough arguments.");
    }

    for file in list_files {
        tests.extend(shell.read_file(file)?.lines().filter_map(|line| {
            static TEST_LINE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
                RegexBuilder::new(
                    r#"(?:fails-if\s*\(\s*(?<fails_if>\w+(?:,\w+)*?)\s*\)\s+)?(?<selector>.*)"#,
                )
                .build()
                .unwrap()
            });

            let trimmed = line.trim();
            let is_comment = trimmed.starts_with("//") || trimmed.starts_with("#");
            let captures = TEST_LINE_REGEX
                .captures(trimmed)
                .expect("Invalid test line: {trimmed}");
            (!trimmed.is_empty() && !is_comment).then(|| TestLine {
                selector: OsString::from(&captures["selector"]),
                fails_if: captures
                    .name("fails_if")
                    .map(|m| {
                        m.as_str()
                            .split_terminator(',')
                            .map(|m| m.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        }))
    }

    // Apply filter if specified
    if let Some(ref filter) = filter {
        let original_count = tests.len();
        tests.retain(|test| {
            let selector_str = test.selector.to_string_lossy();
            let matched = filter.is_match(&selector_str);
            if filter_invert {
                !matched
            } else {
                matched
            }
        });
        let filtered_count = tests.len();
        if filtered_count == original_count {
            log::warn!("Filter did not exclude any tests");
        } else if filtered_count != 0 {
            log::info!(
                "Filter selected {filtered_count} of {original_count} test{}",
                if original_count == 1 { "" } else { "s" },
            );
        } else if filtered_count == 0 {
            bail!("Filter did not select any tests");
        } else {
            bail!("Filtering introduced additional tests??");
        }
    }

    let wgpu_cargo_toml = std::path::absolute(shell.current_dir().join("Cargo.toml"))
        .context("Failed to get path to Cargo.toml")?;

    let cts_revision = shell
        .read_file(CTS_REVISION_PATH)
        .context(format!(
            "Failed to read CTS git SHA from {CTS_REVISION_PATH}"
        ))?
        .trim()
        .to_string();

    if !shell.path_exists(CTS_CHECKOUT_PATH) {
        if skip_checkout {
            bail!("Skipping CTS checkout doesn't make sense when CTS is not present");
        }
        let mut cmd = shell
            .cmd("git")
            .args(["clone", CTS_GIT_URL, CTS_CHECKOUT_PATH])
            .quiet();

        if git_version_at_least(&shell, [2, 49, 0])? {
            log::info!("Cloning CTS shallowly with revision {cts_revision}");
            cmd = cmd.args(["--depth=1", "--revision", &cts_revision]);
            cmd = cmd.args([
                "-c",
                "remote.origin.fetch=+refs/heads/gh-pages:refs/remotes/origin/gh-pages",
            ]);
        } else {
            log::info!("Cloning full checkout of CTS with revision {cts_revision}");
            cmd = cmd.args(["-b", "gh-pages", "--single-branch"]);
        }

        cmd.run().context("Failed to clone CTS")?;

        shell.change_dir(CTS_CHECKOUT_PATH);
    } else if !skip_checkout {
        shell.change_dir(CTS_CHECKOUT_PATH);

        // For new clones, this is set by the cloning commands above, but older
        // clones may not have it. Eventually this can be removed.
        if shell
            .cmd("git")
            .args(["config", "--get", "remote.origin.fetch"])
            .quiet()
            .ignore_stdout()
            .ignore_stderr()
            .run()
            .is_err()
        {
            shell
                .cmd("git")
                .args([
                    "config",
                    "remote.origin.fetch",
                    "+refs/heads/gh-pages:refs/remotes/origin/gh-pages",
                ])
                .quiet()
                .run()
                .context("Failed setting git config")?;
        }

        // If we don't have the CTS commit we want, try to fetch it.
        if shell
            .cmd("git")
            .args(["cat-file", "commit", &cts_revision])
            .quiet()
            .ignore_stdout()
            .ignore_stderr()
            .run()
            .is_err()
        {
            log::info!("Fetching CTS");
            shell
                .cmd("git")
                .args(["fetch", "--quiet"])
                .quiet()
                .run()
                .context("Failed to fetch CTS")?;
        }
    } else {
        shell.change_dir(CTS_CHECKOUT_PATH);
    }

    if !skip_checkout {
        log::info!("Checking out CTS");
        shell
            .cmd("git")
            .args(["checkout", "--quiet", &cts_revision])
            .quiet()
            .run()
            .context("Failed to check out CTS")?;
    } else {
        log::info!("Skipping CTS checkout because --skip-checkout was specified");
    }

    let run_flags = if llvm_cov {
        &["llvm-cov", "--no-cfg-coverage", "--no-report", "run"][..]
    } else {
        &["run"][..]
    };

    if let Some(passthrough_args) = passthrough_args {
        let mut cmd = shell
            .cmd("cargo")
            .args(run_flags)
            .args(["--manifest-path".as_ref(), wgpu_cargo_toml.as_os_str()])
            .args(["-p", "cts_runner"])
            .args(["--bin", "cts_runner"]);

        if release {
            cmd = cmd.arg("--release")
        }

        cmd.args(["--", "./tools/run_deno", "--verbose"])
            .args(&passthrough_args)
            .run()?;

        return Ok(());
    }

    log::info!("Running CTS");
    for test in &tests {
        if let Some(running_on_backend) = &running_on_backend {
            if test.fails_if.contains(running_on_backend) {
                log::info!(
                    "Skipping {} on {} backend",
                    test.selector.to_string_lossy(),
                    running_on_backend,
                );
                continue;
            }
        }

        if !quiet {
            log::info!("Running {}", test.selector.to_string_lossy());
        }

        let mut cmd = shell
            .cmd("cargo")
            .args(run_flags)
            .args(["--manifest-path".as_ref(), wgpu_cargo_toml.as_os_str()])
            .args(["-p", "cts_runner"])
            .args(["--bin", "cts_runner"]);

        if release {
            cmd = cmd.arg("--release")
        }

        cmd = cmd
            .args(["--", "./tools/run_deno", "--verbose"])
            .args([&test.selector]);

        if quiet {
            let output = cmd.ignore_status().output().context("Failed to run CTS")?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                if let Some((_, summary)) = stdout.split_once("** Summary **") {
                    println!("\n== Summary for {} ==", test.selector.to_string_lossy());
                    println!("{}", summary.trim());
                } else {
                    print!("{}", stdout);
                    eprint!("{}", stderr);
                }
            } else {
                print!("{}", stdout);
                eprint!("{}", stderr);
                bail!("CTS failed");
            }
        } else {
            cmd.run().context("CTS failed")?;
        }
    }

    if tests.len() > 1 {
        log::info!("Summary reflects only tests from the last selector, not the entire run.");
    }

    Ok(())
}
