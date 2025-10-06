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

use crate::util::{git_version_at_least, looks_like_git_sha};

/// Path within the repository where the CTS will be checked out.
const CTS_CHECKOUT_PATH: &str = "cts";

/// Path to git's `shallow` file.
const GIT_SHALLOW_PATH: &str = ".git/shallow";

/// Path within the repository to a file containing the git revision of the CTS to check out.
const CTS_REVISION_PATH: &str = "cts_runner/revision.txt";

/// URL of the CTS git repository.
const CTS_GITHUB_PATH: &str = "gpuweb/cts";

/// Path to default CTS test list.
const CTS_DEFAULT_TEST_LIST: &str = "cts_runner/test.lst";

#[derive(Default)]
struct TestLine {
    pub selector: OsString,
    pub fails_if: Vec<String>,
}

fn have_git_sha(shell: &Shell, sha: &str) -> bool {
    shell
        .cmd("git")
        .args(["cat-file", "commit", sha])
        .quiet()
        .ignore_stdout()
        .ignore_stderr()
        .run()
        .is_ok()
}

fn maybe_deepen_git_repo(shell: &Shell, desired: &str) -> anyhow::Result<()> {
    if shell
        .cmd("curl")
        .args([
            "-f",
            "-L",
            "-I",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
        ])
        .arg(format!(
            "https://api.github.com/repos/{CTS_GITHUB_PATH}/commits/{desired}"
        ))
        .quiet()
        .ignore_stdout()
        .ignore_stderr()
        .run()
        .is_err()
    {
        log::warn!("Not deepening repo because the desired CTS SHA was not found on GitHub.");
        return Ok(());
    }

    if !shell.path_exists(GIT_SHALLOW_PATH) {
        log::warn!("Not deepening repo because it is not a shallow clone.");
        return Ok(());
    }

    let shallow = shell
        .read_file(GIT_SHALLOW_PATH)
        .context(format!(
            "Failed to read git shallow SHA from {GIT_SHALLOW_PATH}"
        ))?
        .trim()
        .to_string();

    if !looks_like_git_sha(&shallow) {
        log::warn!(
            "Automatic deepening of git repo requires a shallow clone with a single graft point"
        );
        return Ok(());
    }

    let output = shell
        .cmd("curl")
        .args([
            "-f",
            "-L",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
        ])
        .arg(format!(
            "https://api.github.com/repos/{CTS_GITHUB_PATH}/compare/{desired}...{shallow}"
        ))
        .output()
        .context("Error calling GitHub API")?;

    let gh_json: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&output.stdout).context("Failed parsing GitHub API JSON")?;

    let Some(deepen_count) = gh_json
        .get("total_commits")
        .and_then(serde_json::Value::as_u64)
    else {
        bail!("missing or invalid total_commits");
    };

    log::info!("Fetching CTS with --deepen {deepen_count}");

    shell
        .cmd("git")
        .args(["fetch", "--deepen", &deepen_count.to_string()])
        .quiet()
        .run()
        .context("Failed to deepen git repo")?;

    Ok(())
}

pub fn run_cts(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
) -> anyhow::Result<()> {
    let skip_checkout = args.contains("--skip-checkout");
    let llvm_cov = args.contains("--llvm-cov");
    let release = args.contains("--release");
    let running_on_backend = args.opt_value_from_str::<_, String>("--backend")?;

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
            .args([
                "clone",
                &format!("https://github.com/{CTS_GITHUB_PATH}.git"),
                CTS_CHECKOUT_PATH,
            ])
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
        if !have_git_sha(&shell, &cts_revision) {
            log::info!("Desired SHA not found, fetching CTS");
            shell
                .cmd("git")
                .args(["fetch", "--quiet"])
                .quiet()
                .run()
                .context("Failed to fetch CTS")?;
        }

        // If we still don't have the commit we want, maybe we need more history.
        if !have_git_sha(&shell, &cts_revision) {
            log::info!("Desired SHA still not found, checking if missing from shallow clone");
            maybe_deepen_git_repo(&shell, &cts_revision)?;
        }

        if !have_git_sha(&shell, &cts_revision) {
            bail!("Unable to obtain the desired CTS revision {cts_revision}");
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

        log::info!("Running {}", test.selector.to_string_lossy());
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
            .args([&test.selector])
            .run()
            .context("CTS failed")?;
    }

    if tests.len() > 1 {
        log::info!("Summary reflects only tests from the last selector, not the entire run.");
    }

    Ok(())
}
