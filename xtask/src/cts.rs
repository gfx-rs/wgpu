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

/// Path within the repository where the CTS will be checked out.
const CTS_CHECKOUT_PATH: &str = "cts";

/// Path within the repository to a file containing the git revision of the CTS to check out.
const CTS_REVISION_PATH: &str = "cts_runner/revision.txt";

/// URL of the CTS git repository.
const CTS_GIT_URL: &str = "https://github.com/gpuweb/cts.git";

/// Path to default CTS test list.
const CTS_DEFAULT_TEST_LIST: &str = "cts_runner/test.lst";

static TEST_LINE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(r#"(?:fails-if\s*\(\s*(?<fails_if>\w+)\s*\)\s+)?(?<selector>.*)"#)
        .build()
        .unwrap()
});

#[derive(Default)]
struct TestLine {
    pub selector: OsString,
    pub fails_if: Option<String>,
}

pub fn run_cts(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    let skip_checkout = args.contains("--skip-checkout");
    let llvm_cov = args.contains("--llvm-cov");
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
        log::info!("Reading default test list from {CTS_DEFAULT_TEST_LIST}");
        list_files.push(OsString::from(CTS_DEFAULT_TEST_LIST));
    }

    for file in list_files {
        tests.extend(shell.read_file(file)?.lines().filter_map(|line| {
            let trimmed = line.trim();
            let is_comment = trimmed.starts_with("//") || trimmed.starts_with("#");
            let captures = TEST_LINE_REGEX
                .captures(trimmed)
                .expect("Invalid test line: {trimmed}");
            (!trimmed.is_empty() && !is_comment).then(|| TestLine {
                selector: OsString::from(&captures["selector"]),
                fails_if: captures.name("fails_if").map(|m| m.as_str().to_string()),
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

    log::info!("Running CTS");
    for test in &tests {
        match (&test.fails_if, &running_on_backend) {
            (Some(backend), Some(running_on_backend)) if backend == running_on_backend => {
                log::info!(
                    "Skipping {} on {} backend",
                    test.selector.to_string_lossy(),
                    running_on_backend,
                );
                continue;
            }
            _ => {}
        }

        log::info!("Running {}", test.selector.to_string_lossy());
        shell
            .cmd("cargo")
            .args(run_flags)
            .args(["--manifest-path".as_ref(), wgpu_cargo_toml.as_os_str()])
            .args(["-p", "cts_runner"])
            .args(["--bin", "cts_runner"])
            .args(["--", "./tools/run_deno", "--verbose"])
            .args([&test.selector])
            .run()
            .context("CTS failed")?;
    }

    if tests.len() > 1 {
        log::info!("Summary reflects only tests from the last selector, not the entire run.");
    }

    Ok(())
}

fn git_version_at_least(shell: &Shell, version: GitVersion) -> anyhow::Result<bool> {
    let output = shell
        .cmd("git")
        .args(["--version"])
        .output()
        .context("Failed to invoke `git --version`")?;

    let Some(code) = output.status.code() else {
        anyhow::bail!("`git --version` failed to return an exit code; interrupt via signal, maybe?")
    };

    anyhow::ensure!(code == 0, "`git --version` returned a nonzero exit code");

    let fmt_err_msg = "`git --version` did not have the expected structure";

    let stdout = String::from_utf8(output.stdout).expect(fmt_err_msg);

    let parsed = parse_git_version_output(&stdout).expect(fmt_err_msg);

    Ok(parsed >= version)
}

pub type GitVersion = [u8; 3];

fn parse_git_version_output(output: &str) -> anyhow::Result<GitVersion> {
    const PREFIX: &str = "git version ";

    let raw_version = output
        .strip_prefix(PREFIX)
        .with_context(|| format!("missing `{PREFIX}` prefix"))?;

    let raw_version = raw_version.trim_end(); // There should always be a newline at the end, but
                                              // we don't care if it's missing.

    // Git for Windows suffixes the version with ".windows.<n>".
    // Strip it if present.
    let raw_version = raw_version
        .split_once(".windows")
        .map_or(raw_version, |(before, _after)| before);

    let parsed = GitVersion::try_from(
        raw_version
            .splitn(3, '.')
            .enumerate()
            .map(|(idx, s)| {
                s.parse().with_context(|| {
                    format!("failed to parse version number {idx} ({s:?}) as `u8`")
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(|vec| anyhow::Error::msg(format!("less than 3 version numbers found: {vec:?}")))?;

    log::debug!("detected Git version {raw_version}");

    Ok(parsed)
}

#[test]
fn test_git_version_parsing() {
    macro_rules! test_ok {
        ($input:expr, $expected:expr) => {
            assert_eq!(parse_git_version_output($input).unwrap(), $expected);
        };
    }
    test_ok!("git version 2.3.0", [2, 3, 0]);
    test_ok!("git version 0.255.0", [0, 255, 0]);
    test_ok!("git version 4.5.6", [4, 5, 6]);
    test_ok!("git version 2.3.0.windows.1", [2, 3, 0]);

    macro_rules! test_err {
        ($input:expr, $msg:expr) => {
            assert_eq!(
                parse_git_version_output($input).unwrap_err().to_string(),
                $msg
            )
        };
    }
    test_err!("2.3.0", "missing `git version ` prefix");
    test_err!("", "missing `git version ` prefix");

    test_err!(
        "git version 1.2",
        "less than 3 version numbers found: [1, 2]"
    );

    test_err!(
        "git version 9001",
        "failed to parse version number 0 (\"9001\") as `u8`"
    );
    test_err!(
        "git version ",
        "failed to parse version number 0 (\"\") as `u8`"
    );
    test_err!(
        "git version asdf",
        "failed to parse version number 0 (\"asdf\") as `u8`"
    );
    test_err!(
        "git version 23.beta",
        "failed to parse version number 1 (\"beta\") as `u8`"
    );
    test_err!(
        "git version 1.2.wat",
        "failed to parse version number 2 (\"wat\") as `u8`"
    );
    test_err!(
        "git version 1.2.3.",
        "failed to parse version number 2 (\"3.\") as `u8`"
    );
    test_err!(
        "git version 1.2.3.4",
        "failed to parse version number 2 (\"3.4\") as `u8`"
    );
}
