use std::{io, process::Command};

use anyhow::Context;
use xshell::Shell;

pub(crate) struct Program {
    pub crate_name: &'static str,
    pub binary_name: &'static str,
}

pub(crate) fn looks_like_git_sha(input: &str) -> bool {
    input.len() == 40 && input.chars().all(|c| c.is_ascii_hexdigit())
}

pub(crate) fn check_all_programs(programs: &[Program]) -> anyhow::Result<()> {
    let mut failed_crates = Vec::new();
    for &Program {
        crate_name,
        binary_name,
    } in programs
    {
        let mut cmd = Command::new(binary_name);
        cmd.arg("--help");
        let output = cmd.output();
        match output {
            Ok(_output) => {
                log::info!("Checking for {binary_name} in PATH: ✅");
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::NotFound) => {
                log::error!("Checking for {binary_name} in PATH: ❌");
                failed_crates.push(crate_name);
            }
            Err(e) => {
                log::error!("Checking for {binary_name} in PATH: ❌");
                panic!("Unknown IO error: {e:?}");
            }
        }
    }

    if !failed_crates.is_empty() {
        log::error!(
            "Please install them with: cargo install {}",
            failed_crates.join(" ")
        );

        anyhow::bail!("Missing required programs");
    }

    Ok(())
}

pub(crate) fn git_version_at_least(shell: &Shell, version: GitVersion) -> anyhow::Result<bool> {
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

pub(crate) type GitVersion = [u8; 3];

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
