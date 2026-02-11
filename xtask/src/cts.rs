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

use anyhow::{anyhow, bail, Context};
use core::fmt;
use pico_args::Arguments;
use regex_lite::{Regex, RegexBuilder};
use std::{ffi::OsString, sync::LazyLock};
use xshell::Shell;

use crate::util::{git_version_at_least, parse_binary_from_cargo_json};

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

#[derive(Clone, Copy, Debug)]
enum PrintOutputWhen {
    TestFails,
    Always,
}

pub fn run_cts(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
) -> anyhow::Result<()> {
    let skip_checkout = args.contains("--skip-checkout");
    let llvm_cov = args.contains("--llvm-cov");
    let release = args.contains("--release");

    let output_filter = args
        .opt_value_from_str::<_, String>("--print-output-when")?
        .map(|f| {
            let values = [
                ("test-fails", PrintOutputWhen::TestFails),
                ("always", PrintOutputWhen::Always),
            ];
            let lowered = f.to_ascii_lowercase();
            values
                .iter()
                .find_map(|(cli_str, enum_value)| (&*lowered == *cli_str).then_some(*enum_value))
                .ok_or_else(|| {
                    anyhow!(
                        "`{f}` is not a valid `--print-output-when` value; expected one of {}",
                        fmt::from_fn(|f| {
                            f.debug_list()
                                .entries(values.iter().map(|(cli, _enum)| cli))
                                .finish()
                        })
                    )
                })
        })
        .transpose()?;

    let running_on_backend = args.opt_value_from_str::<_, String>("--backend")?;
    let enable_external_texture = args.contains("--enable-external-texture")
        || (!args.contains("--disable-external-texture")
            && running_on_backend
                .as_ref()
                .is_some_and(|b| ["metal", "dx12"].contains(&b.as_str())));

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

    if running_on_backend.is_none() && (!list_files.is_empty() || tests.is_empty()) {
        log::warn!("The `--backend` option was not provided. `fails-if` conditions and external");
        log::warn!("texture support are handled correctly only when a backend is specified.");
    }

    let mut default_output_filter = PrintOutputWhen::Always;

    if tests.is_empty() && list_files.is_empty() {
        if passthrough_args.is_none() {
            log::info!("Reading default test list from {CTS_DEFAULT_TEST_LIST}");
            list_files.push(OsString::from(CTS_DEFAULT_TEST_LIST));

            default_output_filter = PrintOutputWhen::TestFails;
        }
    } else if passthrough_args.is_some() {
        bail!("Test(s) and test list(s) are incompatible with passthrough arguments.");
    }

    let output_filter = output_filter.unwrap_or(default_output_filter);

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
        .context("Failed to get path to `Cargo.toml`")?;

    let cts_revision = shell
        .read_file(CTS_REVISION_PATH)
        .context(format!(
            "Failed to read CTS git SHA from `{CTS_REVISION_PATH}`"
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
                "-c",
                "advice.detachedHead=false",
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

    let mut cargo_opts: Vec<OsString> = vec![
        "--manifest-path".into(),
        wgpu_cargo_toml.into(),
        "-p".into(),
        "cts_runner".into(),
        "--bin".into(),
        "cts_runner".into(),
    ];
    if release {
        cargo_opts.push("--release".into());
    }

    let env_vars = if llvm_cov {
        // Typically coverage runs are done via cargo with `cargo llvm-cov run`. Running the
        // coverage-instrumented binary directly requires setting some environment variables. See
        // <https://github.com/taiki-e/cargo-llvm-cov/blob/main/README.md#get-coverage-of-external-tests>
        //
        // Unlike regular `llvm-cov run`, which builds artifacts in `target/llvm-cov-target`,
        // `llvm-cov show-env` uses the regular target directory for artifacts. Because of this,
        // the CTS job configures the `install-mesa` and `install-warp` actions with the regular
        // target directory, not the `llvm-cov-target` directory.
        let env = shell
            .cmd("cargo")
            .args(&["llvm-cov", "--no-cfg-coverage", "show-env"])
            .read()
            .context("Failed to get `llvm-cov` environment variables")?
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    None
                } else {
                    line.split_once('=')
                }
            })
            .map(|(key, value)| {
                let value = value.trim_matches('"').trim_matches('\'');
                (key.to_string(), value.to_string())
            })
            .collect::<Vec<_>>()
            .into_iter();

        // Avoid conflicts between coverage and non-coverage build artifacts.
        // This is recommended by the `cargo-llvm-cov` docs.
        shell
            .cmd("cargo")
            .envs(env.clone())
            .args(["llvm-cov", "clean", "--workspace"])
            .run()
            .context("Failed to run `llvm-cov clean`")?;

        env
    } else {
        vec![].into_iter()
    };

    let build_output = shell
        .cmd("cargo")
        .envs(env_vars.clone())
        .args(["build", "--message-format", "json-render-diagnostics"])
        .args(&cargo_opts)
        .read()
        .context("Failed to build `cts_runner`")?;

    let bin = parse_binary_from_cargo_json(&build_output)
        .context("Failed to identify executable from cargo build output")?;

    let cts_bin = &["./tools/run_deno", "--verbose"];

    if let Some(passthrough_args) = passthrough_args {
        return Ok(shell
            .cmd(bin)
            .envs(env_vars)
            .args(cts_bin)
            .args(enable_external_texture.then_some("--enable-external-texture"))
            .args(&passthrough_args)
            .run()?);
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

        if let PrintOutputWhen::Always = output_filter {
            log::info!("Running {}", test.selector.to_string_lossy());
        }

        let cmd = shell
            .cmd(&bin)
            .envs(env_vars.clone())
            .args(enable_external_texture.then_some("--enable-external-texture"))
            .args(cts_bin)
            .args([&test.selector]);

        match output_filter {
            PrintOutputWhen::TestFails => {
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
            }
            PrintOutputWhen::Always => {
                cmd.run().context("CTS failed")?;
            }
        }
    }

    if tests.len() > 1 {
        log::info!("Summary reflects only tests from the last selector, not the entire run.");
    }

    Ok(())
}
