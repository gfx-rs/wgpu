use anyhow::{bail, Context};
use pico_args::Arguments;
use std::ffi::OsString;
use xshell::Shell;

/// Path within the repository where the CTS will be checked out.
const CTS_CHECKOUT_PATH: &str = "cts";

/// Path within the repository to a file containing the git revision of the CTS to check out.
const CTS_REVISION_PATH: &str = "cts_runner/revision.txt";

/// URL of the CTS git repository.
const CTS_GIT_URL: &str = "https://github.com/gpuweb/cts.git";

/// Path to default CTS test list.
const CTS_DEFAULT_TEST_LIST: &str = "cts_runner/test.lst";

pub fn run_cts(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    let skip_checkout = args.contains("--skip-checkout");

    let mut list_files = Vec::<OsString>::new();
    while let Some(file) = args.opt_value_from_str("-f")? {
        list_files.push(file);
    }

    let mut tests = args.finish();

    if tests.is_empty() && list_files.is_empty() {
        log::info!("Reading default test list from {CTS_DEFAULT_TEST_LIST}");
        list_files.push(OsString::from(CTS_DEFAULT_TEST_LIST));
    }

    for file in list_files {
        tests.extend(shell.read_file(file)?.lines().filter_map(|line| {
            let trimmed = line.trim();
            let is_comment = trimmed.starts_with("//") || trimmed.starts_with("#");
            (!is_comment).then(|| OsString::from(trimmed))
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
        log::info!("Cloning CTS");
        shell
            .cmd("git")
            .args(["clone", CTS_GIT_URL, CTS_CHECKOUT_PATH])
            .quiet()
            .run()
            .context("Failed to clone CTS")?;

        shell.change_dir(CTS_CHECKOUT_PATH);
    } else if !skip_checkout {
        shell.change_dir(CTS_CHECKOUT_PATH);

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

    log::info!("Running CTS");
    for test in &tests {
        log::info!("Running {}", test.to_string_lossy());
        shell
            .cmd("cargo")
            .args(["run"])
            .args(["--manifest-path".as_ref(), wgpu_cargo_toml.as_os_str()])
            .args(["-p", "cts_runner"])
            .args(["--bin", "cts_runner"])
            .args(["--", "./tools/run_deno", "--verbose"])
            .args([test])
            .run()
            .context("CTS failed")?;
    }

    if tests.len() > 1 {
        log::info!("Summary reflects only tests from the last selector, not the entire run.");
    }

    Ok(())
}
