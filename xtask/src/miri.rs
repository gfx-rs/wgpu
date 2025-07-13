use pico_args::Arguments;
use xshell::Shell;

pub fn run_miri(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    let toolchain: String = args
        .opt_value_from_str("--toolchain")?
        .unwrap_or_else(|| String::from("nightly"));

    shell
        .cmd("rustup")
        .args([
            "run",
            &toolchain,
            "cargo",
            "miri",
            "nextest",
            "run",
            "--target",
            "x86_64-unknown-linux-gnu",
        ])
        .env(
            "MIRIFLAGS",
            "-Zmiri-disable-isolation -Zmiri-deterministic-floats",
        )
        .env("WGPU_GPU_TESTS_USE_NOOP_BACKEND", "1")
        .quiet()
        .run()?;

    Ok(())
}
