use std::process::Command;

use anyhow::Context;
use cli::Args;

mod cli;
mod test;

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .format_indent(Some(0))
        .init();

    let args = Args::parse();

    if !args.subcommand.required_features_enabled() {
        let features = args.subcommand.features();
        log::info!(
            "Required features \"{features}\" are not enabled, recursing with features enabled"
        );

        let subcommand_args = args.command_args.finish();
        let iter = subcommand_args
            .iter()
            .map(|os| os.as_os_str().to_str().unwrap());

        let status = Command::new("cargo")
            .args(["xtask", "--features", features, args.subcommand.to_str()])
            .args(iter)
            .status()
            .context("Failed to execute recursive cargo xtask")?;

        if status.success() {
            return Ok(());
        } else {
            return Err(anyhow::anyhow!("subcommand failed"));
        }
    }

    run(args)
}

#[allow(unused_mut, unreachable_patterns)]
fn run(mut args: Args) -> anyhow::Result<()> {
    match args.subcommand {
        #[cfg(feature = "run-wasm")]
        cli::Subcommand::RunWasm => {
            log::info!("Running wasm example");
            // Use top-level Cargo.toml instead of xtask/Cargo.toml by default
            let manifest_path = args
                .command_args
                .value_from_str("--manifest-path")
                .unwrap_or_else(|_| "../Cargo.toml".to_string());
            let mut arg_vec = args.command_args.finish();
            arg_vec.push("--manifest-path".into());
            arg_vec.push(manifest_path.into());
            let args = pico_args::Arguments::from_vec(arg_vec);

            cargo_run_wasm::run_wasm_with_css_and_args(
                "body { margin: 0px; }",
                cargo_run_wasm::Args::from_args(args)
                    .map_err(anyhow::Error::msg)
                    .context("failed to parse arguments for `cargo-run-wasm`")?,
            );
            Ok(())
        }
        cli::Subcommand::Test => test::run_tests(args.command_args),
        _ => unreachable!(),
    }
}
