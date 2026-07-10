mod cli;
mod core;
mod error;
mod params;

use clap::Parser as _;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = cli::Args::parse();

    if let Err(e) = real_main(&args) {
        error::print_err(e.as_ref());
        std::process::exit(1);
    }
}

fn real_main(args: &cli::Args) -> anyhow::Result<()> {
    let mut params = params::build_parameters(args)?;
    core::run(args, &mut params)
}
