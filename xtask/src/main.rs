use cli::Args;

mod cli;
mod run_wasm;
mod test;
mod util;

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .format_indent(Some(0))
        .init();

    let args = Args::parse();

    run(args)
}

#[allow(unused_mut, unreachable_patterns)]
fn run(mut args: Args) -> anyhow::Result<()> {
    match args.subcommand {
        cli::Subcommand::RunWasm => run_wasm::run_wasm(args.command_args),
        cli::Subcommand::Test => test::run_tests(args.command_args),
        _ => unreachable!(),
    }
}
