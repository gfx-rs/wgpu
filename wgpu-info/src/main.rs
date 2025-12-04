#![cfg_attr(target_arch = "wasm32", no_main)]
#![cfg(not(target_arch = "wasm32"))]

mod cli;
mod human;
mod report;
#[cfg(test)]
mod tests;
mod texture;

fn main() -> anyhow::Result<()> {
    cli::main()
}
