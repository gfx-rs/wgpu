#[cfg(not(target_arch = "wasm32"))]
mod cli;
#[cfg(not(target_arch = "wasm32"))]
mod human;
#[cfg(not(target_arch = "wasm32"))]
mod report;
#[cfg(not(target_arch = "wasm32"))]
mod texture;

fn main() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        cli::main()?;
    }
    Ok(())
}
