#[cfg(not(target_arch = "wasm32"))]
mod cli;
#[cfg(not(target_arch = "wasm32"))]
mod human;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        cli::main();
    }
}
