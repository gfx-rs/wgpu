#![cfg_attr(target_arch = "wasm32", no_main)]
#![cfg(not(target_arch = "wasm32"))]

#[cfg(test)]
use std::sync::Mutex;

extern crate wgpu_c_backend;

mod cli;
mod human;
mod report;
#[cfg(test)]
mod tests;
mod texture;

fn main() -> anyhow::Result<()> {
    cli::main()
}

/// Some tests modify the `WGPU_NO_CUSTOM_BACKEND` environment variable.
/// This is checked every time a new instance is created anywhere.
/// Modifying env vars on one thread while reading them on another is UB.
/// Additionally, we don't want other threads to be affected unpredictably
/// by this changing around.
#[cfg(test)]
pub(crate) static INSTANCE_MUTEX: Mutex<()> = Mutex::new(());
