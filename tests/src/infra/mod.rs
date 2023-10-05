#[cfg(not(target_arch = "wasm32"))]
use parking_lot::Mutex;

pub use params::{GpuTestConfiguration, RunTestAsync};
pub use report::AdapterReport;
pub use single::{SingleTest, TestInfo};

mod params;
mod report;
mod single;

#[cfg(not(target_arch = "wasm32"))]
pub static TEST_LIST: Mutex<Vec<GpuTestConfiguration>> = Mutex::new(Vec::new());

pub type MainResult = anyhow::Result<()>;

#[cfg(not(target_arch = "wasm32"))]
pub fn main() -> MainResult {
    use anyhow::Context;

    let config_text =
        &std::fs::read_to_string(format!("{}/../.gpuconfig", env!("CARGO_MANIFEST_DIR")))
            .context("Failed to read .gpuconfig, did you run the tests via `cargo xtask test`?")?;
    let report =
        report::GpuReport::from_json(config_text).context("Could not pare .gpuconfig JSON")?;

    let mut test_guard = TEST_LIST.lock();
    execute_native(test_guard.drain(..).flat_map(|test| {
        report
            .devices
            .iter()
            .enumerate()
            .map(move |(adapter_index, adapter)| {
                SingleTest::from_configuration(test.clone(), adapter, adapter_index)
            })
    }));

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_native(tests: impl IntoIterator<Item = SingleTest>) {
    let args = libtest_mimic::Arguments::from_args();
    let trials = tests.into_iter().map(SingleTest::into_trial).collect();

    libtest_mimic::run(&args, trials).exit_if_failed();
}
