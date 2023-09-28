use anyhow::Context;

pub use params::{GpuTestConfiguration, RunTestAsync};
use parking_lot::Mutex;

use crate::infra::single::SingleTest;

mod params;
mod report;
mod single;

pub static TEST_LIST: Mutex<Vec<GpuTestConfiguration>> = Mutex::new(Vec::new());

pub type MainResult = anyhow::Result<()>;

pub fn main() -> MainResult {
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
                SingleTest::from_gpu_test(test.clone(), adapter, adapter_index)
            })
    }));

    Ok(())
}

fn execute_native<'a>(tests: impl IntoIterator<Item = SingleTest>) {
    let args = libtest_mimic::Arguments::from_args();
    let trials = tests.into_iter().map(SingleTest::into_trial).collect();

    libtest_mimic::run(&args, trials).exit_if_failed();
}
