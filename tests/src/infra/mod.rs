use std::sync::Arc;

use anyhow::Context;

pub use params::GpuTest;

mod params;
mod report;
mod single;

pub type MainResult = anyhow::Result<()>;

pub fn main<const TEST_COUNT: usize>(
    test_list: [Arc<dyn params::GpuTest + Send + Sync>; TEST_COUNT],
) -> MainResult {
    let args = libtest_mimic::Arguments::from_args();

    let config_text =
        &std::fs::read_to_string(format!("{}/../.gpuconfig", env!("CARGO_MANIFEST_DIR")))
            .context("failed to read .gpuconfig")?;
    let report =
        report::GpuReport::from_json(config_text).context("Could not pare .gpuconfig JSON")?;

    libtest_mimic::run(
        &args,
        test_list
            .into_iter()
            .flat_map(|test| {
                report
                    .devices
                    .iter()
                    .map(move |device| single::run_test(test.clone(), device))
            })
            .collect(),
    )
    .exit_if_failed();

    Ok(())
}
