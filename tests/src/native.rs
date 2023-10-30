#![cfg(not(target_arch = "wasm32"))]
//! Infrastructure for the native, `cargo-nextest` based harness.
//!
//! This is largly used by [`gpu_test_main`](crate::gpu_test_main) and [`gpu_test`](crate::gpu_test).

use std::{future::Future, pin::Pin};

use anyhow::Context;
use parking_lot::Mutex;

use crate::{
    config::GpuTestConfiguration,
    params::TestInfo,
    report::{AdapterReport, GpuReport},
    run::execute_test,
    AdapterSettings,
};

type NativeTestFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

struct NativeTest {
    name: String,
    future: NativeTestFuture,
}

impl NativeTest {
    fn from_configuration(
        config: GpuTestConfiguration,
        adapter: &AdapterReport,
        settings: AdapterSettings,
    ) -> Self {
        let backend = adapter.info.backend;
        let device_name = &adapter.info.name;

        let test_info = TestInfo::from_configuration(&config, adapter);

        let full_name = format!(
            "[{running_msg}] [{backend:?}/{device_name}/{adapter_index}/{min_features}] {base_name}",
            running_msg = test_info.running_msg,
            base_name = config.name,
            adapter_index = settings.index,
            min_features = if settings.minimum_features { "Min Feat" } else { "Full Feat" },
        );
        Self {
            name: full_name,
            future: Box::pin(async move {
                // Enable metal validation layers if we're running on metal.
                //
                // This is a process-wide setting as it's via environment variable, but all
                // tests are run in separate processes.
                //
                // We don't do this in the instance initializer as we don't want to enable
                // validation layers for the entire process, or other instances.
                //
                // We do not enable metal validation when running on moltenvk.
                let metal_validation = backend == wgpu::Backend::Metal;

                let env_value = if metal_validation { "1" } else { "0" };
                std::env::set_var("MTL_DEBUG_LAYER", env_value);
                std::env::set_var("MTL_SHADER_VALIDATION", env_value);

                execute_test(config, Some(test_info), settings).await;
            }),
        }
    }

    pub fn into_trial(self) -> libtest_mimic::Trial {
        libtest_mimic::Trial::test(self.name, || {
            pollster::block_on(self.future);
            Ok(())
        })
    }
}

#[doc(hidden)]
pub static TEST_LIST: Mutex<Vec<crate::GpuTestConfiguration>> = Mutex::new(Vec::new());

/// Return value for the main function.
pub type MainResult = anyhow::Result<()>;

fn enumerate_adapters(report: &GpuReport) -> Vec<(&AdapterReport, AdapterSettings)> {
    let mut backends = wgpu::Backends::empty();

    let mut adapters = Vec::new();
    for (adapter_index, adapter) in report.devices.iter().enumerate() {
        let adapter_backend: wgpu::Backends = adapter.info.backend.into();
        // The first backend we encounter gets a minimum features.
        if !backends.contains(adapter_backend) {
            backends |= adapter_backend;
            adapters.push((
                adapter,
                AdapterSettings {
                    index: adapter_index,
                    minimum_features: true,
                },
            ));
        }

        adapters.push((
            adapter,
            AdapterSettings {
                index: adapter_index,
                minimum_features: false,
            },
        ));
    }

    adapters
}

/// Main function that runs every gpu function once for every adapter on the system.
pub fn main() -> MainResult {
    let config_text = {
        profiling::scope!("Reading .gpuconfig");
        &std::fs::read_to_string(format!("{}/../.gpuconfig", env!("CARGO_MANIFEST_DIR")))
            .context("Failed to read .gpuconfig, did you run the tests via `cargo xtask test`?")?
    };
    let report = GpuReport::from_json(config_text).context("Could not pare .gpuconfig JSON")?;

    let mut test_guard = TEST_LIST.lock();
    execute_native(test_guard.drain(..).flat_map(|test| {
        enumerate_adapters(&report)
            .into_iter()
            .map(move |(adapter, settings)| {
                NativeTest::from_configuration(test.clone(), adapter, settings)
            })
    }));

    Ok(())
}

fn execute_native(tests: impl IntoIterator<Item = NativeTest>) {
    let args = libtest_mimic::Arguments::from_args();
    let trials = {
        profiling::scope!("collecting tests");
        tests.into_iter().map(NativeTest::into_trial).collect()
    };

    libtest_mimic::run(&args, trials).exit_if_failed();
}
