#![cfg(not(target_arch = "wasm32"))]

use ureq::{http::Response, Agent, Body};

use crate::{native::MainResult, params::TestInfo, report::GpuReport, GpuTestInitializer};

// Called when tests are run for WASM in a browser. Kicks off each
// test by calling the test server which runs the test using playwright.
pub fn run_wasm_browser_tests(tests: Vec<GpuTestInitializer>) -> MainResult {
    let report: GpuReport = serde_json::from_str(
        &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../.wasmgpuconfig"))
            .expect("Failed to read .wasmgpuconfig"),
    )
    .expect("Failed to parse .wasmgpuconfig");

    let trials = tests
        .into_iter()
        .map(|test| {
            let test = test();
            let test_info = TestInfo::from_configuration(&test, &report.devices[0]);
            let backend = report.devices[0].info.backend;

            let full_name = format!(
                "[wasm] [{running_msg}] [{backend:?}] {base_name}",
                running_msg = test_info.running_msg,
                base_name = test.name,
            );

            libtest_mimic::Trial::test(&full_name, move || {
                let agent: Agent = Agent::config_builder()
                    .http_status_as_error(false)
                    .build()
                    .into();

                let response = agent
                    .get("http://127.0.0.1:3000/run_test")
                    .query("wasm", std::env::var("CARGO_PKG_NAME")?)
                    .query("name", &test.name)
                    .call();

                handle_response(response)
            })
        })
        .collect::<Vec<_>>();

    let args = libtest_mimic::Arguments::from_args();
    libtest_mimic::run(&args, trials).exit_if_failed();

    Ok(())
}

fn handle_response(
    response: Result<Response<Body>, ureq::Error>,
) -> Result<(), libtest_mimic::Failed> {
    let mut response = response?;
    if response.status() != 200 {
        Err(response.body_mut().read_to_string()?.into())
    } else {
        Ok(())
    }
}
