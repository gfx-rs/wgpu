use std::{ffi::OsString, process::Child, thread::sleep, time::Duration};

use anyhow::{bail, Context};
use pico_args::Arguments;
use serde_json::Value;
use xshell::Shell;

use crate::util::flatten_args;

struct WasmTestServer(Child);

impl Drop for WasmTestServer {
    // Clean up playwright node processes when parent process ends
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

pub struct Bin<'a> {
    pub name: &'a str,
    pub build_args: Vec<&'a str>,
}

pub fn run_wasm_tests(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
) -> anyhow::Result<()> {
    shell.create_dir("tests/wasm/dist")?;

    for file in shell.read_dir("tests/wasm/web")? {
        shell.copy_file(file, "tests/wasm/dist/")?;
    }

    let headless = args.contains("--headless");
    let debug = args.contains("--debug");

    let cargo_args = flatten_args(args, passthrough_args);

    let bins = [
        Bin {
            name: "wgpu-test",
            build_args: vec![
                "test",
                "-p",
                "wgpu-test",
                "--no-run",
                "--target",
                "wasm32-unknown-unknown",
                "--features",
                "webgl",
                "--test",
                "wgpu-gpu",
            ],
        },
        Bin {
            name: "wgpu-examples",
            build_args: vec![
                "test",
                "-p",
                "wgpu-test",
                "--no-run",
                "--target",
                "wasm32-unknown-unknown",
                "--features",
                "webgl",
                "--test",
                "wgpu-gpu",
            ],
        },
    ];

    for Bin { name, build_args } in bins.iter() {
        let cmd = shell.cmd("cargo").args(build_args);
        cmd.run()?;

        let build_output = shell
            .cmd("cargo")
            .args(build_args)
            .args(["--message-format=json", "-q"])
            .output()?;

        let build_output = String::from_utf8(build_output.stdout)?;

        let mut executable_path = None;
        for line in build_output.lines() {
            let line: serde_json::Value =
                serde_json::from_str(line).context("Failed to parse wasm test build output")?;

            if let Some(reason) = line.get("reason") {
                if reason.as_str() == Some("compiler-artifact") {
                    if let Some(Value::String(executable)) = line.get("executable").cloned() {
                        if executable.ends_with(".wasm") {
                            executable_path = Some(executable);
                        }
                    }
                }
            }
        }

        let Some(executable_path) = executable_path else {
            bail!("Failed to find wasm test binary location");
        };

        shell
            .cmd("wasm-bindgen")
            .args([
                executable_path.as_str(),
                "--out-dir",
                "tests/wasm/dist",
                "--out-name",
                name,
                "--target",
                "web",
            ])
            .run()?;
    }

    let mut server = WasmTestServer(
        std::process::Command::new("node")
            .arg("tests/wasm/runner/index.js")
            .args(if headless { vec!["--headless"] } else { vec![] })
            .spawn()
            .expect("Failed to start wasm test server"),
    );

    loop {
        if ureq::get("http://127.0.0.1:3000/").call().is_ok() {
            break;
        };

        sleep(Duration::from_millis(100));
    }

    let mut response = ureq::get("http://127.0.0.1:3000/gpu_report")
        .query("wasm", bins[0].name)
        .call()
        .expect("Failed to get gpu config from browser");

    let gpu_config = response
        .body_mut()
        .read_to_string()
        .expect("Failed to get gpu config from browser");

    std::fs::write(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../.wasmgpuconfig"),
        gpu_config,
    )
    .expect("Failed to write wasm gpu_config");

    if debug {
        let _ = server.0.wait();
    } else {
        shell
            .cmd("cargo")
            .args(["nextest", "run", "-P", "wasm", "--test-threads", "1"])
            .args(cargo_args)
            .env("TEST_WASM", "true")
            .run()?;
    }
    Ok(())
}
