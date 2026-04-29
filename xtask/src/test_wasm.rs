use std::{ffi::OsString, process::Child, thread::sleep, time::Duration};

use anyhow::{bail, Context};
use pico_args::Arguments;
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

fn parse_wasm_list(list_output: String) -> Option<String> {
    let list_output: serde_json::Value = serde_json::from_str(&list_output).ok()?;

    let map = list_output.get("rust-binaries")?.as_object();

    Some(
        map.iter()
            .next()?
            .iter()
            .next()?
            .1
            .get("binary-path")?
            .as_str()?
            .to_string(),
    )
}

pub fn run_wasm_tests(
    shell: Shell,
    mut args: Arguments,
    passthrough_args: Option<Vec<OsString>>,
) -> anyhow::Result<()> {
    let runner_shell = Shell::new()?;
    runner_shell.change_dir(shell.current_dir().join("tests/wasm/runner"));
    runner_shell.cmd("npm").arg("install").run()?;
    runner_shell
        .cmd("npx")
        .args(["playwright", "install", "chromium", "--with-deps"])
        .run()?;

    shell.remove_path("tests/wasm/dist")?;
    shell.create_dir("tests/wasm/dist")?;

    for file in shell.read_dir("tests/wasm/web")? {
        shell.copy_file(file, "tests/wasm/dist/")?;
    }

    let show = args.contains("--show");
    let debug = args.contains("--debug");

    let cargo_args = flatten_args(args, passthrough_args);

    let bins = [
        Bin {
            name: "wgpu-test",
            build_args: vec!["-p", "wgpu-test", "--test", "wgpu-gpu"],
        },
        Bin {
            name: "wgpu-examples",
            build_args: vec!["-p", "wgpu-examples"],
        },
    ];

    for Bin { name, build_args } in bins.iter() {
        let build_output = shell
            .cmd("cargo")
            .args(["nextest", "list"])
            .args(build_args)
            .args([
                "--list-type",
                "binaries-only",
                "--target",
                "wasm32-unknown-unknown",
                "--features",
                "webgl,exhaust",
                "--message-format=json",
                "-v",
            ])
            .env("RUSTFLAGS", "--cfg wasm_test")
            .output()?;

        let list_output = String::from_utf8(build_output.stdout)?;
        let executable_path = parse_wasm_list(list_output);

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
            .current_dir(shell.current_dir())
            .arg("tests/wasm/runner/index.js")
            .args(if show { vec![] } else { vec!["--headless"] })
            .spawn()
            .context("Failed to start wasm test server")?,
    );

    loop {
        if ureq::get("http://127.0.0.1:3000/").call().is_ok() {
            break;
        };

        sleep(Duration::from_millis(100));
    }

    if debug {
        let _ = server.0.wait();
    } else {
        let mut response = ureq::get("http://127.0.0.1:3000/gpu_report")
            .query("wasm", bins[0].name)
            .call()
            .context("Failed to get gpu config from browser")?;

        let gpu_config = response
            .body_mut()
            .read_to_string()
            .context("Failed to get gpu config from browser")?;

        std::fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../.wasmgpuconfig"),
            gpu_config,
        )
        .context("Failed to write wasm gpu_config")?;

        shell
            .cmd("cargo")
            .args(["nextest", "run", "-P", "wasm", "--test-threads", "5"])
            .args(cargo_args)
            .env("RUSTFLAGS", "--cfg wasm_test")
            .run()?;
    }
    Ok(())
}
