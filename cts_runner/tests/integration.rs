// Tests for cts_runner
//
// As of June 2025, these tests are not run in CI.

use std::{
    ffi::OsStr,
    io::Write,
    path::PathBuf,
    process::{Command, Output},
    str,
};

use tempfile::NamedTempFile;

pub fn target_dir() -> PathBuf {
    let current_exe = std::env::current_exe().unwrap();
    let target_dir = current_exe.parent().unwrap().parent().unwrap();
    target_dir.into()
}

pub fn cts_runner_exe_path() -> PathBuf {
    // Something like /Users/lucacasonato/src/wgpu/target/debug/cts_runner
    let mut p = target_dir().join("cts_runner");
    if cfg!(windows) {
        p.set_extension("exe");
    }
    p
}

fn exec_cts_runner(script_file: impl AsRef<OsStr>) -> Output {
    Command::new(cts_runner_exe_path())
        .arg(script_file)
        .output()
        .unwrap()
}

fn exec_js_file(script_file: impl AsRef<OsStr>) {
    let output = exec_cts_runner(script_file);
    println!("{}", str::from_utf8(&output.stdout).unwrap());
    eprintln!("{}", str::from_utf8(&output.stderr).unwrap());
    assert!(output.status.success());
}

fn check_js_stderr(script: &str, expected: &str) {
    let mut tempfile = NamedTempFile::new().unwrap();
    tempfile.write_all(script.as_bytes()).unwrap();
    tempfile.flush().unwrap();
    let output = exec_cts_runner(tempfile.path());
    assert!(
        output.stdout.is_empty(),
        "unexpected output on stdout: {}",
        str::from_utf8(&output.stdout).unwrap(),
    );
    assert_eq!(str::from_utf8(&output.stderr).unwrap(), expected);
    assert!(output.status.success());
}

fn exec_js(script: &str) {
    check_js_stderr(script, "");
}

#[test]
fn hello_compute_example() {
    exec_js_file("examples/hello-compute.js");
}

#[test]
fn features() {
    exec_js(
        r#"
        const adapter = await navigator.gpu.requestAdapter();

        if (adapter.features.has("mappable-primary-buffers")) {
            throw new TypeError("Adapter should not report support for wgpu native-only features");
        }
    "#,
    );
}

#[test]
fn uncaptured_error() {
    check_js_stderr(
        r#"
            const code = `const val: u32 = 1.1;`;

            const adapter = await navigator.gpu.requestAdapter();
            const device = await adapter.requestDevice();
            device.createShaderModule({ code })
        "#,
        "cts_runner caught WebGPU error:
Shader '' parsing error: the type of `val` is expected to be `u32`, but got `{AbstractFloat}`
  ┌─ wgsl:1:7
  │
1 │ const val: u32 = 1.1;
  │       ^^^ definition of `val`\n\n",
    );
}
