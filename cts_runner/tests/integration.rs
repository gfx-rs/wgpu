// Tests for cts_runner
//
// As of June 2025, these tests are not run in CI.

use std::{
    fmt::{self, Debug, Display},
    path::PathBuf,
    process::Command,
    str,
};

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

pub struct JsError;

impl Display for JsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JavaScript test returned an error")
    }
}

impl Debug for JsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

type JsResult = Result<(), JsError>;

fn exec_js_test(script: &str) -> JsResult {
    let output = Command::new(cts_runner_exe_path())
        .arg(script)
        .output()
        .unwrap();
    println!("{}", str::from_utf8(&output.stdout).unwrap());
    eprintln!("{}", str::from_utf8(&output.stderr).unwrap());
    output.status.success().then_some(()).ok_or(JsError)
}

#[test]
fn hello_compute_example() -> JsResult {
    exec_js_test("examples/hello-compute.js")
}

#[test]
fn features() -> JsResult {
    exec_js_test("tests/features.js")
}
