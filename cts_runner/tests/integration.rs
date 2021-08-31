use std::path::PathBuf;

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

#[test]
fn hello_compute_example() {
    let output = std::process::Command::new(cts_runner_exe_path())
        .arg("examples/hello-compute.js")
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
    assert!(output.status.success())
}
