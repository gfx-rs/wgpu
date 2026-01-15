use std::{
    ffi::OsStr,
    fs,
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

#[test]
fn lst_files_are_sorted() {
    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let files = ["test.lst", "fail.lst", "skip.lst"];

    for file in &files {
        let file_path = workspace_dir.join("cts_runner").join(file);
        let contents = fs::read_to_string(&file_path).unwrap();
        let selectors = contents
            .lines()
            .enumerate()
            .filter_map(|(idx, line)| {
                // Extract selectors (including in comments, removing fails-if annotations)
                let trimmed = line.trim();
                trimmed
                    .find("webgpu:")
                    .or_else(|| trimmed.find("unittests:"))
                    .map(|pos| (idx, &trimmed[pos..]))
            })
            .map(|(idx, line)| {
                // Crude en_US sort. '_' < ',' < ':' < digits < letters
                let sort_key = line
                    .chars()
                    .map(|c| {
                        if c.is_ascii_uppercase() {
                            c.to_ascii_lowercase()
                        } else if c == '_' {
                            ' '
                        } else if c == ':' {
                            '-'
                        } else {
                            c
                        }
                    })
                    .collect::<String>();
                (idx, line, sort_key)
            })
            .collect::<Vec<_>>();

        let mut sorted = selectors.clone();
        sorted.sort_by_key(|(_, _, sort_key)| sort_key.clone());

        if selectors != sorted {
            let (found, expected) = selectors
                .iter()
                .zip(sorted.iter())
                .find(|(a, b)| a != b)
                .unwrap();
            panic!(
                "{} is not sorted. First mismatch on line {}:\nFound: {}\nShould be: {}",
                file,
                found.0 + 1,
                found.1,
                expected.1,
            );
        }
    }
}
