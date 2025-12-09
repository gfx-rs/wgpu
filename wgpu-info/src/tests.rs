use std::{fs::File, io::BufWriter};

const ENV_VAR_SAVE: &str = "WGPU_INFO_SAVE_GPUCONFIG_REPORT";

// We use a test to generate the .gpuconfig file instead of using the cli directly
// as `cargo run --bin wgpu-info` would build a different set of dependencies, causing
// incremental changes to need to rebuild the wgpu stack twice, one for the tests
// and once for the cli binary.
//
// Needs to be kept in sync with the test in xtask/src/test.rs
#[test]
fn generate_gpuconfig_report() {
    let report = crate::report::GpuReport::generate();

    // If we don't get the env var, just test that we can generate the report, but don't save it
    // to avoid a race condition when other tests are reading the file.
    if std::env::var(ENV_VAR_SAVE).is_err() {
        println!("Set {ENV_VAR_SAVE} to generate a .gpuconfig report using this test");
        return;
    }

    let file = File::create(concat!(env!("CARGO_MANIFEST_DIR"), "/../.gpuconfig")).unwrap();
    let buf = BufWriter::new(file);
    report.into_json(buf).unwrap();
}
