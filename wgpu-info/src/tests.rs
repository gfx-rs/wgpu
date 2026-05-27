use std::{collections::BTreeMap, fs::File, io::BufWriter};

fn unified_diff(label: &str, a: &str, b: &str) -> String {
    use std::{fs, process::Command};
    let dir = std::env::temp_dir();
    let a_path = dir.join(format!("wgpu-info-{label}-custom.json"));
    let b_path = dir.join(format!("wgpu-info-{label}-core.json"));
    fs::write(&a_path, a).unwrap();
    fs::write(&b_path, b).unwrap();
    let out = Command::new("diff")
        .args(["-u", "--label", "with-custom", "--label", "without-custom"])
        .args([&a_path, &b_path])
        .output()
        .unwrap();
    String::from_utf8(out.stdout).unwrap()
}

fn adapter_key(info: &wgpu::AdapterInfo) -> String {
    format!("{:?}/{}", info.backend, info.name)
}

// Normalized view of an adapter for comparison purposes.
// - Experimental features stripped (wgpu-c-backend intentionally omits them).
// - Texture format features sorted by format name for deterministic JSON output
//   (HashMap iteration order is random).
#[derive(serde::Serialize)]
struct NormalizedAdapter<'a> {
    info: &'a wgpu::AdapterInfo,
    features: wgpu::Features,
    limits: &'a wgpu::Limits,
    texture_format_features: BTreeMap<String, &'a wgpu::TextureFormatFeatures>,
}

fn normalize(dev: &crate::report::AdapterReport) -> NormalizedAdapter<'_> {
    let features = dev.features & !wgpu::Features::all_experimental_mask();
    let texture_format_features = dev
        .texture_format_features
        .iter()
        .map(|(fmt, feats)| (format!("{fmt:?}"), feats))
        .collect();
    NormalizedAdapter {
        info: &dev.info,
        features,
        limits: &dev.limits,
        texture_format_features,
    }
}

fn to_json(value: &impl serde::Serialize) -> String {
    serde_json::to_string_pretty(value).unwrap()
}

#[test]
fn custom_backend_matches_wgpu_core() {
    let with_custom = crate::report::GpuReport::generate();

    std::env::set_var("WGPU_NO_CUSTOM_BACKEND", "1");
    let without_custom = crate::report::GpuReport::generate();
    std::env::remove_var("WGPU_NO_CUSTOM_BACKEND");

    let without_map: std::collections::HashMap<String, &crate::report::AdapterReport> =
        without_custom
            .devices
            .iter()
            .map(|d| (adapter_key(&d.info), d))
            .collect();

    let mut failures: Vec<String> = Vec::new();

    for custom_dev in &with_custom.devices {
        let key = adapter_key(&custom_dev.info);
        let Some(core_dev) = without_map.get(&key) else {
            println!(
                "custom_backend_matches_wgpu_core: skipping {key} (not in wgpu-core run)"
            );
            continue;
        };

        let a = to_json(&normalize(custom_dev));
        let b = to_json(&normalize(core_dev));

        if a != b {
            failures.push(format!(
                "Adapter '{}' differs:\n{}",
                key,
                unified_diff(&key.replace('/', "_"), &a, &b)
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "GpuReport differs for {} adapter(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

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
