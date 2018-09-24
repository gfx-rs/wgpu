extern crate cbindgen;

use std::path::PathBuf;

const HEADER: &str = "
#ifdef WGPU_REMOTE
    typedef uint32_t WGPUId;
#else
    typedef void *WGPUId;
#endif
";

fn main() {
    let mut crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.push("../wgpu-native");

    let config = cbindgen::Config {
        header: Some(String::from(HEADER.trim())),
        enumeration: cbindgen::EnumConfig {
            prefix_with_name: true,
            ..Default::default()
        },
        export: cbindgen::ExportConfig {
            prefix: Some(String::from("WGPU")),
            exclude: vec![
                // We manually define `Id` is within the header, so exclude it here
                String::from("Id"),
            ],
            ..Default::default()
        },
        language: cbindgen::Language::C,
        ..Default::default()
    };

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .unwrap()
        .write_to_file("wgpu.h");
}
