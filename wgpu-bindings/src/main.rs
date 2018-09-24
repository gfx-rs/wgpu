extern crate cbindgen;

use std::path::PathBuf;

fn main() {
    let mut crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.push("../wgpu-native");

    let config = cbindgen::Config {
        header: Some(String::from("#ifdef WGPU_REMOTE\n    typedef uint32_t WGPUId;\n#else\n    typedef void *WGPUId;\n#endif")),
        enumeration: cbindgen::EnumConfig {
            prefix_with_name: true,
            ..Default::default()
        },
        export: cbindgen::ExportConfig {
            prefix: Some(String::from("WGPU")),
            exclude: vec![
                // We manually define `Id` is with an `#ifdef`, so exclude it here
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
