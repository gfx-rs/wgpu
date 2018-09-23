extern crate cbindgen;

use std::path::PathBuf;

fn main() {
    let mut crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.push("../wgpu-native");

    let config = cbindgen::Config {
        enumeration: cbindgen::EnumConfig {
            prefix_with_name: true,
            ..Default::default()
        },
        ..Default::default()
    };

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .with_language(cbindgen::Language::C)
        .with_item_prefix("WGPU")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("wgpu.h");
}
