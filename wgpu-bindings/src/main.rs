extern crate cbindgen;

use std::path::PathBuf;

fn main() {
    let mut crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.push("../wgpu-native");

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("bindings.h");
}
