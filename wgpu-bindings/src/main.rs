extern crate cbindgen;

use cbindgen::ItemType as It;
use std::path::PathBuf;

struct Binding<'a> {
    library: &'a str,
    features: &'a [&'a str],
    output: &'a str,
    item_types: &'a [It],
}

const BINDINGS: [Binding; 3] = [
    Binding {
        library: "wgpu-native",
        features: &["local"],
        output: "wgpu.h",
        item_types: &[It::Enums, It::Constants, It::Structs, It::Typedefs, It::Functions],
    },
    Binding {
        library: "wgpu-native",
        features: &["remote"],
        output: "wgpu-native.h",
        item_types: &[It::Enums, It::Constants, It::Structs, It::Typedefs],
    },
    Binding {
        library: "wgpu-remote",
        features: &[],
        output: "wgpu-remote.h",
        item_types: &[It::Enums, It::Constants, It::Structs, It::Typedefs, It::OpaqueItems, It::Functions],
    },
];

fn main() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let parent = crate_dir.parent().unwrap();

    for bind in &BINDINGS {
        let config = cbindgen::Config {
            enumeration: cbindgen::EnumConfig {
                prefix_with_name: true,
                ..Default::default()
            },
            export: cbindgen::ExportConfig {
                prefix: Some(String::from("WGPU")),
                item_types: bind.item_types.to_vec(),
                ..Default::default()
            },
            language: cbindgen::Language::C,
            ..Default::default()
        };

        println!("Generating {}...", bind.output);
        cbindgen::Builder::new()
            .with_crate(parent.join(bind.library))
            .with_config(config)
            .with_parse_expand(&[bind.library])
            .with_parse_expand_features(bind.features)
            .generate()
            .unwrap()
            .write_to_file(crate_dir.join(bind.output));
    }
}
