#![cfg(feature = "wgsl-in")]

use naga::{front::wgsl, valid::Validator};
use std::{ffi::OsStr, fs, path::Path};

/// Runs through all example shaders and ensures they are valid wgsl.
#[test]
pub fn parse_example_wgsl() {
    let example_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("examples");

    println!("Looking for examples in {}", example_path.display());

    let mut example_paths = Vec::new();
    for example_entry in walkdir::WalkDir::new(example_path) {
        let Ok(dir_entry) = example_entry else {
            continue;
        };

        if !dir_entry.file_type().is_file() {
            continue;
        }

        let path = dir_entry.path();

        if path.extension().map(OsStr::to_string_lossy).as_deref() == Some("wgsl") {
            example_paths.push(path.to_path_buf());
        }
    }

    assert!(!example_paths.is_empty(), "No examples found!");

    println!("Found {} examples", example_paths.len());

    for example_path in example_paths {
        println!("\tParsing {}", example_path.display());

        let shader = fs::read_to_string(&example_path).unwrap();

        let module = wgsl::parse_str(&shader).unwrap();
        //TODO: re-use the validator
        Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
