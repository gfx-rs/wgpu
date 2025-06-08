// Cargo-metadata doesn't compile on wasm due to old cargo-util-schemas dependency.
// Since this test isn't dependent on the current architecture, we can just skip it on wasm without any issues.
#![cfg(not(target_arch = "wasm32"))]

use std::process::Command;

#[derive(Debug)]
enum Search<'a> {
    Positive(&'a str),
    Negative(&'a str),
}

#[derive(Debug)]
struct Requirement<'a> {
    human_readable_name: &'a str,
    target: &'a str,
    packages: &'a [&'a str],
    features: &'a [&'a str],
    default_features: bool,
    search_terms: &'a [Search<'a>],
}

fn check_feature_dependency(requirement: Requirement) {
    println!("Checking: {}", requirement.human_readable_name);

    let mut args = Vec::new();
    args.extend(["tree", "--target", requirement.target]);

    for package in requirement.packages {
        args.push("--package");
        args.push(package);
    }

    if !requirement.default_features {
        args.push("--no-default-features");
    }

    let features = requirement.features.join(",");
    if !requirement.features.is_empty() {
        args.push("--features");
        args.push(&features);
    }

    println!("$ cargo {}", args.join(" "));

    let output = Command::new("cargo")
        .args(&args)
        .output()
        .expect("Failed to run cargo tree")
        .stdout;
    let output = String::from_utf8(output).expect("Output is not valid UTF-8");

    let mut any_failed = false;
    println!("{output}");

    for (i, search_term) in requirement.search_terms.iter().enumerate() {
        // Add a space and after to make sure we're getting a full match
        let found = match search_term {
            Search::Positive(search_term) => output.contains(&format!(" {search_term} ")),
            Search::Negative(search_term) => !output.contains(&format!(" {search_term} ")),
        };

        if found {
            println!(
                "✅ Passed! ({} of {})",
                i + 1,
                requirement.search_terms.len()
            );
        } else {
            println!(
                "❌ Failed! ({} of {})",
                i + 1,
                requirement.search_terms.len()
            );
            any_failed = true;
        }
    }

    assert!(!any_failed);
}

fn get_all_wgpu_features() -> Vec<String> {
    let metadata = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .unwrap();

    metadata
        .packages
        .iter()
        .find(|p| p.name.as_str() == "wgpu")
        .unwrap()
        .features
        .keys()
        .cloned()
        .collect()
}

#[test]
fn wasm32_without_webgl_or_noop_does_not_depend_on_wgpu_core() {
    let all_features = get_all_wgpu_features();

    let removed_features = ["webgl", "noop", "wgpu-core"];

    let features_no_webgl: Vec<&str> = all_features
        .iter()
        .map(String::as_str)
        .filter(|&feature| !removed_features.contains(&feature))
        .collect();

    check_feature_dependency(Requirement {
        human_readable_name:
            "wasm32 without `webgl` or `noop` feature does not depend on `wgpu-core`",
        target: "wasm32-unknown-unknown",
        packages: &["wgpu"],
        features: &features_no_webgl,
        default_features: false,
        search_terms: &[Search::Negative("wgpu-core")],
    });
}

#[test]
fn wasm32_with_webgpu_and_wgsl_does_not_depend_on_naga() {
    check_feature_dependency(Requirement {
        human_readable_name: "wasm32 with `webgpu` and `wgsl` feature does not depend on `naga`",
        target: "wasm32-unknown-unknown",
        packages: &["wgpu"],
        features: &["webgpu", "wgsl"],
        default_features: false,
        search_terms: &[Search::Negative("naga")],
    });
}

#[test]
fn wasm32_with_webgl_depends_on_glow() {
    check_feature_dependency(Requirement {
        human_readable_name: "wasm32 with `webgl` feature depends on `glow`",
        target: "wasm32-unknown-unknown",
        packages: &["wgpu"],
        features: &["webgl"],
        default_features: false,
        search_terms: &[Search::Positive("glow")],
    });
}

#[test]
fn wasm32_with_only_custom_backend_does_not_depend_on_web_specifics() {
    check_feature_dependency(Requirement {
        human_readable_name: "wasm32 with only the `custom` backend does not depend on web-specific bindings [`wasm-bindgen`, `js-sys`, `web-sys`]",
        target: "wasm32-unknown-unknown",
        packages: &["wgpu"],
        features: &["custom"],
        default_features: false,
        search_terms: &[Search::Negative("wasm-bindgen"), Search::Negative("js-sys"), Search::Negative("web-sys")],
    });
}

#[test]
fn wasm32_with_webgpu_backend_does_depend_on_web_specifics() {
    check_feature_dependency(Requirement {
        human_readable_name: "wasm32 with the `webgpu` backend depends on web-specific bindings [`wasm-bindgen`, `js-sys`, `web-sys`]",
        target: "wasm32-unknown-unknown",
        packages: &["wgpu"],
        features: &["webgpu"],
        default_features: false,
        search_terms: &[Search::Positive("wasm-bindgen"), Search::Positive("js-sys"), Search::Positive("web-sys")],
    });
}

#[test]
fn wasm32_with_webgl_backend_does_depend_on_web_specifics() {
    check_feature_dependency(Requirement {
        human_readable_name: "wasm32 with the `webgl` backend depends on web-specific bindings [`wasm-bindgen`, `js-sys`, `web-sys`]",
        target: "wasm32-unknown-unknown",
        packages: &["wgpu"],
        features: &["webgl"],
        default_features: false,
        search_terms: &[Search::Positive("wasm-bindgen"), Search::Positive("js-sys"), Search::Positive("web-sys")],
    });
}

#[test]
fn windows_with_webgpu_webgl_backend_does_not_depend_on_web_specifics() {
    check_feature_dependency(Requirement {
        human_readable_name: "windows with the `webgpu` and `webgl` backends enabled does not depend on web-specific bindings [`wasm-bindgen`, `js-sys`, `web-sys`]",
        target: "x86_64-pc-windows-msvc",
        packages: &["wgpu"],
        features: &["webgpu", "webgl"],
        default_features: false,
        search_terms: &[Search::Negative("wasm-bindgen"), Search::Negative("js-sys"), Search::Negative("web-sys")],
    });
}

#[test]
fn windows_with_webgl_does_not_depend_on_glow() {
    check_feature_dependency(Requirement {
        human_readable_name: "windows with `webgl` does not depend on `glow`",
        target: "x86_64-pc-windows-msvc",
        packages: &["wgpu"],
        features: &["webgl"],
        default_features: false,
        search_terms: &[Search::Negative("glow")],
    });
}

#[test]
fn apple_with_vulkan_does_not_depend_on_ash() {
    check_feature_dependency(Requirement {
        human_readable_name: "apple with `vulkan` feature does not depend on `ash`",
        target: "aarch64-apple-darwin",
        packages: &["wgpu"],
        features: &["vulkan"],
        default_features: false,
        search_terms: &[Search::Negative("ash")],
    });
}

#[test]
fn apple_with_vulkan_portability_depends_on_ash_and_renderdoc_sys() {
    check_feature_dependency(Requirement {
        human_readable_name:
            "apple with `vulkan-portability` feature depends on `ash` and `renderdoc-sys`",
        target: "aarch64-apple-darwin",
        packages: &["wgpu"],
        features: &["vulkan-portability"],
        default_features: false,
        search_terms: &[Search::Positive("ash"), Search::Positive("renderdoc-sys")],
    });
}

#[test]
fn apple_with_gles_does_not_depend_on_glow() {
    check_feature_dependency(Requirement {
        human_readable_name: "apple with 'gles' feature does not depend on 'glow'",
        target: "aarch64-apple-darwin",
        packages: &["wgpu"],
        features: &["gles"],
        default_features: false,
        search_terms: &[Search::Negative("glow")],
    });
}

#[test]
fn apple_with_angle_depends_on_glow_and_renderdoc_sys() {
    check_feature_dependency(Requirement {
        human_readable_name: "apple with 'angle' feature depends on 'glow' and `renderdoc-sys`",
        target: "aarch64-apple-darwin",
        packages: &["wgpu"],
        features: &["angle"],
        default_features: false,
        search_terms: &[Search::Positive("glow"), Search::Positive("renderdoc-sys")],
    });
}

#[test]
fn apple_with_no_features_does_not_depend_on_renderdoc_sys() {
    check_feature_dependency(Requirement {
        human_readable_name: "apple with no features does not depend on 'renderdoc-sys'",
        target: "aarch64-apple-darwin",
        packages: &["wgpu"],
        features: &[],
        default_features: false,
        search_terms: &[Search::Negative("renderdoc-sys")],
    });
}

#[test]
fn windows_with_no_features_does_not_depend_on_glow_windows_or_ash() {
    check_feature_dependency(Requirement {
        human_readable_name:
            "windows with no features does not depend on 'glow', `windows`, or `ash`",
        target: "x86_64-pc-windows-msvc",
        packages: &["wgpu"],
        features: &[],
        default_features: false,
        search_terms: &[
            Search::Negative("glow"),
            Search::Negative("windows"),
            Search::Negative("ash"),
        ],
    });
}

#[test]
fn windows_with_no_features_depends_on_renderdoc_sys() {
    check_feature_dependency(Requirement {
        human_readable_name: "windows with no features depends on renderdoc-sys",
        target: "x86_64-pc-windows-msvc",
        packages: &["wgpu"],
        features: &[],
        default_features: false,
        search_terms: &[Search::Positive("renderdoc-sys")],
    });
}

#[test]
fn emscripten_with_webgl_does_not_depend_on_glow() {
    check_feature_dependency(Requirement {
        human_readable_name: "emscripten with webgl feature does not depend on glow",
        target: "wasm32-unknown-emscripten",
        packages: &["wgpu"],
        features: &["webgl"],
        default_features: false,
        search_terms: &[Search::Negative("glow")],
    });
}

#[test]
fn emscripten_with_gles_depends_on_glow() {
    check_feature_dependency(Requirement {
        human_readable_name: "emscripten with gles feature depends on glow",
        target: "wasm32-unknown-emscripten",
        packages: &["wgpu"],
        features: &["gles"],
        default_features: false,
        search_terms: &[Search::Positive("glow")],
    });
}

#[test]
fn x86_64_does_not_depend_on_portable_atomic() {
    check_feature_dependency(Requirement {
        human_readable_name: "x86-64 does not depend on portable-atomic",
        target: "x86_64-unknown-linux-gnu",
        packages: &["wgpu"],
        features: &[],
        default_features: false,
        search_terms: &[Search::Negative("portable-atomic")],
    });
}

#[test]
fn ppc32_does_depend_on_portable_atomic() {
    check_feature_dependency(Requirement {
        human_readable_name: "ppc32 does depend on portable-atomic",
        target: "powerpc-unknown-linux-gnu",
        packages: &["wgpu"],
        features: &[],
        default_features: false,
        search_terms: &[Search::Positive("portable-atomic")],
    });
}
