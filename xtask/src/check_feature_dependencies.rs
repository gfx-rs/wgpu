use pico_args::Arguments;
use xshell::Shell;

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

const ALL_WGPU_FEATURES: &[&str] = &[
    "dx12",
    "metal",
    "webgpu",
    "angle",
    "vulkan-portability",
    "webgl",
    "spirv",
    "glsl",
    "wgsl",
    "naga-ir",
    "serde",
    "counters",
    "fragile-send-sync-non-atomic-wasm",
    "static-dxc",
];

pub fn check_feature_dependencies(shell: Shell, arguments: Arguments) -> anyhow::Result<()> {
    let mut _args = arguments.finish();

    let features_no_webgl: Vec<&str> = ALL_WGPU_FEATURES
        .iter()
        .copied()
        .filter(|feature| *feature != "webgl")
        .collect();

    let requirements = [
        Requirement {
            human_readable_name: "wasm32 without `webgl` feature does not depend on `wgpu-core`",
            target: "wasm32-unknown-unknown",
            packages: &["wgpu"],
            features: &features_no_webgl,
            default_features: false,
            search_terms: &[Search::Negative("wgpu-core")],
        },
        Requirement {
            human_readable_name:
                "wasm32 with `webgpu` and `wgsl` feature does not depend on `naga`",
            target: "wasm32-unknown-unknown",
            packages: &["wgpu"],
            features: &["webgpu", "wgsl"],
            default_features: false,
            search_terms: &[Search::Negative("naga")],
        },
        Requirement {
            human_readable_name: "wasm32 with `webgl` feature depends on `glow`",
            target: "wasm32-unknown-unknown",
            packages: &["wgpu"],
            features: &["webgl"],
            default_features: false,
            search_terms: &[Search::Positive("glow")],
        },
        Requirement {
            human_readable_name: "windows with `webgl` does not depend on `glow`",
            target: "x86_64-pc-windows-msvc",
            packages: &["wgpu"],
            features: &["webgl"],
            default_features: false,
            search_terms: &[Search::Negative("glow")],
        },
        Requirement {
            human_readable_name: "apple with `vulkan` feature does not depend on `ash`",
            target: "aarch64-apple-darwin",
            packages: &["wgpu"],
            features: &["vulkan"],
            default_features: false,
            search_terms: &[Search::Negative("ash")],
        },
        Requirement {
            human_readable_name:
                "apple with `vulkan-portability` feature depends on `ash` and `renderdoc-sys`",
            target: "aarch64-apple-darwin",
            packages: &["wgpu"],
            features: &["vulkan-portability"],
            default_features: false,
            search_terms: &[Search::Positive("ash"), Search::Positive("renderdoc-sys")],
        },
        Requirement {
            human_readable_name: "apple with 'gles' feature does not depend on 'glow'",
            target: "aarch64-apple-darwin",
            packages: &["wgpu"],
            features: &["gles"],
            default_features: false,
            search_terms: &[Search::Negative("glow")],
        },
        Requirement {
            human_readable_name: "apple with 'angle' feature depends on 'glow' and `renderdoc-sys`",
            target: "aarch64-apple-darwin",
            packages: &["wgpu"],
            features: &["angle"],
            default_features: false,
            search_terms: &[Search::Positive("glow"), Search::Positive("renderdoc-sys")],
        },
        Requirement {
            human_readable_name: "apple with no features does not depend on 'renderdoc-sys'",
            target: "aarch64-apple-darwin",
            packages: &["wgpu"],
            features: &[],
            default_features: false,
            search_terms: &[Search::Negative("renderdoc-sys")],
        },
        Requirement {
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
        },
        Requirement {
            human_readable_name: "windows with no features depends on renderdoc-sys",
            target: "x86_64-pc-windows-msvc",
            packages: &["wgpu"],
            features: &[],
            default_features: false,
            search_terms: &[Search::Positive("renderdoc-sys")],
        },
        Requirement {
            human_readable_name: "emscripten with webgl feature does not depend on glow",
            target: "wasm32-unknown-emscripten",
            packages: &["wgpu"],
            features: &["webgl"],
            default_features: false,
            search_terms: &[Search::Negative("glow")],
        },
        Requirement {
            human_readable_name: "emscripten with gles feature depends on glow",
            target: "wasm32-unknown-emscripten",
            packages: &["wgpu"],
            features: &["gles"],
            default_features: false,
            search_terms: &[Search::Positive("glow")],
        },
        Requirement {
            human_readable_name: "x86-64 does not depend on portable-atomic",
            target: "x86_64-unknown-linux-gnu",
            packages: &["wgpu"],
            features: &[],
            default_features: false,
            search_terms: &[Search::Negative("portable-atomic")],
        },
        Requirement {
            human_readable_name: "ppc32 does depend on portable-atomic",
            target: "powerpc-unknown-linux-gnu",
            packages: &["wgpu"],
            features: &[],
            default_features: false,
            search_terms: &[Search::Positive("portable-atomic")],
        },
    ];

    let mut any_failures = false;
    for requirement in requirements {
        let mut cmd = shell
            .cmd("cargo")
            .args(["tree", "--target", requirement.target]);

        for package in requirement.packages {
            cmd = cmd.arg("--package").arg(package);
        }

        if !requirement.default_features {
            cmd = cmd.arg("--no-default-features");
        }

        if !requirement.features.is_empty() {
            cmd = cmd.arg("--features").arg(requirement.features.join(","));
        }

        log::info!("Checking Requirement: {}", requirement.human_readable_name);
        log::debug!("{:#?}", requirement);
        log::debug!("$ {cmd}");

        let output = cmd.read()?;

        log::debug!("{output}");

        for (i, search_term) in requirement.search_terms.into_iter().enumerate() {
            // Add a space and after to make sure we're getting a full match
            let found = match search_term {
                Search::Positive(search_term) => output.contains(&format!(" {search_term} ")),
                Search::Negative(search_term) => !output.contains(&format!(" {search_term} ")),
            };

            if found {
                log::info!(
                    "✅ Passed! ({} of {})",
                    i + 1,
                    requirement.search_terms.len()
                );
            } else {
                log::info!(
                    "❌ Failed! ({} of {})",
                    i + 1,
                    requirement.search_terms.len()
                );
                any_failures = true;
            }
        }
    }

    if any_failures {
        anyhow::bail!("Some feature dependencies are not met");
    }

    Ok(())
}
