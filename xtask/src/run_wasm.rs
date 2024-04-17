use anyhow::Context;

use pico_args::Arguments;
use xshell::Shell;

use crate::util::{check_all_programs, Program};

pub(crate) fn run_wasm(shell: Shell, mut args: Arguments) -> Result<(), anyhow::Error> {
    let no_serve = args.contains("--no-serve");
    let release = args.contains("--release");

    let programs_needed: &[_] = if no_serve {
        &[Program {
            crate_name: "wasm-bindgen-cli",
            binary_name: "wasm-bindgen",
        }]
    } else {
        &[
            Program {
                crate_name: "wasm-bindgen-cli",
                binary_name: "wasm-bindgen",
            },
            Program {
                crate_name: "simple-http-server",
                binary_name: "simple-http-server",
            },
        ]
    };

    check_all_programs(programs_needed)?;

    let release_flag: &[_] = if release { &["--release"] } else { &[] };
    let output_dir = if release { "release" } else { "debug" };

    log::info!("building webgpu examples");

    let cargo_args = args.finish();

    xshell::cmd!(
        shell,
        "cargo build --target wasm32-unknown-unknown --bin wgpu-examples {release_flag...}"
    )
    .args(&cargo_args)
    .quiet()
    .run()
    .context("Failed to build webgpu examples for wasm")?;

    log::info!("running wasm-bindgen on webgpu examples");

    xshell::cmd!(
        shell,
        "wasm-bindgen target/wasm32-unknown-unknown/{output_dir}/wgpu-examples.wasm --target web --no-typescript --out-dir target/generated --out-name webgpu"
    )
    .quiet()
    .run()
    .context("Failed to run wasm-bindgen")?;

    log::info!("building webgl examples");

    xshell::cmd!(
        shell,
        "cargo build --target wasm32-unknown-unknown --bin wgpu-examples --no-default-features --features wgsl,webgl {release_flag...}"
    )
    .args(&cargo_args)
    .quiet()
    .run()
    .context("Failed to build webgl examples for wasm")?;

    log::info!("running wasm-bindgen on webgl examples");

    xshell::cmd!(
            shell,
            "wasm-bindgen target/wasm32-unknown-unknown/{output_dir}/wgpu-examples.wasm --target web --no-typescript --out-dir target/generated --out-name webgl2"
        )
        .quiet()
        .run()
        .context("Failed to run wasm-bindgen")?;

    let static_files = shell
        .read_dir("examples/static")
        .context("Failed to enumerate static files")?;

    for file in static_files {
        log::info!(
            "copying static file \"{}\"",
            file.canonicalize().unwrap().display()
        );

        shell
            .copy_file(&file, "target/generated")
            .with_context(|| format!("Failed to copy static file \"{}\"", file.display()))?;
    }

    if !no_serve {
        log::info!("serving on port 8000");

        xshell::cmd!(
            shell,
            "simple-http-server target/generated -c wasm,html,js -i --coep --coop"
        )
        .quiet()
        .run()
        .context("Failed to simple-http-server")?;
    }

    Ok(())
}
