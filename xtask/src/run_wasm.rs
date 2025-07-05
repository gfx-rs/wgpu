use anyhow::Context;

use pico_args::Arguments;
use xshell::Shell;

use crate::util::{check_all_programs, Program};

pub(crate) fn run_wasm(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    let should_serve = !args.contains("--no-serve");
    let release = args.contains("--release");

    let mut programs_needed = vec![Program {
        crate_name: "wasm-bindgen-cli",
        binary_name: "wasm-bindgen",
    }];

    if should_serve {
        programs_needed.push(Program {
            crate_name: "simple-http-server",
            binary_name: "simple-http-server",
        });
    }

    check_all_programs(&programs_needed)?;

    let release_flag: &[_] = if release { &["--release"] } else { &[] };
    let output_dir = if release { "release" } else { "debug" };

    log::info!("building webgpu examples");

    let cargo_args = args.finish();

    xshell::cmd!(
        shell,
        "cargo build --target wasm32-unknown-unknown -p wgpu-examples --no-default-features --features webgpu {release_flag...}"
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
        "cargo build --target wasm32-unknown-unknown -p wgpu-examples --no-default-features --features webgl {release_flag...}"
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
        .read_dir("examples/features/web-static")
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

    if should_serve {
        log::info!("serving on port 8000");

        // Explicitly specify the IP address to 127.0.0.1 since otherwise simple-http-server will
        // print http://0.0.0.0:8000 as url which is not a secure context and thus doesn't allow
        // running WebGPU!
        xshell::cmd!(
            shell,
            "simple-http-server target/generated -c wasm,html,js -i --coep --coop --ip 127.0.0.1"
        )
        .quiet()
        .run()
        .context("Failed to simple-http-server")?;
    }

    Ok(())
}
