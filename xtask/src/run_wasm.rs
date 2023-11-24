use anyhow::Context;

use pico_args::Arguments;

use crate::util::check_all_programs;

pub(crate) fn run_wasm(mut args: Arguments) -> Result<(), anyhow::Error> {
    let no_serve = args.contains("--no-serve");
    let release = args.contains("--release");

    let programs_needed: &[_] = if no_serve {
        &["wasm-bindgen"]
    } else {
        &["wasm-bindgen", "simple-http-server"]
    };

    check_all_programs(programs_needed)?;

    let release_flag: &[_] = if release { &["--release"] } else { &[] };
    let output_dir = if release { "release" } else { "debug" };

    let shell = xshell::Shell::new().context("Couldn't create xshell shell")?;
    shell.change_dir(String::from(env!("CARGO_MANIFEST_DIR")) + "/..");

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
        "cargo build --target wasm32-unknown-unknown --bin wgpu-examples --features webgl {release_flag...}"
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
