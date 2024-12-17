use anyhow::Context;
use pico_args::Arguments;
use xshell::Shell;

use crate::bad_arguments;

/// Path to the webgpu_sys folder relative to the root of the repository
const WEBGPU_SYS_PATH: &str = "wgpu/src/backend/webgpu/webgpu_sys";
/// Path to the temporary clone of the wasm-bindgen repository.
///
/// This should be synchronized with the path in .gitignore.
const WASM_BINDGEN_TEMP_CLONE_PATH: &str =
    "wgpu/src/backend/webgpu/webgpu_sys/wasm_bindgen_clone_tmp";
/// Prefix of the file names in the wasm-bindgen repository that we need to copy.
///
/// This prefix will be added to all of the feature names to get the full file name.
/// Relative to the root of the wasm-bindgen repository.
const WEB_SYS_FILE_PREFIX: &str = "crates/web-sys/src/features/gen_";
const WEB_SYS_FEATURES_NEEDED: &[&str] = &[
    "Gpu",
    "GpuAdapter",
    "GpuAdapterInfo",
    "GpuAddressMode",
    "GpuAutoLayoutMode",
    "GpuBindGroup",
    "GpuBindGroupDescriptor",
    "GpuBindGroupEntry",
    "GpuBindGroupLayout",
    "GpuBindGroupLayoutDescriptor",
    "GpuBindGroupLayoutEntry",
    "GpuBlendComponent",
    "GpuBlendFactor",
    "GpuBlendOperation",
    "GpuBlendState",
    "GpuBuffer",
    "GpuBufferBinding",
    "GpuBufferBindingLayout",
    "GpuBufferBindingType",
    "GpuBufferDescriptor",
    "GpuBufferMapState",
    "GpuCanvasAlphaMode",
    "GpuCanvasContext",
    "GpuCanvasConfiguration",
    "GpuCanvasToneMapping",
    "GpuCanvasToneMappingMode",
    "GpuColorDict",
    "GpuColorTargetState",
    "GpuCommandBuffer",
    "GpuCommandBufferDescriptor",
    "GpuCommandEncoder",
    "GpuCommandEncoderDescriptor",
    "GpuCompareFunction",
    "GpuCompilationInfo",
    "GpuCompilationMessage",
    "GpuCompilationMessageType",
    "GpuComputePassDescriptor",
    "GpuComputePassEncoder",
    "GpuComputePassTimestampWrites",
    "GpuComputePipeline",
    "GpuComputePipelineDescriptor",
    "GpuCullMode",
    "GpuDepthStencilState",
    "GpuDevice",
    "GpuDeviceDescriptor",
    "GpuDeviceLostInfo",
    "GpuDeviceLostReason",
    "GpuError",
    "GpuErrorFilter",
    "GpuExternalTexture",
    "GpuExternalTextureBindingLayout",
    "GpuExternalTextureDescriptor",
    // "GpuExtent2dDict", Not yet implemented in web_sys
    "GpuExtent3dDict",
    "GpuFeatureName",
    "GpuFilterMode",
    "GpuFragmentState",
    "GpuFrontFace",
    "GpuTexelCopyBufferInfo",
    "GpuCopyExternalImageSourceInfo",
    "GpuTexelCopyTextureInfo",
    "GpuCopyExternalImageDestInfo",
    "GpuTexelCopyBufferLayout",
    "GpuIndexFormat",
    "GpuLoadOp",
    "gpu_map_mode",
    "GpuMipmapFilterMode",
    "GpuMultisampleState",
    "GpuObjectDescriptorBase",
    "GpuOrigin2dDict",
    "GpuOrigin3dDict",
    "GpuOutOfMemoryError",
    "GpuPipelineDescriptorBase",
    "GpuPipelineLayout",
    "GpuPipelineLayoutDescriptor",
    "GpuPowerPreference",
    "GpuPrimitiveState",
    "GpuPrimitiveTopology",
    "GpuProgrammableStage",
    "GpuQuerySet",
    "GpuQuerySetDescriptor",
    "GpuQueryType",
    "GpuQueue",
    "GpuQueueDescriptor",
    "GpuRenderBundle",
    "GpuRenderBundleDescriptor",
    "GpuRenderBundleEncoder",
    "GpuRenderBundleEncoderDescriptor",
    "GpuRenderPassColorAttachment",
    "GpuRenderPassDepthStencilAttachment",
    "GpuRenderPassDescriptor",
    "GpuRenderPassEncoder",
    "GpuRenderPassTimestampWrites",
    "GpuRenderPipeline",
    "GpuRenderPipelineDescriptor",
    "GpuRequestAdapterOptions",
    "GpuSampler",
    "GpuSamplerBindingLayout",
    "GpuSamplerBindingType",
    "GpuSamplerDescriptor",
    "GpuShaderModule",
    "GpuShaderModuleDescriptor",
    "GpuStencilFaceState",
    "GpuStencilOperation",
    "GpuStorageTextureAccess",
    "GpuStorageTextureBindingLayout",
    "GpuStoreOp",
    "GpuSupportedFeatures",
    "GpuSupportedLimits",
    "GpuTexture",
    "GpuTextureAspect",
    "GpuTextureBindingLayout",
    "GpuTextureDescriptor",
    "GpuTextureDimension",
    "GpuTextureFormat",
    "GpuTextureSampleType",
    "GpuTextureView",
    "GpuTextureViewDescriptor",
    "GpuTextureViewDimension",
    "GpuUncapturedErrorEvent",
    "GpuUncapturedErrorEventInit",
    "GpuValidationError",
    "GpuVertexAttribute",
    "GpuVertexBufferLayout",
    "GpuVertexFormat",
    "GpuVertexState",
    "GpuVertexStepMode",
    "WgslLanguageFeatures",
];

pub(crate) fn run_vendor_web_sys(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    // -- Argument Parsing --

    let no_cleanup = args.contains("--no-cleanup");

    // We only allow one of these arguments to be passed at a time.
    let version: Option<String> = args.opt_value_from_str("--version")?;
    let path_to_checkout_arg: Option<String> = args.opt_value_from_str("--path-to-checkout")?;

    // Plain text of the command that was run.
    let argument_description;
    // Path to the checkout we're using
    let path_to_wasm_bindgen_checkout;
    match (path_to_checkout_arg.as_deref(), version.as_deref()) {
        (Some(path), None) => {
            argument_description = format!("--path-to-checkout {path}");
            path_to_wasm_bindgen_checkout = path
        }
        (None, Some(version)) => {
            argument_description = format!("--version {version}");
            path_to_wasm_bindgen_checkout = WASM_BINDGEN_TEMP_CLONE_PATH
        }
        (Some(_), Some(_)) => {
            bad_arguments!("Cannot use both --path-to-checkout and --version at the same time")
        }
        (None, None) => {
            bad_arguments!("Expected either --path-to-checkout or --version")
        }
    };

    let unknown_args = args.finish();
    if !unknown_args.is_empty() {
        bad_arguments!(
            "Unknown arguments to vendor-web-sys subcommand: {:?}",
            unknown_args
        );
    }

    // -- Main Logic --

    eprintln!("# Removing {WEBGPU_SYS_PATH}");
    shell
        .remove_path(WEBGPU_SYS_PATH)
        .context("could not remove webgpu_sys")?;

    if let Some(ref version) = version {
        eprintln!("# Cloning wasm-bindgen repository with version {version}");
        shell
            .cmd("git")
            .args([
                "clone",
                "-b",
                version,
                "--depth",
                "1",
                "https://github.com/rustwasm/wasm-bindgen.git",
                WASM_BINDGEN_TEMP_CLONE_PATH,
            ])
            .ignore_stderr()
            .run()
            .context("Could not clone wasm-bindgen repository")?;
    }

    if let Some(ref path) = path_to_checkout_arg {
        eprintln!("# Using local checkout of wasm-bindgen at {path}");
    }

    shell
        .create_dir(WEBGPU_SYS_PATH)
        .context("Could not create webgpu_sys folder")?;

    // The indentation here does not matter, as rustfmt will normalize it.
    let file_prefix = format!("\
        // DO NOT EDIT THIS FILE!
        // 
        // This module part of a subset of web-sys that is used by wgpu's webgpu backend.
        //
        // These bindings are vendored into wgpu for the sole purpose of letting
        // us pin the WebGPU backend to a specific version of the bindings, not
        // to enable local changes. There are no provisions to preserve changes
        // you make here the next time we re-vendor the bindings.
        //
        // The `web-sys` crate does not treat breaking changes to the WebGPU API
        // as semver breaking changes, as WebGPU is \"unstable\". This means Cargo
        // will not let us mix versions of `web-sys`, pinning WebGPU bindings to
        // a specific version, while letting other bindings like WebGL get
        // updated. Vendoring WebGPU was the workaround we chose.
        //
        // Vendoring also allows us to avoid building `web-sys` with
        // `--cfg=web_sys_unstable_apis`, needed to get the WebGPU bindings.
        //
        // If you want to improve the generated code, please submit a PR to the https://github.com/rustwasm/wasm-bindgen repository.
        // 
        // This file was generated by the `cargo xtask vendor-web-sys {argument_description}` command.\n"
    );

    eprintln!(
        "# Copying {} files and removing `#[cfg(...)]` attributes",
        WEB_SYS_FEATURES_NEEDED.len()
    );

    // Matches all `#[cfg(...)]` attributes, including multi-line ones.
    //
    // The ? is very important, otherwise the regex will match the entire file.
    let regex = regex_lite::RegexBuilder::new(r#"#\[cfg\(.*?\)\]"#)
        .dot_matches_new_line(true)
        .build()
        .unwrap();

    for &feature in WEB_SYS_FEATURES_NEEDED {
        let feature_file =
            format!("{path_to_wasm_bindgen_checkout}/{WEB_SYS_FILE_PREFIX}{feature}.rs");
        let destination = format!("{WEBGPU_SYS_PATH}/gen_{feature}.rs");

        let file_contents = shell
            .read_file(&feature_file)
            .context("Could not read file")?;

        let mut file_contents = regex.replace_all(&file_contents, "").to_string();
        file_contents.insert_str(0, &file_prefix);

        shell
            .write_file(destination, file_contents.as_bytes())
            .context("could not write file")?;
    }

    eprintln!("# Writing mod.rs file");

    let mut module_file_contents = format!(
        "\
        //! Bindings to the WebGPU API.
        //! 
        //! Internally vendored from the `web-sys` crate until the WebGPU binding are stabilized.
        {file_prefix}
        #![allow(unused_imports, non_snake_case)]\n"
    );

    module_file_contents.push_str("use web_sys::{Event, EventTarget};\n");

    for &feature in WEB_SYS_FEATURES_NEEDED {
        module_file_contents.push_str(&format!("mod gen_{};\n", feature));
        module_file_contents.push_str(&format!("pub use gen_{}::*;\n", feature));
    }

    shell.write_file(format!("{}/mod.rs", WEBGPU_SYS_PATH), module_file_contents)?;

    eprintln!("# Formatting files");

    shell
        .cmd("rustfmt")
        .arg(format!("{WEBGPU_SYS_PATH}/mod.rs"))
        .run()
        .context("could not format")?;

    if !no_cleanup {
        // We only need to remove this if we cloned it in the first place.
        if version.is_some() {
            eprintln!("# Removing wasm-bindgen clone");
            shell
                .remove_path(WASM_BINDGEN_TEMP_CLONE_PATH)
                .context("could not remove wasm-bindgen clone")?;
        }
    } else {
        eprintln!("# Skipping cleanup");
    }

    eprintln!("# Finished!");

    Ok(())
}
