use std::{
    io::{BufRead, BufReader},
    path::Path,
    process::Stdio,
};

use anyhow::{bail, Context};

use crate::{
    cli::{ValidateHlslCommand, ValidateSubcommand},
    fs::open_file,
    glob::visit_files,
    path::join_path,
    process::{which, EasyCommand},
    result::{ErrorStatus, LogIfError},
};

fn ack_visiting(path: &Path) {
    log::info!("Validating {}", path.display());
}

pub(crate) fn validate(cmd: ValidateSubcommand) -> anyhow::Result<()> {
    let snapshots_base_out = join_path(["tests", "out"]);
    let err_status = match cmd {
        ValidateSubcommand::Spirv => {
            let spirv_as = "spirv-as";
            which(spirv_as)?;

            let spirv_val = "spirv-val";
            which(spirv_val)?;

            visit_files(snapshots_base_out, "spv/*.spvasm", |path| {
                validate_spirv(path, spirv_as, spirv_val)
            })
        }
        ValidateSubcommand::Metal => {
            let xcrun = "xcrun";
            which(xcrun)?;
            visit_files(snapshots_base_out, "msl/*.msl", |path| {
                validate_metal(path, xcrun)
            })
        }
        ValidateSubcommand::Glsl => {
            let glslang_validator = "glslangValidator";
            which(glslang_validator)?;
            let mut err_status = ErrorStatus::NoFailuresFound;
            for (glob, type_arg) in [
                ("glsl/*.Vertex.glsl", "vert"),
                ("glsl/*.Fragment.glsl", "frag"),
                ("glsl/*.Compute.glsl", "comp"),
            ] {
                let type_err_status = visit_files(&snapshots_base_out, glob, |path| {
                    validate_glsl(path, type_arg, glslang_validator)
                });
                err_status = err_status.merge(type_err_status);
            }
            err_status
        }
        ValidateSubcommand::Dot => {
            let dot = "dot";
            which(dot)?;
            visit_files(snapshots_base_out, "dot/*.dot", |path| {
                validate_dot(path, dot)
            })
        }
        ValidateSubcommand::Wgsl => {
            let mut paths = vec![];
            let mut error_status = visit_files(snapshots_base_out, "wgsl/*.wgsl", |path| {
                paths.push(path.to_owned());
                Ok(())
            });
            validate_wgsl(&paths).log_if_err_found(&mut error_status);
            error_status
        }
        ValidateSubcommand::Hlsl(cmd) => match cmd {
            ValidateHlslCommand::Dxc => {
                let bin = "dxc";
                which(bin)?;
                visit_files(snapshots_base_out, "hlsl/*.hlsl", |path| {
                    visit_hlsl_shaders(path, bin, validate_hlsl_with_dxc)
                })
            }
            ValidateHlslCommand::Fxc => {
                let bin = "fxc";
                which(bin)?;
                visit_files(snapshots_base_out, "hlsl/*.hlsl", |path| {
                    visit_hlsl_shaders(path, bin, validate_hlsl_with_fxc)
                })
            }
        },
    };
    match err_status {
        ErrorStatus::NoFailuresFound => Ok(()),
        ErrorStatus::OneOrMoreFailuresFound => {
            bail!("failed to validate one or more files, see above output for more details")
        }
    }
}

fn validate_spirv(path: &Path, spirv_as: &str, spirv_val: &str) -> anyhow::Result<()> {
    ack_visiting(path);
    let second_line = {
        let mut file = BufReader::new(open_file(path)?);
        let mut buf = String::new();
        file.read_line(&mut buf)
            .with_context(|| format!("failed to read first line from {path:?}"))?;
        buf.clear();
        file.read_line(&mut buf)
            .with_context(|| format!("failed to read second line from {path:?}"))?;
        buf
    };
    let expected_header_prefix = "; Version: ";
    let Some(version) =
        second_line.strip_prefix(expected_header_prefix) else {
            bail!(
                "no {expected_header_prefix:?} header found in {path:?}"
            );
        };
    let file = open_file(path)?;
    let mut spirv_as_cmd = EasyCommand::new(spirv_as, |cmd| {
        cmd.stdin(Stdio::from(file))
            .stdout(Stdio::piped())
            .arg("--target-env")
            .arg(format!("spv{version}"))
            .args(["-", "-o", "-"])
    });
    let child = spirv_as_cmd
        .spawn()
        .with_context(|| format!("failed to spawn {spirv_as_cmd:?}"))?;
    EasyCommand::new(spirv_val, |cmd| cmd.stdin(child.stdout.unwrap())).success()
}

fn validate_metal(path: &Path, xcrun: &str) -> anyhow::Result<()> {
    ack_visiting(path);
    let first_line = {
        let mut file = BufReader::new(open_file(path)?);
        let mut buf = String::new();
        file.read_line(&mut buf)
            .with_context(|| format!("failed to read header from {path:?}"))?;
        buf
    };
    let expected_header_prefix = "// language: ";
    let Some(language) =
        first_line.strip_prefix(expected_header_prefix) else {
            bail!(
                "no {expected_header_prefix:?} header found in {path:?}"
            );
        };
    let language = language.strip_suffix('\n').unwrap_or(language);

    let file = open_file(path)?;
    EasyCommand::new(xcrun, |cmd| {
        cmd.stdin(Stdio::from(file))
            .args(["-sdk", "macosx", "metal", "-mmacosx-version-min=10.11"])
            .arg(format!("-std=macos-{language}"))
            .args(["-x", "metal", "-", "-o", "/dev/null"])
    })
    .success()
}

fn validate_glsl(path: &Path, type_arg: &str, glslang_validator: &str) -> anyhow::Result<()> {
    ack_visiting(path);
    let file = open_file(path)?;
    EasyCommand::new(glslang_validator, |cmd| {
        cmd.stdin(Stdio::from(file))
            .args(["--stdin", "-S"])
            .arg(type_arg)
    })
    .success()
}

fn validate_dot(path: &Path, dot: &str) -> anyhow::Result<()> {
    ack_visiting(path);
    let file = open_file(path)?;
    EasyCommand::new(dot, |cmd| {
        cmd.stdin(Stdio::from(file)).stdout(Stdio::null())
    })
    .success()
}

fn validate_wgsl(paths: &[std::path::PathBuf]) -> anyhow::Result<()> {
    EasyCommand::new("cargo", |cmd| {
        cmd.args(["run", "-p", "naga-cli", "--", "--bulk-validate"])
            .args(paths)
    })
    .success()
}

fn visit_hlsl_shaders(
    path: &Path,
    bin: &str,
    mut consume_config_item: impl FnMut(&Path, hlsl_snapshots::ConfigItem, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    ack_visiting(path);
    let hlsl_snapshots::Config {
        vertex,
        fragment,
        compute,
    } = hlsl_snapshots::Config::from_path(path.with_extension("ron"))?;
    let mut status = ErrorStatus::NoFailuresFound;
    for shader in [vertex, fragment, compute].into_iter().flatten() {
        consume_config_item(path, shader, bin).log_if_err_found(&mut status);
    }
    match status {
        ErrorStatus::NoFailuresFound => Ok(()),
        ErrorStatus::OneOrMoreFailuresFound => bail!(
            "one or more shader HLSL shader tests failed for {}",
            path.display()
        ),
    }
}

fn validate_hlsl_with_dxc(
    file: &Path,
    config_item: hlsl_snapshots::ConfigItem,
    dxc: &str,
) -> anyhow::Result<()> {
    // Reference:
    // <https://github.com/microsoft/DirectXShaderCompiler/blob/6ee4074a4b43fa23bf5ad27e4f6cafc6b835e437/tools/clang/docs/UsingDxc.rst>.
    validate_hlsl(
        file,
        dxc,
        config_item,
        &[
            "-Wno-parentheses-equality",
            "-Zi",
            "-Qembed_debug",
            "-Od",
            "-HV",
            "2018",
        ],
    )
}

fn validate_hlsl_with_fxc(
    file: &Path,
    config_item: hlsl_snapshots::ConfigItem,
    fxc: &str,
) -> anyhow::Result<()> {
    let Some(Ok(shader_model_major_version)) = config_item
        .target_profile
        .split('_')
        .nth(1)
        .map(|segment| segment.parse::<u8>()) else {
            bail!(
                "expected target profile of the form \
                 `{{model}}_{{major}}_{{minor}}`, found invalid target \
                 profile {:?} in file {}",
                config_item.target_profile,
                file.display()
            )
        };
    // NOTE: This isn't implemented by `fxc.exe`; see
    // <https://learn.microsoft.com/en-us/windows/win32/direct3dtools/dx-graphics-tools-fxc-syntax#profiles>.
    if shader_model_major_version < 6 {
        // Reference:
        // <https://learn.microsoft.com/en-us/windows/win32/direct3dtools/dx-graphics-tools-fxc-syntax>.
        validate_hlsl(file, fxc, config_item, &["-Zi", "-Od"])
    } else {
        log::debug!(
            "skipping config. item {config_item:?} because the \
             shader model major version is > 6"
        );
        Ok(())
    }
}

fn validate_hlsl(
    file: &Path,
    bin: &str,
    config_item: hlsl_snapshots::ConfigItem,
    params: &[&str],
) -> anyhow::Result<()> {
    let hlsl_snapshots::ConfigItem {
        entry_point,
        target_profile,
    } = config_item;
    EasyCommand::new(bin, |cmd| {
        cmd.arg(file)
            .arg("-T")
            .arg(&target_profile)
            .arg("-E")
            .arg(&entry_point)
            .args(params)
            .stdout(Stdio::null())
    })
    .success()
    .with_context(|| {
        format!(
            "failed to validate entry point {entry_point:?} with profile \
                 {target_profile:?}"
        )
    })
}
