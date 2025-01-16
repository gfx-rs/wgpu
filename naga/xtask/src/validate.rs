use std::{
    io::{BufRead, BufReader},
    path::Path,
    process::Stdio,
};

use anyhow::{bail, Context};

use crate::{
    cli::{ValidateHlslCommand, ValidateSubcommand},
    fs::open_file,
    path::join_path,
    process::{which, EasyCommand},
};

type Job = Box<dyn FnOnce() -> anyhow::Result<()> + Send + std::panic::UnwindSafe + 'static>;

pub(crate) fn validate(cmd: ValidateSubcommand) -> anyhow::Result<()> {
    let mut jobs = vec![];
    collect_validation_jobs(&mut jobs, cmd)?;

    let progress_bar = indicatif::ProgressBar::new(jobs.len() as u64);

    let (tx_results, rx_results) = std::sync::mpsc::channel();
    let enqueuing_thread = std::thread::spawn(move || -> anyhow::Result<()> {
        for job in jobs {
            let tx_results = tx_results.clone();
            crate::jobserver::start_job_thread(move || {
                let result = match std::panic::catch_unwind(|| job()) {
                    Ok(result) => result,
                    Err(payload) => Err(match payload.downcast_ref::<&str>() {
                        Some(message) => {
                            anyhow::anyhow!("Validation job thread panicked: {}", message)
                        }
                        None => anyhow::anyhow!("Validation job thread panicked"),
                    }),
                };
                tx_results.send(result).unwrap();
            })?;
        }
        Ok(())
    });

    let mut all_good = true;
    for result in rx_results {
        if let Err(error) = result {
            all_good = false;
            progress_bar.suspend(|| {
                log::error!("{:#}", error);
            });
        }
        progress_bar.inc(1);
    }

    progress_bar.finish_and_clear();

    anyhow::ensure!(
        all_good,
        "failed to validate one or more files, see above output for more details"
    );

    if let Err(error) = enqueuing_thread.join().unwrap() {
        bail!("Error enqueuing jobs:\n{:#}", error);
    }

    Ok(())
}

fn collect_validation_jobs(jobs: &mut Vec<Job>, cmd: ValidateSubcommand) -> anyhow::Result<()> {
    let snapshots_base_out = join_path(["tests", "out"]);
    match cmd {
        ValidateSubcommand::Spirv => {
            let spirv_as = "spirv-as";
            which(spirv_as)?;

            let spirv_val = "spirv-val";
            which(spirv_val)?;

            push_job_for_each_file(snapshots_base_out, "spv/*.spvasm", jobs, |path| {
                validate_spirv(&path, spirv_as, spirv_val)
            });
        }
        ValidateSubcommand::Metal => {
            let xcrun = "xcrun";
            which(xcrun)?;
            push_job_for_each_file(snapshots_base_out, "msl/*.msl", jobs, |path| {
                validate_metal(&path, xcrun)
            });
        }
        ValidateSubcommand::Glsl => {
            let glslang_validator = "glslangValidator";
            which(glslang_validator)?;
            for (glob, type_arg) in [
                ("glsl/*.Vertex.glsl", "vert"),
                ("glsl/*.Fragment.glsl", "frag"),
                ("glsl/*.Compute.glsl", "comp"),
            ] {
                push_job_for_each_file(&snapshots_base_out, glob, jobs, |path| {
                    validate_glsl(&path, type_arg, glslang_validator)
                });
            }
        }
        ValidateSubcommand::Dot => {
            let dot = "dot";
            which(dot)?;
            push_job_for_each_file(snapshots_base_out, "dot/*.dot", jobs, |path| {
                validate_dot(&path, dot)
            });
        }
        ValidateSubcommand::Wgsl => {
            let mut paths = vec![];
            crate::glob::for_each_file(snapshots_base_out, "wgsl/*.wgsl", |path_result| {
                try_push_job(jobs, |_| {
                    paths.push(path_result?.to_owned());
                    Ok(())
                })
            });
            if !paths.is_empty() {
                jobs.push(Box::new(move || validate_wgsl(&paths)));
            }
        }
        ValidateSubcommand::Hlsl(cmd) => {
            let bin;
            let validator: fn(&Path, hlsl_snapshots::ConfigItem, &str) -> anyhow::Result<()>;
            match cmd {
                ValidateHlslCommand::Dxc => {
                    bin = "dxc";
                    which(bin)?;
                    validator = validate_hlsl_with_dxc;
                }
                ValidateHlslCommand::Fxc => {
                    bin = "fxc";
                    which(bin)?;
                    validator = validate_hlsl_with_fxc;
                }
            }

            crate::glob::for_each_file(snapshots_base_out, "hlsl/*.hlsl", |path_result| {
                try_push_job(jobs, |jobs| {
                    let path = path_result?;
                    push_job_for_each_hlsl_config_item(&path, bin, jobs, validator)?;
                    Ok(())
                })
            });
        }
        ValidateSubcommand::All => {
            for subcmd in ValidateSubcommand::all() {
                let should_validate = match &subcmd {
                    ValidateSubcommand::Wgsl
                    | ValidateSubcommand::Spirv
                    | ValidateSubcommand::Glsl
                    | ValidateSubcommand::Dot => true,
                    ValidateSubcommand::Metal => cfg!(target_vendor = "apple"),
                    // The FXC compiler is only available on Windows.
                    //
                    // The DXC compiler can be built and run on any platform,
                    // but they don't make Linux releases and it's not clear
                    // what Git commit actually works on Linux, so restrict
                    // that to Windows as well.
                    ValidateSubcommand::Hlsl(
                        ValidateHlslCommand::Dxc | ValidateHlslCommand::Fxc,
                    ) => cfg!(target_os = "windows"),
                    ValidateSubcommand::All => continue,
                };
                if should_validate {
                    collect_validation_jobs(jobs, subcmd)?;
                }
            }
        }
    };

    Ok(())
}

fn push_job_for_each_file(
    top_dir: impl AsRef<Path>,
    pattern: impl AsRef<Path>,
    jobs: &mut Vec<Job>,
    f: impl FnOnce(std::path::PathBuf) -> anyhow::Result<()>
        + Clone
        + Send
        + std::panic::UnwindSafe
        + 'static,
) {
    crate::glob::for_each_file(top_dir, pattern, move |path_result| {
        // Let each job closure stand on its own.
        let f = f.clone();
        jobs.push(Box::new(|| f(path_result?)));
    });
}

/// Call `f` to extend `jobs`, but if `f` itself fails, push a job that reports that.
fn try_push_job(jobs: &mut Vec<Job>, f: impl FnOnce(&mut Vec<Job>) -> anyhow::Result<()>) {
    if let Err(error) = f(jobs) {
        jobs.push(Box::new(|| Err(error)));
    }
}

fn validate_spirv(path: &Path, spirv_as: &str, spirv_val: &str) -> anyhow::Result<()> {
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
    let Some(version) = second_line
        .strip_prefix(expected_header_prefix)
        .map(str::trim)
    else {
        bail!("no {expected_header_prefix:?} header found in {path:?}");
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
    let error_message = || {
        format!(
            "Failed to validate {path:?}.
Note: Labels and line numbers will not match the input file.
      Use this command to view the corresponding spvasm:
      '{spirv_as} --target-env spv{version} {} -o - | spirv-dis'\n",
            path.display(),
        )
    };
    EasyCommand::new(spirv_val, |cmd| cmd.stdin(child.stdout.unwrap()))
        .success()
        .with_context(error_message)
}

fn validate_metal(path: &Path, xcrun: &str) -> anyhow::Result<()> {
    let first_line = {
        let mut file = BufReader::new(open_file(path)?);
        let mut buf = String::new();
        file.read_line(&mut buf)
            .with_context(|| format!("failed to read header from {path:?}"))?;
        buf
    };
    let expected_header_prefix = "// language: ";
    let Some(language) = first_line.strip_prefix(expected_header_prefix) else {
        bail!("no {expected_header_prefix:?} header found in {path:?}");
    };
    let language = language.strip_suffix('\n').unwrap_or(language);
    let std_arg = if language.starts_with("metal1") || language.starts_with("metal2") {
        format!("-std=macos-{language}")
    } else {
        format!("-std={language}")
    };
    let file = open_file(path)?;
    EasyCommand::new(xcrun, |cmd| {
        cmd.stdin(Stdio::from(file))
            .args(["-sdk", "macosx", "metal", "-mmacosx-version-min=10.11"])
            .arg(std_arg)
            .args(["-x", "metal", "-", "-o", "/dev/null"])
    })
    .success()
}

fn validate_glsl(path: &Path, type_arg: &str, glslang_validator: &str) -> anyhow::Result<()> {
    let file = open_file(path)?;
    EasyCommand::new(glslang_validator, |cmd| {
        cmd.stdin(Stdio::from(file))
            .args(["--stdin", "-S"])
            .arg(type_arg)
    })
    .success()
}

fn validate_dot(path: &Path, dot: &str) -> anyhow::Result<()> {
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

fn push_job_for_each_hlsl_config_item(
    path: &Path,
    bin: &str,
    jobs: &mut Vec<Job>,
    validator: impl FnMut(&Path, hlsl_snapshots::ConfigItem, &str) -> anyhow::Result<()>
        + Clone
        + Send
        + std::panic::UnwindSafe
        + 'static,
) -> anyhow::Result<()> {
    let hlsl_snapshots::Config {
        vertex,
        fragment,
        compute,
    } = hlsl_snapshots::Config::from_path(path.with_extension("ron"))?;
    for shader in [vertex, fragment, compute].into_iter().flatten() {
        // Let each job closure stand on its own.
        let mut validator = validator.clone();
        let path = path.to_owned();
        let bin = bin.to_owned();
        jobs.push(Box::new(move || validator(&path, shader, &bin)));
    }
    Ok(())
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
        .map(|segment| segment.parse::<u8>())
    else {
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
