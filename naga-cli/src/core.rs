//! Pure translation core: parse, validate, and emit output.

use crate::cli::{Args, InputKind, OutputFormat};
use crate::error::CliError;
use crate::output::{
    glsl_parse_errors_to_diagnostics, spv_error_to_diagnostic, validation_error_to_diagnostic,
    wgsl_parse_error_to_diagnostic, Diagnostic, JsonOutput, Reflection,
};
use crate::params::Parameters;
use anyhow::{anyhow, Context as _};
use naga::compact::KeepUnused;
use std::fs;
use std::io::Read as _;
use std::path::Path;

pub struct Parsed {
    pub module: naga::Module,
    pub input_text: Option<String>,
    pub language: naga::back::spv::SourceLanguage,
}

/// Run the naga-cli pipeline.
///
/// Returns `Ok(true)` on success, `Ok(false)` when a handled failure has already
/// been emitted (JSON to stdout in json mode, or stderr diagnostics in text mode),
/// and `Err(e)` for hard/unexpected errors.
pub fn run(args: &Args, params: &mut Parameters) -> anyhow::Result<bool> {
    if args.bulk_validate {
        // bulk_validate is text-only for v1; json mode is not supported there.
        bulk_validate(&args.files, params)?;
        return Ok(true);
    }

    let is_json = args.format == OutputFormat::Json;

    let mut files = args.files.iter();

    let (input_path, input) = if let Some(path) = args.stdin_file_path.as_ref() {
        let mut input = vec![];
        std::io::stdin().lock().read_to_end(&mut input)?;
        (Path::new(path), input)
    } else if let Some(path) = files.next() {
        let path = Path::new(path);
        (path, fs::read(path)?)
    } else {
        return Err(CliError("Input file path is not specified").into());
    };

    let file_name = input_path.to_string_lossy().into_owned();

    // In JSON mode we collect diagnostics and a success flag instead of
    // short-circuiting on errors.
    let mut json_diagnostics: Vec<Diagnostic> = Vec::new();
    let mut json_reflection: Option<Reflection> = None;

    // --- Parse step ---
    // In JSON mode, parse_input_json returns typed parse failures as diagnostics.
    // In text mode, we propagate the anyhow error upward normally.
    let parsed = if is_json {
        match parse_input_json(input_path, input, params) {
            Ok(p) => Some(p),
            Err(diags) => {
                json_diagnostics.extend(diags);
                None
            }
        }
    } else {
        Some(parse_input(input_path, input, params)?)
    };

    // If parse failed in json mode, emit JSON and return.
    let Some(Parsed {
        mut module,
        input_text,
        language,
    }) = parsed
    else {
        let out = JsonOutput { success: false, diagnostics: json_diagnostics, reflection: None };
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(false);
    };

    // Include debugging information if requested.
    // We build a local copy of spv_out with debug_info set, so we can borrow
    // from input_text and file_name (which are locals) without requiring them
    // to outlive params (which may have a 'static lifetime parameter).
    let spv_out_with_debug: Option<naga::back::spv::Options<'_>>;
    if args.generate_debug_symbols {
        if let Some(ref input_text) = input_text {
            let mut opts = params.spv_out.clone();
            opts.flags.set(naga::back::spv::WriterFlags::DEBUG, true);
            opts.debug_info = Some(naga::back::spv::DebugInfo {
                source_code: input_text,
                file_name: &file_name,
                language,
            });
            spv_out_with_debug = Some(opts);
        } else {
            eprintln!(
                "warning: `--generate-debug-symbols` was passed, \
                       but input is not human-readable: {}",
                input_path.display()
            );
            spv_out_with_debug = None;
        }
    } else {
        spv_out_with_debug = None;
    }

    let output_paths = files;

    // Decide which capabilities our output formats can support.
    let validation_caps = output_paths
        .clone()
        .fold(params.capabilities, |caps, path| {
            use naga::valid::Capabilities as C;
            let allowed = match Path::new(path).extension().and_then(|ex| ex.to_str()) {
                Some("wgsl") => naga::back::wgsl::supported_capabilities(),
                Some("metal") => naga::back::msl::supported_capabilities(),
                Some("hlsl") => naga::back::hlsl::supported_capabilities(),
                Some("spv") | Some("spirv") => naga::back::spv::supported_capabilities(),
                Some("glsl") | Some("frag") | Some("vert") | Some("comp") | Some("task")
                | Some("mesh") => naga::back::glsl::supported_capabilities(),
                _ => C::all() - C::TEXTURE_EXTERNAL,
            };
            caps & allowed
        });

    // Validate the IR before compaction.
    let info = match naga::valid::Validator::new(params.validation_flags, validation_caps)
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)
    {
        Ok(info) => Some(info),
        Err(error) => {
            if is_json {
                json_diagnostics
                    .push(validation_error_to_diagnostic(&error, input_text.as_deref()));
            } else {
                // Validation failure is not fatal. Just report the error.
                if let Some(input) = &input_text {
                    let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                    error.emit_to_stderr_with_path(input, filename.unwrap_or("input"));
                } else {
                    crate::error::print_err(&error);
                }
            }
            None
        }
    };

    // Compact the module, if requested.
    //
    // Note that when output is to a non-WGSL shader language, we will call
    // `process_overrides`, which does its own compaction even if it is not
    // explicitly requested on the command line.
    let info = if args.compact {
        // Compact only if validation succeeded. Otherwise, compaction may panic.
        if info.is_some() {
            // Write out the module state before compaction, if requested.
            if let Some(ref before_compaction) = args.before_compaction {
                write_output(&module, &info, params, spv_out_with_debug.as_ref(), before_compaction)?;
            }

            naga::compact::compact(&mut module, KeepUnused::No);

            // Re-validate the IR after compaction.
            match naga::valid::Validator::new(params.validation_flags, validation_caps)
                .validate(&module)
            {
                Ok(info) => Some(info),
                Err(error) => {
                    if is_json {
                        json_diagnostics.push(validation_error_to_diagnostic(
                            &error,
                            input_text.as_deref(),
                        ));
                    } else {
                        eprintln!("Error validating compacted module:");
                        if let Some(input) = &input_text {
                            let filename =
                                input_path.file_name().and_then(std::ffi::OsStr::to_str);
                            error.emit_to_stderr_with_path(input, filename.unwrap_or("input"));
                        } else {
                            crate::error::print_err(&error);
                        }
                    }
                    None
                }
            }
        } else {
            if !is_json {
                eprintln!("Skipping compaction due to validation failure.");
            }
            None
        }
    } else {
        info
    };

    // If no output was requested, report validation results and stop here.
    //
    // If the user asked for output, don't stop: some output formats (".txt",
    // ".dot", ".bin") can be generated even without a `ModuleInfo`.
    if output_paths.clone().next().is_none() {
        if is_json {
            let success = info.is_some();
            if success {
                json_reflection = Some(Reflection::from_module(&module));
            }
            let out = JsonOutput {
                success,
                diagnostics: json_diagnostics,
                reflection: json_reflection,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
            return Ok(success);
        } else if info.is_some() {
            println!("Validation successful");
            return Ok(true);
        } else {
            return Ok(false);
        }
    }

    // There are output paths; run them.
    if is_json && info.is_some() {
        json_reflection = Some(Reflection::from_module(&module));
    }

    for output_path in output_paths {
        write_output(&module, &info, params, spv_out_with_debug.as_ref(), output_path)?;
    }

    if is_json {
        let success = !json_diagnostics.iter().any(|d| matches!(d.severity, crate::output::Severity::Error));
        let out = JsonOutput {
            success,
            diagnostics: json_diagnostics,
            reflection: json_reflection,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(success);
    }

    Ok(true)
}

fn parse_input(input_path: &Path, input: Vec<u8>, params: &Parameters) -> anyhow::Result<Parsed> {
    let input_kind = match params.input_kind {
        Some(kind) => kind,
        None => input_path
            .extension()
            .context("Input filename has no extension")?
            .to_str()
            .context("Input filename not valid unicode")?
            .parse::<InputKind>()
            .map_err(|e| anyhow!("Unable to determine --input-kind from filename: {e}"))?,
    };

    Ok(match input_kind {
        InputKind::Bin => Parsed {
            module: bincode::serde::decode_from_slice(&input, bincode::config::standard())?.0,
            input_text: None,
            language: naga::back::spv::SourceLanguage::Unknown,
        },
        InputKind::Spv => Parsed {
            module: naga::front::spv::parse_u8_slice(&input, &params.spv_in)?,
            input_text: None,
            language: naga::back::spv::SourceLanguage::Unknown,
        },
        InputKind::Wgsl => {
            let input = String::from_utf8(input)?;
            let options = naga::front::wgsl::Options {
                parse_doc_comments: false,
                capabilities: params.capabilities,
            };
            let mut frontend = naga::front::wgsl::Frontend::new_with_options(options);
            let result = frontend.parse(&input);
            match result {
                Ok(v) => Parsed {
                    module: v,
                    input_text: Some(input),
                    language: naga::back::spv::SourceLanguage::WGSL,
                },
                Err(ref e) => {
                    let message = anyhow!(
                        "Could not parse WGSL:\n{}",
                        e.emit_to_string_with_path(&input, &input_path.display().to_string())
                    );
                    return Err(message);
                }
            }
        }
        InputKind::Glsl => {
            let shader_stage = match params.shader_stage {
                Some(stage) => stage.to_stage(),
                None => {
                    let file_stem = input_path
                        .file_stem()
                        .context("Unable to determine file stem from input filename.")?;
                    let inner_ext = Path::new(file_stem)
                        .extension()
                        .context("Unable to determine inner extension from input filename.")?
                        .to_str()
                        .context("Input filename not valid unicode")?;
                    match inner_ext {
                        "vert" => naga::ShaderStage::Vertex,
                        "frag" => naga::ShaderStage::Fragment,
                        "comp" => naga::ShaderStage::Compute,
                        other => return Err(anyhow!("Unknown GLSL stage extension: {other}")),
                    }
                }
            };
            let input = String::from_utf8(input)?;
            let mut parser = naga::front::glsl::Frontend::default();
            let module = parser
                .parse(
                    &naga::front::glsl::Options {
                        stage: shader_stage,
                        defines: params.defines.clone(),
                    },
                    &input,
                )
                .map_err(|error| {
                    let filename = input_path
                        .file_name()
                        .and_then(std::ffi::OsStr::to_str)
                        .unwrap_or("glsl");
                    anyhow!(
                        "Could not parse GLSL:\n{}",
                        error.emit_to_string_with_path(&input, filename)
                    )
                })?;
            Parsed {
                module,
                input_text: Some(input),
                language: naga::back::spv::SourceLanguage::GLSL,
            }
        }
    })
}

/// JSON-mode variant of `parse_input`: on parse failure, returns structured
/// diagnostics instead of an anyhow error. Hard errors (I/O, UTF-8) that
/// are not parse failures are still returned as `anyhow::Error` by using
/// a nested result; we convert them to a single diagnostic message here.
fn parse_input_json(
    input_path: &Path,
    input: Vec<u8>,
    params: &Parameters,
) -> Result<Parsed, Vec<Diagnostic>> {
    let input_kind = match params.input_kind {
        Some(kind) => kind,
        None => {
            // Extension detection errors are hard errors; convert to a single diagnostic.
            let result: anyhow::Result<InputKind> = (|| {
                input_path
                    .extension()
                    .context("Input filename has no extension")?
                    .to_str()
                    .context("Input filename not valid unicode")?
                    .parse::<InputKind>()
                    .map_err(|e| anyhow!("Unable to determine --input-kind from filename: {e}"))
            })();
            match result {
                Ok(k) => k,
                Err(e) => {
                    return Err(vec![Diagnostic {
                        severity: crate::output::Severity::Error,
                        message: e.to_string(),
                        location: None,
                        labels: Vec::new(),
                        notes: Vec::new(),
                    }]);
                }
            }
        }
    };

    Ok(match input_kind {
        InputKind::Bin => {
            let module = bincode::serde::decode_from_slice(&input, bincode::config::standard())
                .map_err(|e| {
                    vec![Diagnostic {
                        severity: crate::output::Severity::Error,
                        message: e.to_string(),
                        location: None,
                        labels: Vec::new(),
                        notes: Vec::new(),
                    }]
                })?
                .0;
            Parsed {
                module,
                input_text: None,
                language: naga::back::spv::SourceLanguage::Unknown,
            }
        }
        InputKind::Spv => {
            let module =
                naga::front::spv::parse_u8_slice(&input, &params.spv_in).map_err(|e| {
                    vec![spv_error_to_diagnostic(&e)]
                })?;
            Parsed {
                module,
                input_text: None,
                language: naga::back::spv::SourceLanguage::Unknown,
            }
        }
        InputKind::Wgsl => {
            let input = String::from_utf8(input).map_err(|e| {
                vec![Diagnostic {
                    severity: crate::output::Severity::Error,
                    message: e.to_string(),
                    location: None,
                    labels: Vec::new(),
                    notes: Vec::new(),
                }]
            })?;
            let options = naga::front::wgsl::Options {
                parse_doc_comments: false,
                capabilities: params.capabilities,
            };
            let mut frontend = naga::front::wgsl::Frontend::new_with_options(options);
            match frontend.parse(&input) {
                Ok(module) => Parsed {
                    module,
                    input_text: Some(input),
                    language: naga::back::spv::SourceLanguage::WGSL,
                },
                Err(ref e) => {
                    return Err(vec![wgsl_parse_error_to_diagnostic(e, &input)]);
                }
            }
        }
        InputKind::Glsl => {
            let shader_stage: Result<naga::ShaderStage, Vec<Diagnostic>> =
                match params.shader_stage {
                    Some(stage) => Ok(stage.to_stage()),
                    None => {
                        let result: anyhow::Result<naga::ShaderStage> = (|| {
                            let file_stem = input_path
                                .file_stem()
                                .context("Unable to determine file stem from input filename.")?;
                            let inner_ext = Path::new(file_stem)
                                .extension()
                                .context(
                                    "Unable to determine inner extension from input filename.",
                                )?
                                .to_str()
                                .context("Input filename not valid unicode")?;
                            Ok(match inner_ext {
                                "vert" => naga::ShaderStage::Vertex,
                                "frag" => naga::ShaderStage::Fragment,
                                "comp" => naga::ShaderStage::Compute,
                                other => {
                                    return Err(anyhow!("Unknown GLSL stage extension: {other}"))
                                }
                            })
                        })();
                        result.map_err(|e| {
                            vec![Diagnostic {
                                severity: crate::output::Severity::Error,
                                message: e.to_string(),
                                location: None,
                                labels: Vec::new(),
                                notes: Vec::new(),
                            }]
                        })
                    }
                };
            let shader_stage = shader_stage?;
            let input = String::from_utf8(input).map_err(|e| {
                vec![Diagnostic {
                    severity: crate::output::Severity::Error,
                    message: e.to_string(),
                    location: None,
                    labels: Vec::new(),
                    notes: Vec::new(),
                }]
            })?;
            let mut parser = naga::front::glsl::Frontend::default();
            match parser.parse(
                &naga::front::glsl::Options {
                    stage: shader_stage,
                    defines: params.defines.clone(),
                },
                &input,
            ) {
                Ok(module) => Parsed {
                    module,
                    input_text: Some(input),
                    language: naga::back::spv::SourceLanguage::GLSL,
                },
                Err(ref errors) => {
                    return Err(glsl_parse_errors_to_diagnostics(errors, &input));
                }
            }
        }
    })
}

fn write_output(
    module: &naga::Module,
    info: &Option<naga::valid::ModuleInfo>,
    params: &Parameters,
    spv_out_override: Option<&naga::back::spv::Options<'_>>,
    output_path: &str,
) -> anyhow::Result<()> {
    let entry_point = match params.entry_point.as_deref() {
        Some(name) => {
            let ep_index = module
                .entry_points
                .iter()
                .position(|ep| ep.name == *name)
                .ok_or_else(|| anyhow!("Unable to find the entry point: {name}"))?;
            Some((module.entry_points[ep_index].stage, name))
        }
        None => None,
    };

    match Path::new(&output_path)
        .extension()
        .ok_or(CliError("Output filename has no extension"))?
        .to_str()
        .ok_or(CliError("Output filename not valid unicode"))?
    {
        "txt" => {
            use std::io::Write;

            let mut file = fs::File::create(output_path)?;
            writeln!(file, "{module:#?}")?;
            if let Some(ref info) = *info {
                writeln!(file)?;
                writeln!(file, "{info:#?}")?;
            }
        }
        "bin" => {
            let mut file = fs::File::create(output_path)?;
            bincode::serde::encode_into_std_write(module, &mut file, bincode::config::standard())?;
        }
        "metal" => {
            use naga::back::msl;

            let mut options = params.msl.clone();
            options.bounds_check_policies = params.bounds_check_policies;

            let info = info.as_ref().ok_or(CliError(
                "Generating metal output requires validation to \
                 succeed, and it failed in a previous step",
            ))?;

            let (module, info) = naga::back::pipeline_constants::process_overrides(
                module,
                info,
                entry_point.filter(|_| params.compact),
                &params.overrides,
            )?;

            let pipeline_options = msl::PipelineOptions::default();
            let (msl, _) =
                msl::write_string(&module, &info, &options, &pipeline_options)?;
            fs::write(output_path, msl)?;
        }
        "spv" => {
            use naga::back::spv;

            let pipeline_options = entry_point.map(|(shader_stage, name)| spv::PipelineOptions {
                entry_point: name.to_owned(),
                shader_stage,
            });

            let info = info.as_ref().ok_or(CliError(
                "Generating SPIR-V output requires validation to \
                 succeed, and it failed in a previous step",
            ))?;

            let (module, info) = naga::back::pipeline_constants::process_overrides(
                module,
                info,
                entry_point.filter(|_| params.compact),
                &params.overrides,
            )?;

            let spv_opts = spv_out_override.unwrap_or(&params.spv_out);
            let spv = spv::write_vec(&module, &info, spv_opts, pipeline_options.as_ref())?;
            let bytes = spv
                .iter()
                .fold(Vec::with_capacity(spv.len() * 4), |mut v, w| {
                    v.extend_from_slice(&w.to_le_bytes());
                    v
                });

            fs::write(output_path, bytes.as_slice())?;
        }
        stage @ ("vert" | "frag" | "comp") => {
            use naga::back::glsl;

            let file_ext_stage = match stage {
                "vert" => naga::ShaderStage::Vertex,
                "frag" => naga::ShaderStage::Fragment,
                "comp" => naga::ShaderStage::Compute,
                _ => unreachable!(), // exhaustive: guarded by outer match on "vert" | "frag" | "comp"
            };

            let (ep_stage, ep_name) = match entry_point {
                Some((stage, name)) => {
                    if stage != file_ext_stage {
                        eprintln!(
                            "warning: the shader stage `{stage:?}` of the selected entry point \
                                `{name}` in the input file does not match the shader stage \
                                implied by the file name",
                        );
                    }
                    (stage, name.to_string())
                }
                _ => (file_ext_stage, "main".to_string()),
            };

            let pipeline_options = glsl::PipelineOptions {
                entry_point: ep_name,
                shader_stage: ep_stage,
                multiview: None,
            };

            let info = info.as_ref().ok_or(CliError(
                "Generating glsl output requires validation to \
                 succeed, and it failed in a previous step",
            ))?;

            let (module, info) = naga::back::pipeline_constants::process_overrides(
                module,
                info,
                entry_point.filter(|_| params.compact),
                &params.overrides,
            )?;

            let mut buffer = String::new();
            let mut writer = glsl::Writer::new(
                &mut buffer,
                &module,
                &info,
                &params.glsl,
                &pipeline_options,
                params.bounds_check_policies,
            )?;
            writer.write()?;
            fs::write(output_path, buffer)?;
        }
        "dot" => {
            use naga::back::dot;

            let output = dot::write(module, info.as_ref(), params.dot.clone())?;
            fs::write(output_path, output)?;
        }
        "hlsl" => {
            use naga::back::hlsl;

            let info = info.as_ref().ok_or(CliError(
                "Generating hlsl output requires validation to \
                 succeed, and it failed in a previous step",
            ))?;

            let (module, info) = naga::back::pipeline_constants::process_overrides(
                module,
                info,
                entry_point.filter(|_| params.compact),
                &params.overrides,
            )?;

            let mut buffer = String::new();
            let pipeline_options = Default::default();
            let mut writer = hlsl::Writer::new(&mut buffer, &params.hlsl, &pipeline_options);
            writer.write(&module, &info, None)?;
            fs::write(output_path, buffer)?;
        }
        "wgsl" => {
            use naga::back::wgsl;

            let wgsl = wgsl::write_string(
                module,
                info.as_ref().ok_or(CliError(
                    "Generating wgsl output requires validation to \
                     succeed, and it failed in a previous step",
                ))?,
                wgsl::WriterFlags::empty(),
            )?;
            fs::write(output_path, wgsl)?;
        }
        other => {
            println!("Unknown output extension: {other}");
        }
    }

    Ok(())
}

fn bulk_validate(files: &[String], params: &Parameters) -> anyhow::Result<()> {
    let mut invalid = vec![];
    for input_path in files {
        let path = Path::new(&input_path);
        let input = fs::read(path)?;

        let Parsed {
            module,
            input_text,
            language: _,
        } = match parse_input(path, input, params) {
            Ok(parsed) => parsed,
            Err(error) => {
                invalid.push(input_path.clone());
                eprintln!("Error validating {input_path}:");
                eprintln!("{error}");
                continue;
            }
        };

        let mut validator =
            naga::valid::Validator::new(params.validation_flags, params.capabilities);
        validator.subgroup_stages(naga::valid::ShaderStages::all());
        validator.subgroup_operations(naga::valid::SubgroupOperationSet::all());

        if let Err(error) = validator.validate(&module) {
            invalid.push(input_path.clone());
            eprintln!("Error validating {input_path}:");
            if let Some(input) = &input_text {
                let filename = path.file_name().and_then(std::ffi::OsStr::to_str);
                error.emit_to_stderr_with_path(input, filename.unwrap_or("input"));
            } else {
                crate::error::print_err(&error);
            }
        }
    }

    if !invalid.is_empty() {
        use std::fmt::Write;
        let mut formatted = String::new();
        writeln!(
            &mut formatted,
            "Validation failed for the following inputs:"
        )
        .unwrap(); // infallible: writeln! to an in-memory String never fails
        for path in invalid {
            writeln!(&mut formatted, "  {path}").unwrap(); // infallible: writeln! to an in-memory String never fails
        }
        return Err(anyhow!(formatted));
    }

    Ok(())
}
