#![allow(clippy::manual_strip)]
#[allow(unused_imports)]
use std::fs;
use std::{error::Error, fmt, io::Read, path::Path, str::FromStr};

/// Translate shaders to different formats.
#[derive(argh::FromArgs, Debug, Clone)]
struct Args {
    /// bitmask of the ValidationFlags to be used, use 0 to disable validation
    #[argh(option)]
    validate: Option<u8>,

    /// what policy to use for index bounds checking for arrays, vectors, and
    /// matrices.
    ///
    /// May be `Restrict` (force all indices in-bounds), `ReadZeroSkipWrite`
    /// (out-of-bounds indices read zeros, and don't write at all), or
    /// `Unchecked` (generate the simplest code, and whatever happens, happens)
    ///
    /// `Unchecked` is the default.
    #[argh(option)]
    index_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// what policy to use for index bounds checking for arrays, vectors, and
    /// matrices, when they are stored in globals in the `storage` or `uniform`
    /// storage classes.
    ///
    /// Possible values are the same as for `index-bounds-check-policy`. If
    /// omitted, defaults to the index bounds check policy.
    #[argh(option)]
    buffer_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// what policy to use for texture bounds checking.
    ///
    /// Possible values are the same as for `index-bounds-check-policy`. If
    /// omitted, defaults to the index bounds check policy.
    #[argh(option)]
    image_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// directory to dump the SPIR-V block context dump to
    #[argh(option)]
    block_ctx_dir: Option<String>,

    /// the shader entrypoint to use when compiling to GLSL
    #[argh(option)]
    entry_point: Option<String>,

    /// the shader profile to use, for example `es`, `core`, `es330`, if translating to GLSL
    #[argh(option)]
    profile: Option<GlslProfileArg>,

    /// the shader model to use if targeting HLSL
    ///
    /// May be `50`, 51`, or `60`
    #[argh(option)]
    shader_model: Option<ShaderModelArg>,

    /// if the selected frontends/backends support coordinate space conversions,
    /// disable them
    #[argh(switch)]
    keep_coordinate_space: bool,

    /// in dot output, include only the control flow graph
    #[argh(switch)]
    dot_cfg_only: bool,

    /// specify file path to process STDIN as
    #[argh(option)]
    stdin_file_path: Option<String>,

    /// generate debug symbols, only works for spv-out for now
    #[argh(option, short = 'g')]
    generate_debug_symbols: Option<bool>,

    /// show version
    #[argh(switch)]
    version: bool,

    /// the input and output files.
    ///
    /// First positional argument is the input file. If not specified, the
    /// input will be read from stdin. In the case, --stdin-file-path must also
    /// be specified.
    ///
    /// The rest arguments are the output files. If not specified, only
    /// validation will be performed.
    #[argh(positional)]
    files: Vec<String>,
}

/// Newtype so we can implement [`FromStr`] for `BoundsCheckPolicy`.
#[derive(Debug, Clone, Copy)]
struct BoundsCheckPolicyArg(naga::proc::BoundsCheckPolicy);

impl FromStr for BoundsCheckPolicyArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use naga::proc::BoundsCheckPolicy;
        Ok(Self(match s.to_lowercase().as_str() {
            "restrict" => BoundsCheckPolicy::Restrict,
            "readzeroskipwrite" => BoundsCheckPolicy::ReadZeroSkipWrite,
            "unchecked" => BoundsCheckPolicy::Unchecked,
            _ => {
                return Err(format!(
                    "Invalid value for --index-bounds-check-policy: {s}"
                ))
            }
        }))
    }
}

/// Newtype so we can implement [`FromStr`] for `ShaderModel`.
#[derive(Debug, Clone)]
struct ShaderModelArg(naga::back::hlsl::ShaderModel);

impl FromStr for ShaderModelArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use naga::back::hlsl::ShaderModel;
        Ok(Self(match s.to_lowercase().as_str() {
            "50" => ShaderModel::V5_0,
            "51" => ShaderModel::V5_1,
            "60" => ShaderModel::V6_0,
            _ => return Err(format!("Invalid value for --shader-model: {s}")),
        }))
    }
}

/// Newtype so we can implement [`FromStr`] for [`naga::back::glsl::Version`].
#[derive(Clone, Debug)]
struct GlslProfileArg(naga::back::glsl::Version);

impl FromStr for GlslProfileArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use naga::back::glsl::Version;
        Ok(Self(if s.starts_with("core") {
            Version::Desktop(s[4..].parse().unwrap_or(330))
        } else if s.starts_with("es") {
            Version::new_gles(s[2..].parse().unwrap_or(310))
        } else {
            return Err(format!("Unknown profile: {s}"));
        }))
    }
}

#[derive(Default)]
struct Parameters<'a> {
    validation_flags: naga::valid::ValidationFlags,
    bounds_check_policies: naga::proc::BoundsCheckPolicies,
    entry_point: Option<String>,
    keep_coordinate_space: bool,
    spv_block_ctx_dump_prefix: Option<String>,
    spv: naga::back::spv::Options<'a>,
    msl: naga::back::msl::Options,
    glsl: naga::back::glsl::Options,
    hlsl: naga::back::hlsl::Options,
}

trait PrettyResult {
    type Target;
    fn unwrap_pretty(self) -> Self::Target;
}

fn print_err(error: &dyn Error) {
    eprint!("{error}");

    let mut e = error.source();
    if e.is_some() {
        eprintln!(": ");
    } else {
        eprintln!();
    }

    while let Some(source) = e {
        eprintln!("\t{source}");
        e = source.source();
    }
}

impl<T, E: Error> PrettyResult for Result<T, E> {
    type Target = T;
    fn unwrap_pretty(self) -> T {
        match self {
            Result::Ok(value) => value,
            Result::Err(error) => {
                print_err(&error);
                std::process::exit(1);
            }
        }
    }
}

fn main() {
    if let Err(e) = run() {
        print_err(e.as_ref());
        std::process::exit(1);
    }
}

/// Error type for the CLI
#[derive(Debug, Clone)]
struct CliError(&'static str);
impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for CliError {}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Initialize default parameters
    //TODO: read the parameters from RON?
    let mut params = Parameters::default();

    // Parse commandline arguments
    let args: Args = argh::from_env();
    if args.version {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let (input_path, input) = if let Some(path) = args.files.first() {
        let path = Path::new(path);
        (path, fs::read(path)?)
    } else if let Some(path) = &args.stdin_file_path {
        let mut input = vec![];
        std::io::stdin().lock().read_to_end(&mut input)?;
        (Path::new(path), input)
    } else {
        return Err(CliError("Input file path is not specified").into());
    };
    let output_paths = args.files.get(1..).unwrap_or(&[]);

    // Update parameters from commandline arguments
    if let Some(bits) = args.validate {
        params.validation_flags = naga::valid::ValidationFlags::from_bits(bits)
            .ok_or(CliError("Invalid validation flags"))?;
    }
    if let Some(policy) = args.index_bounds_check_policy {
        params.bounds_check_policies.index = policy.0;
    }
    params.bounds_check_policies.buffer = match args.buffer_bounds_check_policy {
        Some(arg) => arg.0,
        None => params.bounds_check_policies.index,
    };
    params.bounds_check_policies.image = match args.image_bounds_check_policy {
        Some(arg) => arg.0,
        None => params.bounds_check_policies.index,
    };
    params.spv_block_ctx_dump_prefix = args.block_ctx_dir;
    params.entry_point = args.entry_point;
    if let Some(version) = args.profile {
        params.glsl.version = version.0;
    }
    if let Some(model) = args.shader_model {
        params.hlsl.shader_model = model.0;
    }
    params.keep_coordinate_space = args.keep_coordinate_space;

    let (module, input_text) = match Path::new(&input_path)
        .extension()
        .ok_or(CliError("Input filename has no extension"))?
        .to_str()
        .ok_or(CliError("Input filename not valid unicode"))?
    {
        "bin" => (bincode::deserialize(&input)?, None),
        "spv" => {
            let options = naga::front::spv::Options {
                adjust_coordinate_space: !params.keep_coordinate_space,
                strict_capabilities: false,
                block_ctx_dump_prefix: params
                    .spv_block_ctx_dump_prefix
                    .map(std::path::PathBuf::from),
            };
            naga::front::spv::parse_u8_slice(&input, &options).map(|m| (m, None))?
        }
        "wgsl" => {
            let input = String::from_utf8(input)?;
            let result = naga::front::wgsl::parse_str(&input);
            match result {
                Ok(v) => (v, Some(input)),
                Err(ref e) => {
                    let path = input_path.to_string_lossy();
                    e.emit_to_stderr_with_path(&input, &path);
                    return Err(CliError("Could not parse WGSL").into());
                }
            }
        }
        ext @ ("vert" | "frag" | "comp") => {
            let input = String::from_utf8(input)?;
            let mut parser = naga::front::glsl::Frontend::default();

            (
                parser
                    .parse(
                        &naga::front::glsl::Options {
                            stage: match ext {
                                "vert" => naga::ShaderStage::Vertex,
                                "frag" => naga::ShaderStage::Fragment,
                                "comp" => naga::ShaderStage::Compute,
                                _ => unreachable!(),
                            },
                            defines: Default::default(),
                        },
                        &input,
                    )
                    .unwrap_or_else(|errors| {
                        let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                        emit_glsl_parser_error(errors, filename.unwrap_or("glsl"), &input);
                        std::process::exit(1);
                    }),
                Some(input),
            )
        }
        _ => return Err(CliError("Unknown input file extension").into()),
    };

    // Decide which capabilities our output formats can support.
    let validation_caps =
        output_paths
            .iter()
            .fold(naga::valid::Capabilities::all(), |caps, path| {
                use naga::valid::Capabilities as C;
                let missing = match Path::new(path).extension().and_then(|ex| ex.to_str()) {
                    Some("wgsl") => C::CLIP_DISTANCE | C::CULL_DISTANCE,
                    Some("metal") => C::CULL_DISTANCE,
                    _ => C::empty(),
                };
                caps & !missing
            });

    // validate the IR
    let info = match naga::valid::Validator::new(params.validation_flags, validation_caps)
        .validate(&module)
    {
        Ok(info) => Some(info),
        Err(error) => {
            if let Some(input) = &input_text {
                let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                emit_annotated_error(&error, filename.unwrap_or("input"), input);
            }
            print_err(&error);
            None
        }
    };

    if output_paths.is_empty() {
        if info.is_some() {
            println!("Validation successful");
            return Ok(());
        } else {
            std::process::exit(-1);
        }
    }

    for output_path in output_paths {
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
                if let Some(ref info) = info {
                    writeln!(file)?;
                    writeln!(file, "{info:#?}")?;
                }
            }
            "bin" => {
                let file = fs::File::create(output_path)?;
                bincode::serialize_into(file, &module)?;
            }
            "metal" => {
                use naga::back::msl;

                let mut options = params.msl.clone();
                options.bounds_check_policies = params.bounds_check_policies;

                let pipeline_options = msl::PipelineOptions::default();
                let (msl, _) = msl::write_string(
                    &module,
                    info.as_ref().ok_or(CliError(
                        "Generating metal output requires validation to \
                        succeed, and it failed in a previous step",
                    ))?,
                    &options,
                    &pipeline_options,
                )
                .unwrap_pretty();
                fs::write(output_path, msl)?;
            }
            "spv" => {
                use naga::back::spv;

                let pipeline_options_owned;
                let pipeline_options = match params.entry_point {
                    Some(ref name) => {
                        let ep_index = module
                            .entry_points
                            .iter()
                            .position(|ep| ep.name == *name)
                            .expect("Unable to find the entry point");
                        pipeline_options_owned = spv::PipelineOptions {
                            entry_point: name.clone(),
                            shader_stage: module.entry_points[ep_index].stage,
                        };
                        Some(&pipeline_options_owned)
                    }
                    None => None,
                };

                params.spv.bounds_check_policies = params.bounds_check_policies;

                //Insert Debug infos
                params.spv.debug_info = args.generate_debug_symbols.and_then(|debug| {
                    params.spv.flags.set(spv::WriterFlags::DEBUG, debug);

                    if debug {
                        Some(spv::DebugInfo {
                            source_code: input_text.as_ref()?,
                            file_name: input_path.file_name().and_then(std::ffi::OsStr::to_str)?,
                        })
                    } else {
                        None
                    }
                });

                params.spv.flags.set(
                    spv::WriterFlags::ADJUST_COORDINATE_SPACE,
                    !params.keep_coordinate_space,
                );

                let spv = spv::write_vec(
                    &module,
                    info.as_ref().ok_or(CliError(
                        "Generating SPIR-V output requires validation to \
                        succeed, and it failed in a previous step",
                    ))?,
                    &params.spv,
                    pipeline_options,
                )
                .unwrap_pretty();
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

                let pipeline_options = glsl::PipelineOptions {
                    entry_point: match params.entry_point {
                        Some(ref name) => name.clone(),
                        None => "main".to_string(),
                    },
                    shader_stage: match stage {
                        "vert" => naga::ShaderStage::Vertex,
                        "frag" => naga::ShaderStage::Fragment,
                        "comp" => naga::ShaderStage::Compute,
                        _ => unreachable!(),
                    },
                    multiview: None,
                };

                let mut buffer = String::new();
                let mut writer = glsl::Writer::new(
                    &mut buffer,
                    &module,
                    info.as_ref().ok_or(CliError(
                        "Generating glsl output requires validation to \
                        succeed, and it failed in a previous step",
                    ))?,
                    &params.glsl,
                    &pipeline_options,
                    params.bounds_check_policies,
                )
                .unwrap_pretty();
                writer.write()?;
                fs::write(output_path, buffer)?;
            }
            "dot" => {
                use naga::back::dot;

                let output = dot::write(
                    &module,
                    info.as_ref(),
                    naga::back::dot::Options {
                        cfg_only: args.dot_cfg_only,
                    },
                )?;
                fs::write(output_path, output)?;
            }
            "hlsl" => {
                use naga::back::hlsl;
                let mut buffer = String::new();
                let mut writer = hlsl::Writer::new(&mut buffer, &params.hlsl);
                writer
                    .write(
                        &module,
                        info.as_ref().ok_or(CliError(
                            "Generating hlsl output requires validation to \
                            succeed, and it failed in a previous step",
                        ))?,
                    )
                    .unwrap_pretty();
                fs::write(output_path, buffer)?;
            }
            "wgsl" => {
                use naga::back::wgsl;

                let wgsl = wgsl::write_string(
                    &module,
                    info.as_ref().ok_or(CliError(
                        "Generating wgsl output requires validation to \
                        succeed, and it failed in a previous step",
                    ))?,
                    wgsl::WriterFlags::empty(),
                )
                .unwrap_pretty();
                fs::write(output_path, wgsl)?;
            }
            other => {
                println!("Unknown output extension: {other}");
            }
        }
    }

    Ok(())
}

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use naga::WithSpan;

pub fn emit_glsl_parser_error(errors: Vec<naga::front::glsl::Error>, filename: &str, source: &str) {
    let files = SimpleFile::new(filename, source);
    let config = codespan_reporting::term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);

    for err in errors {
        let mut diagnostic = Diagnostic::error().with_message(err.kind.to_string());

        if let Some(range) = err.meta.to_range() {
            diagnostic = diagnostic.with_labels(vec![Label::primary((), range)]);
        }

        term::emit(&mut writer.lock(), &config, &files, &diagnostic).expect("cannot write error");
    }
}

pub fn emit_annotated_error<E: Error>(ann_err: &WithSpan<E>, filename: &str, source: &str) {
    let files = SimpleFile::new(filename, source);
    let config = codespan_reporting::term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);

    let diagnostic = Diagnostic::error().with_labels(
        ann_err
            .spans()
            .map(|(span, desc)| {
                Label::primary((), span.to_range().unwrap()).with_message(desc.to_owned())
            })
            .collect(),
    );

    term::emit(&mut writer.lock(), &config, &files, &diagnostic).expect("cannot write error");
}
