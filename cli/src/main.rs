#![allow(clippy::manual_strip)]
#[allow(unused_imports)]
use std::fs;
use std::{error::Error, fmt, path::Path, str::FromStr};

/// Translate shaders to different formats
#[derive(argh::FromArgs, Debug, Clone)]
struct Args {
    /// validate the shader during translation
    #[argh(switch)]
    validate: bool,

    /// what policy to use for index bounds checking.
    ///
    /// May be `Restrict`, `ReadZeroSkipWrite`, or `UndefinedBehavior`
    #[argh(option)]
    index_bounds_check_policy: Option<IndexBoundsCheckPolicyArg>,

    /// directory to dump the SPIR-V flow dump to
    #[argh(option)]
    flow_dir: Option<String>,

    /// the shader entrypoint to use when compiling to GLSL
    #[argh(option)]
    entry_point: Option<String>,

    /// the shader profile to use, for example `es`, `core`, `es330`, if translating to GLSL
    #[argh(option)]
    profile: Option<GlslProfileArg>,

    /// the shader model to use if targeting HSLS
    ///
    /// May be `50`, 51`, or `60`
    #[argh(option)]
    shader_model: Option<ShaderModelArg>,

    /// the input file
    #[argh(positional)]
    input: String,

    /// the output file. If not specified, only validation will be performed
    #[argh(positional)]
    output: Vec<String>,
}

/// Newtype so we can implement [`FromStr`] for `IndexBoundsCheckPolicy`.
#[derive(Debug, Clone)]
struct IndexBoundsCheckPolicyArg(naga::back::IndexBoundsCheckPolicy);

impl FromStr for IndexBoundsCheckPolicyArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use naga::back::IndexBoundsCheckPolicy;
        Ok(Self(match s.to_lowercase().as_str() {
            "restrict" => IndexBoundsCheckPolicy::Restrict,
            "readzeroskipwrite" => IndexBoundsCheckPolicy::ReadZeroSkipWrite,
            "undefinedbehavior" => IndexBoundsCheckPolicy::UndefinedBehavior,
            _ => {
                return Err(format!(
                    "Invalid value for --index-bounds-check-policy: {}",
                    s
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
            _ => return Err(format!("Invalid value for --shader-model: {}", s)),
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
            Version::Embedded(s[2..].parse().unwrap_or(310))
        } else {
            return Err(format!("Unknown profile: {}", s));
        }))
    }
}

#[derive(Default)]
struct Parameters {
    validation_flags: naga::valid::ValidationFlags,
    index_bounds_check_policy: naga::back::IndexBoundsCheckPolicy,
    entry_point: Option<String>,
    spv_adjust_coordinate_space: bool,
    spv_flow_dump_prefix: Option<String>,
    spv: naga::back::spv::Options,
    msl: naga::back::msl::Options,
    glsl: naga::back::glsl::Options,
    hlsl: naga::back::hlsl::Options,
}

trait PrettyResult {
    type Target;
    fn unwrap_pretty(self) -> Self::Target;
}

fn print_err(error: &dyn Error) {
    eprint!("{}", error);

    let mut e = error.source();
    if e.is_some() {
        eprintln!(": ");
    } else {
        eprintln!();
    }

    while let Some(source) = e {
        eprintln!("\t{}", source);
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
    let input_path = Path::new(&args.input);
    let output_paths = args.output;

    // Update parameters from commandline arguments
    if args.validate {
        params.validation_flags = naga::valid::ValidationFlags::all();
    }
    if let Some(policy) = args.index_bounds_check_policy {
        params.index_bounds_check_policy = policy.0;
    }
    params.spv_flow_dump_prefix = args.flow_dir;
    params.entry_point = args.entry_point;
    if let Some(version) = args.profile {
        params.glsl.version = version.0;
    }
    if let Some(model) = args.shader_model {
        params.hlsl.shader_model = model.0;
    }

    let module = match Path::new(&input_path)
        .extension()
        .ok_or(CliError("Input filename has no extension"))?
        .to_str()
        .ok_or(CliError("Input filename not valid unicode"))?
    {
        "spv" => {
            let options = naga::front::spv::Options {
                adjust_coordinate_space: params.spv_adjust_coordinate_space,
                strict_capabilities: false,
                flow_graph_dump_prefix: params.spv_flow_dump_prefix.map(std::path::PathBuf::from),
            };
            let input = fs::read(input_path)?;
            naga::front::spv::parse_u8_slice(&input, &options)?
        }
        "wgsl" => {
            let input = fs::read_to_string(input_path)?;
            let result = naga::front::wgsl::parse_str(&input);
            match result {
                Ok(v) => v,
                Err(ref e) => {
                    e.emit_to_stderr(&input);
                    return Err(CliError("Could not parse WGSL").into());
                }
            }
        }
        ext @ "vert" | ext @ "frag" | ext @ "comp" => {
            let input = fs::read_to_string(input_path)?;
            let mut parser = naga::front::glsl::Parser::default();

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
                .unwrap_or_else(|err| {
                    let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                    emit_glsl_parser_error(err, filename.unwrap_or("glsl"), &input);
                    std::process::exit(1);
                })
        }
        _ => return Err(CliError("Unknown input file extension").into()),
    };

    // validate the IR
    let info = match naga::valid::Validator::new(
        params.validation_flags,
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    {
        Ok(info) => Some(info),
        Err(error) => {
            print_err(&error);
            None
        }
    };

    if output_paths.is_empty() {
        if info.is_some() {
            println!("Validation successful");
            return Ok(());
        } else {
            std::process::exit(!0);
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
                writeln!(file, "{:#?}", module)?;
                if let Some(ref info) = info {
                    writeln!(file)?;
                    writeln!(file, "{:#?}", info)?;
                }
            }
            "metal" => {
                use naga::back::msl;

                let pipeline_options = msl::PipelineOptions::default();
                let (msl, _) = msl::write_string(
                    &module,
                    info.as_ref().ok_or(CliError(
                        "Generating metal output requires validation to \
                        succeed, and it failed in a previous step",
                    ))?,
                    &params.msl,
                    &pipeline_options,
                )
                .unwrap_pretty();
                fs::write(output_path, msl)?;
            }
            "spv" => {
                use naga::back::spv;

                params.spv.index_bounds_check_policy = params.index_bounds_check_policy;

                let spv = spv::write_vec(
                    &module,
                    info.as_ref().ok_or(CliError(
                        "Generating SPIR-V output requires validation to \
                        succeed, and it failed in a previous step",
                    ))?,
                    &params.spv,
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
            stage @ "vert" | stage @ "frag" | stage @ "comp" => {
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
                )
                .unwrap_pretty();
                writer.write()?;
                fs::write(output_path, buffer)?;
            }
            "dot" => {
                use naga::back::dot;

                let output = dot::write(&module, info.as_ref())?;
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
                            "Generating hsls output requires validation to \
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
                )
                .unwrap_pretty();
                fs::write(output_path, wgsl)?;
            }
            other => {
                println!("Unknown output extension: {}", other);
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

pub fn emit_glsl_parser_error(err: naga::front::glsl::ParseError, filename: &str, source: &str) {
    let diagnostic = match err.kind.metadata() {
        Some(metadata) => Diagnostic::error()
            .with_message(err.kind.to_string())
            .with_labels(vec![Label::primary((), metadata.start..metadata.end)]),
        None => Diagnostic::error().with_message(err.kind.to_string()),
    };

    let files = SimpleFile::new(filename, source);
    let config = codespan_reporting::term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);
    term::emit(&mut writer.lock(), &config, &files, &diagnostic).expect("cannot write error");
}
