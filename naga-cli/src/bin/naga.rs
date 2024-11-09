#![allow(clippy::manual_strip)]
use anyhow::{anyhow, Context as _};
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

    /// what policy to use for texture loads bounds checking.
    ///
    /// Possible values are the same as for `index-bounds-check-policy`. If
    /// omitted, defaults to the index bounds check policy.
    #[argh(option)]
    image_load_bounds_check_policy: Option<BoundsCheckPolicyArg>,

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

    /// the shader stage, for example 'frag', 'vert', or 'compute'.
    /// if the shader stage is unspecified it will be derived from
    /// the file extension.
    #[argh(option)]
    shader_stage: Option<ShaderStage>,

    /// the kind of input, e.g. 'glsl', 'wgsl', 'spv', or 'bin'.
    #[argh(option)]
    input_kind: Option<InputKind>,

    /// the metal version to use, for example, 1.0, 1.1, 1.2, etc.
    #[argh(option)]
    metal_version: Option<MslVersionArg>,

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
    #[argh(switch, short = 'g')]
    generate_debug_symbols: bool,

    /// compact the module's IR and revalidate.
    ///
    /// Output files will reflect the compacted IR. If you want to see the IR as
    /// it was before compaction, use the `--before-compaction` option.
    #[argh(switch)]
    compact: bool,

    /// write the module's IR before compaction to the given file.
    ///
    /// This implies `--compact`. Like any other output file, the filename
    /// extension determines the form in which the module is written.
    #[argh(option)]
    before_compaction: Option<String>,

    /// bulk validation mode: all filenames are inputs to read and validate.
    #[argh(switch)]
    bulk_validate: bool,

    /// show version
    #[argh(switch)]
    version: bool,

    /// override value, of the form "foo=N,bar=M", repeatable
    #[argh(option, long = "override")]
    overrides: Vec<Overrides>,

    /// the input and output files.
    ///
    /// First positional argument is the input file. If not specified, the
    /// input will be read from stdin. In the case, --stdin-file-path must also
    /// be specified.
    ///
    /// The rest arguments are the output files. If not specified, only
    /// validation will be performed.
    ///
    /// In bulk validation mode, these are all input files to be validated.
    #[argh(positional)]
    files: Vec<String>,

    /// defines to be passed to the parser (only glsl is supported)
    #[argh(option, short = 'D')]
    defines: Vec<Defines>,
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
            "61" => ShaderModel::V6_1,
            "62" => ShaderModel::V6_2,
            "63" => ShaderModel::V6_3,
            "64" => ShaderModel::V6_4,
            "65" => ShaderModel::V6_5,
            "66" => ShaderModel::V6_6,
            "67" => ShaderModel::V6_7,
            _ => return Err(format!("Invalid value for --shader-model: {s}")),
        }))
    }
}

/// Newtype so we can implement [`FromStr`] for `ShaderSource`.
#[derive(Debug, Clone, Copy)]
struct ShaderStage(naga::ShaderStage);

impl FromStr for ShaderStage {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use naga::ShaderStage;
        Ok(Self(match s.to_lowercase().as_str() {
            "frag" | "fragment" => ShaderStage::Fragment,
            "comp" | "compute" => ShaderStage::Compute,
            "vert" | "vertex" => ShaderStage::Vertex,
            _ => return Err(anyhow!("Invalid shader stage: {s}")),
        }))
    }
}

/// Input kind/file extension mapping
#[derive(Debug, Clone, Copy)]
enum InputKind {
    Bincode,
    Glsl,
    SpirV,
    Wgsl,
}
impl FromStr for InputKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "bin" => InputKind::Bincode,
            "glsl" => InputKind::Glsl,
            "spv" => InputKind::SpirV,
            "wgsl" => InputKind::Wgsl,
            _ => return Err(anyhow!("Invalid value for --input-kind: {s}")),
        })
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

/// Newtype so we can implement [`FromStr`] for a Metal Language Version.
#[derive(Clone, Debug)]
struct MslVersionArg((u8, u8));

impl FromStr for MslVersionArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.split('.');

        let check_value = |iter: &mut core::str::Split<_>| {
            iter.next()
                .ok_or_else(|| format!("Invalid value for --metal-version: {s}"))?
                .parse::<u8>()
                .map_err(|err| format!("Invalid value for --metal-version: '{s}': {err}"))
        };

        let major = check_value(&mut iter)?;
        let minor = check_value(&mut iter)?;

        Ok(Self((major, minor)))
    }
}

#[derive(Clone, Debug)]
struct Overrides {
    pairs: Vec<(String, f64)>,
}

impl FromStr for Overrides {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut pairs = vec![];
        for pair in s.split(',') {
            let Some((name, value)) = pair.split_once('=') else {
                return Err(format!("value needs a `=`: {pair:?}"));
            };
            let value = f64::from_str(value.trim()).map_err(|err| format!("{err}: {value:?}"))?;
            pairs.push((name.trim().to_string(), value));
        }
        Ok(Overrides { pairs })
    }
}

#[derive(Clone, Debug)]
struct Defines {
    pairs: Vec<(String, String)>,
}

impl FromStr for Defines {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut pairs = vec![];
        for pair in s.split(',') {
            let (name, value) = match pair.split_once('=') {
                Some((name, value)) => (name, value),
                None => (pair, ""), // Default to an empty string if no '=' is found
            };
            pairs.push((name.trim().to_string(), value.trim().to_string()));
        }
        Ok(Defines { pairs })
    }
}

#[derive(Default)]
struct Parameters<'a> {
    validation_flags: naga::valid::ValidationFlags,
    bounds_check_policies: naga::proc::BoundsCheckPolicies,
    entry_point: Option<String>,
    keep_coordinate_space: bool,
    overrides: naga::back::PipelineConstants,
    spv_in: naga::front::spv::Options,
    spv_out: naga::back::spv::Options<'a>,
    dot: naga::back::dot::Options,
    msl: naga::back::msl::Options,
    glsl: naga::back::glsl::Options,
    hlsl: naga::back::hlsl::Options,
    input_kind: Option<InputKind>,
    shader_stage: Option<ShaderStage>,
    defines: FastHashMap<String, String>,
}

trait PrettyResult {
    type Target;
    fn unwrap_pretty(self) -> Self::Target;
}

#[cold]
#[inline(never)]
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

fn run() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    // Parse commandline arguments
    let args: Args = argh::from_env();
    if args.version {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // Initialize default parameters
    //TODO: read the parameters from RON?
    let mut params = Parameters::default();

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
    params.bounds_check_policies.image_load = match args.image_load_bounds_check_policy {
        Some(arg) => arg.0,
        None => params.bounds_check_policies.index,
    };
    params.overrides = args
        .overrides
        .iter()
        .flat_map(|o| &o.pairs)
        .cloned()
        .collect();

    params.defines = args
        .defines
        .iter()
        .flat_map(|o| &o.pairs)
        .cloned()
        .collect();

    params.spv_in = naga::front::spv::Options {
        adjust_coordinate_space: !args.keep_coordinate_space,
        strict_capabilities: false,
        block_ctx_dump_prefix: args.block_ctx_dir.clone().map(std::path::PathBuf::from),
    };

    params.entry_point.clone_from(&args.entry_point);
    if let Some(ref version) = args.profile {
        params.glsl.version = version.0;
    }
    if let Some(ref model) = args.shader_model {
        params.hlsl.shader_model = model.0;
    }
    if let Some(ref version) = args.metal_version {
        params.msl.lang_version = version.0;
    }
    params.keep_coordinate_space = args.keep_coordinate_space;

    params.dot.cfg_only = args.dot_cfg_only;

    params.spv_out.bounds_check_policies = params.bounds_check_policies;
    params.spv_out.flags.set(
        naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE,
        !params.keep_coordinate_space,
    );

    if args.bulk_validate {
        return bulk_validate(args, &params);
    }

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

    params.input_kind = args.input_kind;
    params.shader_stage = args.shader_stage;

    let Parsed {
        mut module,
        input_text,
        language,
    } = parse_input(input_path, input, &params)?;

    // Include debugging information if requested.
    if args.generate_debug_symbols {
        if let Some(ref input_text) = input_text {
            params
                .spv_out
                .flags
                .set(naga::back::spv::WriterFlags::DEBUG, true);
            params.spv_out.debug_info = Some(naga::back::spv::DebugInfo {
                source_code: input_text,
                file_name: input_path,
                language,
            })
        } else {
            eprintln!(
                "warning: `--generate-debug-symbols` was passed, \
                       but input is not human-readable: {}",
                input_path.display()
            );
        }
    }

    let output_paths = files;

    // Decide which capabilities our output formats can support.
    let validation_caps =
        output_paths
            .clone()
            .fold(naga::valid::Capabilities::all(), |caps, path| {
                use naga::valid::Capabilities as C;
                let missing = match Path::new(path).extension().and_then(|ex| ex.to_str()) {
                    Some("wgsl") => C::CLIP_DISTANCE | C::CULL_DISTANCE,
                    Some("metal") => C::CULL_DISTANCE,
                    _ => C::empty(),
                };
                caps & !missing
            });

    // Validate the IR before compaction.
    let info = match naga::valid::Validator::new(params.validation_flags, validation_caps)
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)
    {
        Ok(info) => Some(info),
        Err(error) => {
            // Validation failure is not fatal. Just report the error.
            if let Some(input) = &input_text {
                let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                error.emit_to_stderr_with_path(input, filename.unwrap_or("input"));
            } else {
                print_err(&error);
            }
            None
        }
    };

    // Compact the module, if requested.
    let info = if args.compact || args.before_compaction.is_some() {
        // Compact only if validation succeeded. Otherwise, compaction may panic.
        if info.is_some() {
            // Write out the module state before compaction, if requested.
            if let Some(ref before_compaction) = args.before_compaction {
                write_output(&module, &info, &params, before_compaction)?;
            }

            naga::compact::compact(&mut module);

            // Re-validate the IR after compaction.
            match naga::valid::Validator::new(params.validation_flags, validation_caps)
                .validate(&module)
            {
                Ok(info) => Some(info),
                Err(error) => {
                    // Validation failure is not fatal. Just report the error.
                    eprintln!("Error validating compacted module:");
                    if let Some(input) = &input_text {
                        let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                        error.emit_to_stderr_with_path(input, filename.unwrap_or("input"));
                    } else {
                        print_err(&error);
                    }
                    None
                }
            }
        } else {
            eprintln!("Skipping compaction due to validation failure.");
            None
        }
    } else {
        info
    };

    // If no output was requested, then report validation results and stop here.
    //
    // If the user asked for output, don't stop: some output formats (".txt",
    // ".dot", ".bin") can be generated even without a `ModuleInfo`.
    if output_paths.clone().next().is_none() {
        if info.is_some() {
            println!("Validation successful");
            return Ok(());
        } else {
            std::process::exit(-1);
        }
    }

    for output_path in output_paths {
        write_output(&module, &info, &params, output_path)?;
    }

    Ok(())
}

struct Parsed {
    module: naga::Module,
    input_text: Option<String>,
    language: naga::back::spv::SourceLanguage,
}

fn parse_input(input_path: &Path, input: Vec<u8>, params: &Parameters) -> anyhow::Result<Parsed> {
    let input_kind = match params.input_kind {
        Some(kind) => kind,
        None => input_path
            .extension()
            .context("Input filename has no extension")?
            .to_str()
            .context("Input filename not valid unicode")?
            .parse()
            .context("Unable to determine --input-kind from filename")?,
    };

    Ok(match input_kind {
        InputKind::Bincode => Parsed {
            module: bincode::deserialize(&input)?,
            input_text: None,
            language: naga::back::spv::SourceLanguage::Unknown,
        },
        InputKind::SpirV => Parsed {
            module: naga::front::spv::parse_u8_slice(&input, &params.spv_in)?,
            input_text: None,
            language: naga::back::spv::SourceLanguage::Unknown,
        },
        InputKind::Wgsl => {
            let input = String::from_utf8(input)?;
            let result = naga::front::wgsl::parse_str(&input);
            match result {
                Ok(v) => Parsed {
                    module: v,
                    input_text: Some(input),
                    language: naga::back::spv::SourceLanguage::WGSL,
                },
                Err(ref e) => {
                    let message = anyhow!(
                        "Could not parse WGSL:\n{}",
                        e.emit_to_string_with_path(&input, input_path)
                    );
                    return Err(message);
                }
            }
        }
        InputKind::Glsl => {
            let shader_stage = match params.shader_stage {
                Some(shader_stage) => shader_stage,
                None => {
                    // filename.shader_stage.glsl -> filename.shader_stage
                    let file_stem = input_path
                        .file_stem()
                        .context("Unable to determine file stem from input filename.")?;
                    // filename.shader_stage -> shader_stage
                    let inner_ext = Path::new(file_stem)
                        .extension()
                        .context("Unable to determine inner extension from input filename.")?
                        .to_str()
                        .context("Input filename not valid unicode")?;
                    inner_ext.parse().context("from input filename")?
                }
            };
            let input = String::from_utf8(input)?;
            let mut parser = naga::front::glsl::Frontend::default();
            Parsed {
                module: parser
                    .parse(
                        &naga::front::glsl::Options {
                            stage: shader_stage.0,
                            defines: params.defines.clone(),
                        },
                        &input,
                    )
                    .unwrap_or_else(|error| {
                        let filename = input_path
                            .file_name()
                            .and_then(std::ffi::OsStr::to_str)
                            .unwrap_or("glsl");
                        let mut writer = StandardStream::stderr(ColorChoice::Auto);
                        error.emit_to_writer_with_path(&mut writer, &input, filename);
                        std::process::exit(1);
                    }),
                input_text: Some(input),
                language: naga::back::spv::SourceLanguage::GLSL,
            }
        }
    })
}

fn write_output(
    module: &naga::Module,
    info: &Option<naga::valid::ModuleInfo>,
    params: &Parameters,
    output_path: &str,
) -> anyhow::Result<()> {
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
            let file = fs::File::create(output_path)?;
            bincode::serialize_into(file, module)?;
        }
        "metal" => {
            use naga::back::msl;

            let mut options = params.msl.clone();
            options.bounds_check_policies = params.bounds_check_policies;

            let info = info.as_ref().ok_or(CliError(
                "Generating metal output requires validation to \
                 succeed, and it failed in a previous step",
            ))?;

            let (module, info) =
                naga::back::pipeline_constants::process_overrides(module, info, &params.overrides)
                    .unwrap_pretty();

            let pipeline_options = msl::PipelineOptions::default();
            let (msl, _) =
                msl::write_string(&module, &info, &options, &pipeline_options).unwrap_pretty();
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

            let info = info.as_ref().ok_or(CliError(
                "Generating SPIR-V output requires validation to \
                 succeed, and it failed in a previous step",
            ))?;

            let (module, info) =
                naga::back::pipeline_constants::process_overrides(module, info, &params.overrides)
                    .unwrap_pretty();

            let spv =
                spv::write_vec(&module, &info, &params.spv_out, pipeline_options).unwrap_pretty();
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

            let info = info.as_ref().ok_or(CliError(
                "Generating glsl output requires validation to \
                 succeed, and it failed in a previous step",
            ))?;

            let (module, info) =
                naga::back::pipeline_constants::process_overrides(module, info, &params.overrides)
                    .unwrap_pretty();

            let mut buffer = String::new();
            let mut writer = glsl::Writer::new(
                &mut buffer,
                &module,
                &info,
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

            let output = dot::write(module, info.as_ref(), params.dot.clone())?;
            fs::write(output_path, output)?;
        }
        "hlsl" => {
            use naga::back::hlsl;

            let info = info.as_ref().ok_or(CliError(
                "Generating hlsl output requires validation to \
                 succeed, and it failed in a previous step",
            ))?;

            let (module, info) =
                naga::back::pipeline_constants::process_overrides(module, info, &params.overrides)
                    .unwrap_pretty();

            let mut buffer = String::new();
            let mut writer = hlsl::Writer::new(&mut buffer, &params.hlsl);
            writer.write(&module, &info, None).unwrap_pretty();
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
            )
            .unwrap_pretty();
            fs::write(output_path, wgsl)?;
        }
        other => {
            println!("Unknown output extension: {other}");
        }
    }

    Ok(())
}

fn bulk_validate(args: Args, params: &Parameters) -> anyhow::Result<()> {
    let mut invalid = vec![];
    for input_path in args.files {
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
            naga::valid::Validator::new(params.validation_flags, naga::valid::Capabilities::all());
        validator.subgroup_stages(naga::valid::ShaderStages::all());
        validator.subgroup_operations(naga::valid::SubgroupOperationSet::all());

        if let Err(error) = validator.validate(&module) {
            invalid.push(input_path.clone());
            eprintln!("Error validating {input_path}:");
            if let Some(input) = &input_text {
                let filename = path.file_name().and_then(std::ffi::OsStr::to_str);
                error.emit_to_stderr_with_path(input, filename.unwrap_or("input"));
            } else {
                print_err(&error);
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
        .unwrap();
        for path in invalid {
            writeln!(&mut formatted, "  {path}").unwrap();
        }
        return Err(anyhow!(formatted));
    }

    Ok(())
}

use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use naga::FastHashMap;
