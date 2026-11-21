//! Command-line argument definitions.

use clap::{ArgGroup, Parser, ValueEnum};

/// Expanded help shown by `naga --help` (the terse form is shown by `naga -h`).
const LONG_ABOUT: &str = "\
Translate and validate shaders between WGSL, SPIR-V, GLSL, MSL, HLSL, and DOT.

The input format is inferred from the input file extension (override with --input-kind); the
output format is inferred from the output file extension. Use `-` as the input and/or output
path to read from stdin / write to stdout — since `-` has no extension, give the format with
--input-kind / --output-kind (and --shader-stage for GLSL). With no output file, the shader is
only validated.

EXAMPLES
  # Validate a WGSL shader (no output file).
  naga shader.wgsl

  # Compile a single entry point to SPIR-V, dropping everything unreachable from it
  # (--compact) and without debug symbols (omit -g).
  naga shader.wgsl out.spv --entry-point main --compact

  # Translate WGSL to HLSL / MSL (format follows the output extension).
  naga shader.wgsl out.hlsl
  naga shader.wgsl out.metal

  # GLSL output picks the stage from the extension (.vert / .frag / .comp); it emits one
  # entry point (defaults to `main`, else pass --entry-point).
  naga shader.wgsl out.frag --entry-point fs_main

  # Read from stdin (`-`); --input-kind names the format.
  cat shader.wgsl | naga - --input-kind wgsl out.spv

  # Write to stdout instead of a file (`-` needs --output-kind for the format).
  naga shader.wgsl - --output-kind hlsl
  naga shader.wgsl - --output-kind spv > out.spv

  # Full pipe: stdin to stdout.
  cat shader.wgsl | naga - --input-kind wgsl - --output-kind spv > out.spv

  # Machine-readable diagnostics + reflection for tooling.
  naga shader.wgsl --format json

  # Print the JSON Schema for the --config document.
  naga --print-config-schema

OPTIONS: FLAGS vs CONFIG
  Options can be set as individual flags OR via a JSON config (--config <file> /
  --config-json <string>), never both: passing any flag except --color alongside --config is
  an error. In config mode only the positional input/output file paths are accepted on the
  command line; everything else lives in the JSON. Run `naga --print-config-schema` to see every config key
  and its type — every flag has a corresponding config key, so config is a complete single
  source. Reading stdin / writing stdout via `-` works in config mode too: set the format with
  the `input_kind` / `output_kind` (and `shader_stage`) config keys, since the matching flags
  are exclusive with --config.

STRUCTURED OUTPUT
  --format json emits one JSON document on stdout with `diagnostics`
  (severity/message/location/labels/notes) and `reflection` (entry points, resources,
  overrides) — intended for editor integrations and CI. --json-out <file> writes that document
  to a file instead, so the shader can go to stdout.

EXTERNAL TOOLS (found on PATH, or set with --spirv-val-path / --spirv-opt-path / --dxc-path)
  --spirv-val and --spirv-opt validate / optimize SPIR-V output; --dxc compiles each HLSL
  entry point to `<hlsl-stem>.<entry-point>.dxil`. With stdout (`-`) output, SPIR-V hooks
  run on a temporary file, and --dxc (which then requires --entry-point) writes the DXIL to
  stdout instead of the HLSL.

NOTES
  * Coordinate-space conversion is controlled by --keep-coordinate-space, not by spv_out.flags.
  * --task-limits and --validate-mesh-output fan out to all applicable backends; in a config,
    set the flat per-backend keys `task_dispatch_limits` / `mesh_shader_primitive_indices_clamp`.
  * --entry-point also limits MSL and HLSL output to that entry point.
  * SPIR-V input warns about unknown capabilities unless --spv-in-strict-capabilities is given
    (also when `spv_in` is set in a config without `strict_capabilities`).
  * The validator allows subgroup operations in all stages; restrict with the
    `subgroup_stages` / `subgroup_operations` config keys.
  * In --format json every diagnostic is `error` or `warning`, and SPIR-V parse errors have no
    source location.
  * `--zero-initialize-workgroup-memory native` is honored only by the SPIR-V backend; the other
    backends treat it as `polyfill`.";

const STYLES: clap::builder::Styles = {
    use clap::builder::styling::{AnsiColor, Effects};
    clap::builder::Styles::styled()
        .header(AnsiColor::Green.on_default().effects(Effects::BOLD))
        .usage(AnsiColor::Green.on_default().effects(Effects::BOLD))
        .literal(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
        .placeholder(AnsiColor::Cyan.on_default())
        .error(AnsiColor::Red.on_default().effects(Effects::BOLD))
        .valid(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
        .invalid(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
};

/// Translate shaders to different formats.
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = LONG_ABOUT, styles = STYLES)]
#[command(group(ArgGroup::new("options").multiple(true).conflicts_with("config_input").args([
    "fake_missing_bindings",
    "force_loop_bounding",
    "ray_query_initialization_tracking",
    "zero_initialize_workgroup_memory",
])))]
pub struct Args {
    /// Bitmask of the ValidationFlags to be used; use 0 to disable validation.
    #[arg(long, group = "options")]
    pub validate: Option<u8>,

    /// Policy for index bounds checking of arrays, vectors, and matrices.
    #[arg(long, group = "options")]
    pub index_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Bounds-check policy for arrays/vectors/matrices in `storage`/`uniform` globals.
    /// Defaults to the index bounds check policy.
    #[arg(long, group = "options")]
    pub buffer_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Bounds-check policy for texture loads. Defaults to the index bounds check policy.
    #[arg(long, group = "options")]
    pub image_load_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Debugging aid for naga developers: with SPIR-V input, write each function's
    /// block context to `<dir>/block_ctx.<Stage>-<name>.txt`.
    #[arg(long, hide = true, alias = "block-ctx-dir", group = "options")]
    pub spv_in_block_ctx_dir: Option<String>,

    /// The shader entrypoint. With `--compact`, anything unreachable from it is dropped.
    #[arg(long, group = "options")]
    pub entry_point: Option<String>,

    /// GLSL version to target (`core` = core330, `es` = es310, `webgl` = WebGL 2 / ES 3.00).
    #[arg(long, value_enum, alias = "profile", group = "options")]
    pub glsl_version: Option<GlslVersionArg>,

    /// HLSL shader model.
    #[arg(long, value_enum, alias = "shader-model", group = "options")]
    pub hlsl_shader_model: Option<HlslShaderModelArg>,

    /// HLSL: the pipeline renders to multisampled attachments (disables conservative depth).
    #[arg(long, group = "options")]
    pub hlsl_multisampled: bool,

    /// MSL: emit and force `point_size` (for vertex/mesh shaders drawing point primitives).
    #[arg(long, group = "options")]
    pub msl_force_point_size: bool,

    /// GLSL: number of views to render to with multiview.
    #[arg(long, group = "options")]
    pub glsl_multiview: Option<core::num::NonZeroU32>,

    /// WGSL output: always write explicit types instead of relying on inference.
    #[arg(long, group = "options")]
    pub wgsl_explicit_types: bool,

    /// SPIR-V input: reject modules declaring capabilities naga does not know, instead of warning.
    #[arg(long, group = "options")]
    pub spv_in_strict_capabilities: bool,

    /// SPIR-V version.
    #[arg(long, value_parser = version_parser(SPIRV_VERSIONS), group = "options")]
    pub spirv_version: Option<(u8, u8)>,

    /// Shader stage; derived from the file extension if unspecified. Selects the input
    /// stage for GLSL.
    #[arg(long, group = "options")]
    pub shader_stage: Option<ShaderStageArg>,

    /// Kind of input: `glsl`, `wgsl`, `spv`, or `bin` (overrides extension detection;
    /// required for stdin `-`).
    #[arg(long, group = "options")]
    pub input_kind: Option<InputKind>,

    /// Output format when writing to stdout (output path `-`), which has no extension
    /// to infer from. Ignored when the output is a regular file.
    #[arg(long, value_enum, group = "options")]
    pub output_kind: Option<OutputKind>,

    /// Metal Shading Language version.
    #[arg(long, value_parser = version_parser(METAL_VERSIONS), group = "options")]
    pub metal_version: Option<(u8, u8)>,

    /// Disable coordinate-space conversions where the frontend/backend supports them.
    #[arg(long, group = "options")]
    pub keep_coordinate_space: bool,

    /// In dot output, include only the control flow graph.
    #[arg(long, group = "options")]
    pub dot_cfg_only: bool,

    /// Generate debug symbols (spv-out only, for now).
    #[arg(short = 'g', long, group = "options")]
    pub generate_debug_symbols: bool,

    /// Compact the module's IR and revalidate.
    #[arg(long, group = "options")]
    pub compact: bool,

    /// Write the module's IR before compaction to the given file. Implies `--compact`.
    #[arg(long, group = "options")]
    pub before_compaction: Option<String>,

    /// Bulk validation mode: all filenames are inputs to read and validate.
    #[arg(long, group = "options")]
    pub bulk_validate: bool,

    /// Pipeline-constant override, of the form "foo=N,bar=M"; repeatable.
    #[arg(long = "override", value_parser = parse_overrides, group = "options")]
    pub overrides: Vec<Overrides>,

    /// Preprocessor defines for the GLSL frontend, "KEY=VALUE"; repeatable.
    #[arg(short = 'D', long = "defines", value_parser = parse_defines, group = "options")]
    pub defines: Vec<Defines>,

    /// Capabilities filter: comma-separated names, a numeric bitflags value, "none", or "all".
    #[arg(long, default_value = "all", value_parser = parse_capabilities, group = "options")]
    pub capabilities: naga::valid::Capabilities,

    /// Mesh shader task dispatch limits, as "X,Y".
    #[arg(long, value_parser = parse_task_limits, group = "options")]
    pub task_limits: Option<naga::back::TaskDispatchLimits>,

    /// Whether the mesh shader output should be validated.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, group = "options")]
    pub validate_mesh_output: bool,

    /// Options shared across the SPIR-V, MSL, and HLSL backends.
    #[command(flatten)]
    pub common: naga::back::CommonBackendOptions,

    /// How to zero-initialize workgroup memory (native | polyfill | none).
    #[arg(long, value_enum)]
    pub zero_initialize_workgroup_memory: Option<naga::back::ZeroInitializeWorkgroupMemoryMode>,

    /// Read all translation options from a JSON config file (mutually exclusive with option flags).
    #[arg(long, group = "config_input")]
    pub config: Option<String>,

    /// Read all translation options from an inline JSON string (mutually exclusive with option flags).
    #[arg(long, group = "config_input", conflicts_with = "config")]
    pub config_json: Option<String>,

    /// Print the JSON Schema for the config document and exit.
    #[arg(long)]
    pub print_config_schema: bool,

    /// After writing SPIR-V output, validate it with `spirv-val`.
    #[arg(long, group = "options")]
    pub spirv_val: bool,

    /// After writing SPIR-V output, optimize it in place with `spirv-opt -O`.
    #[arg(long, group = "options")]
    pub spirv_opt: bool,

    /// After writing HLSL output, compile each entry point to DXIL with `dxc`.
    #[arg(long, group = "options")]
    pub dxc: bool,

    /// Path to the `spirv-val` executable (default: search PATH).
    #[arg(long, requires = "spirv_val", group = "options")]
    pub spirv_val_path: Option<std::path::PathBuf>,

    /// Path to the `spirv-opt` executable (default: search PATH).
    #[arg(long, requires = "spirv_opt", group = "options")]
    pub spirv_opt_path: Option<std::path::PathBuf>,

    /// Path to the `dxc` executable (default: search PATH).
    #[arg(long, requires = "dxc", group = "options")]
    pub dxc_path: Option<std::path::PathBuf>,

    /// Output format for diagnostics and reflection: `text` (human, default) or `json`.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, group = "options")]
    pub format: OutputFormat,

    /// Write the `--format json` document to this file instead of stdout (implies
    /// `--format json`), so the shader itself can be written to stdout.
    #[arg(long, group = "options")]
    pub json_out: Option<String>,

    /// When to use colors in diagnostics and help. Allowed together with --config.
    #[arg(long, value_enum, default_value_t = ColorArg::Auto)]
    pub color: ColorArg,

    /// Input file (stdin if omitted), then output files. In bulk mode, all are inputs.
    pub files: Vec<String>,
}

#[derive(
    clap::ValueEnum,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorArg {
    #[default]
    Auto,
    Always,
    Never,
}

impl ColorArg {
    /// Find `--color` before clap parses, so clap's own help and errors honor it.
    pub fn from_raw_args(args: impl IntoIterator<Item = std::ffi::OsString>) -> Self {
        let mut args = args.into_iter();
        let mut value = None;
        while let Some(arg) = args.next() {
            let arg = arg.to_string_lossy().into_owned();
            if arg == "--" {
                break;
            } else if arg == "--color" {
                value = args.next().map(|v| v.to_string_lossy().into_owned());
            } else if let Some(v) = arg.strip_prefix("--color=") {
                value = Some(v.to_owned());
            }
        }
        value
            .and_then(|v| <Self as ValueEnum>::from_str(&v, true).ok())
            .unwrap_or_default()
    }

    pub fn to_clap(self) -> clap::ColorChoice {
        match self {
            Self::Auto => clap::ColorChoice::Auto,
            Self::Always => clap::ColorChoice::Always,
            Self::Never => clap::ColorChoice::Never,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsCheckPolicyArg {
    Restrict,
    ReadZeroSkipWrite,
    Unchecked,
}

impl BoundsCheckPolicyArg {
    pub fn to_policy(self) -> naga::proc::BoundsCheckPolicy {
        use naga::proc::BoundsCheckPolicy as P;
        match self {
            BoundsCheckPolicyArg::Restrict => P::Restrict,
            BoundsCheckPolicyArg::ReadZeroSkipWrite => P::ReadZeroSkipWrite,
            BoundsCheckPolicyArg::Unchecked => P::Unchecked,
        }
    }
}

#[derive(
    ValueEnum,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ShaderStageArg {
    Vert,
    Frag,
    Comp,
}

impl ShaderStageArg {
    pub fn to_stage(self) -> naga::ShaderStage {
        match self {
            ShaderStageArg::Vert => naga::ShaderStage::Vertex,
            ShaderStageArg::Frag => naga::ShaderStage::Fragment,
            ShaderStageArg::Comp => naga::ShaderStage::Compute,
        }
    }
}

#[derive(
    ValueEnum,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum InputKind {
    Bin,
    Glsl,
    Spv,
    Wgsl,
}

/// Output format for stdout writes, mapping to the same format keys the output-file
/// extension would select.
#[derive(
    ValueEnum,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum OutputKind {
    Wgsl,
    Spv,
    Hlsl,
    Metal,
    Dot,
    Txt,
    Bin,
    /// GLSL vertex shader.
    Vert,
    /// GLSL fragment shader.
    Frag,
    /// GLSL compute shader.
    Comp,
}

impl OutputKind {
    /// The output-file extension this format corresponds to (the key used by `write_output`).
    pub fn as_ext(self) -> &'static str {
        match self {
            OutputKind::Wgsl => "wgsl",
            OutputKind::Spv => "spv",
            OutputKind::Hlsl => "hlsl",
            OutputKind::Metal => "metal",
            OutputKind::Dot => "dot",
            OutputKind::Txt => "txt",
            OutputKind::Bin => "bin",
            OutputKind::Vert => "vert",
            OutputKind::Frag => "frag",
            OutputKind::Comp => "comp",
        }
    }
}

impl std::str::FromStr for InputKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "bin" => InputKind::Bin,
            "glsl" => InputKind::Glsl,
            "spv" => InputKind::Spv,
            "wgsl" => InputKind::Wgsl,
            _ => return Err(format!("Unknown input kind: {s}")),
        })
    }
}

/// Newtype wrapper so clap can collect repeated `--override` values.
#[derive(Clone, Debug)]
pub struct Overrides {
    pub pairs: Vec<(String, f64)>,
}

/// Newtype wrapper so clap can collect repeated `-D`/`--defines` values.
#[derive(Clone, Debug)]
pub struct Defines {
    pub pairs: Vec<(String, String)>,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlslVersionArg {
    Core140,
    Core150,
    #[value(alias = "core")]
    Core330,
    Core400,
    Core410,
    Core420,
    Core430,
    Core440,
    Core450,
    Core460,
    Es300,
    #[value(alias = "es")]
    Es310,
    Es320,
    Webgl,
}

impl GlslVersionArg {
    pub fn to_version(self) -> naga::back::glsl::Version {
        use naga::back::glsl::Version as V;
        match self {
            Self::Core140 => V::Desktop(140),
            Self::Core150 => V::Desktop(150),
            Self::Core330 => V::Desktop(330),
            Self::Core400 => V::Desktop(400),
            Self::Core410 => V::Desktop(410),
            Self::Core420 => V::Desktop(420),
            Self::Core430 => V::Desktop(430),
            Self::Core440 => V::Desktop(440),
            Self::Core450 => V::Desktop(450),
            Self::Core460 => V::Desktop(460),
            Self::Es300 => V::new_gles(300),
            Self::Es310 => V::new_gles(310),
            Self::Es320 => V::new_gles(320),
            Self::Webgl => V::Embedded {
                version: 300,
                is_webgl: true,
            },
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlslShaderModelArg {
    #[value(name = "50", alias = "5.0")]
    V5_0,
    #[value(name = "51", alias = "5.1")]
    V5_1,
    #[value(name = "60", alias = "6.0")]
    V6_0,
    #[value(name = "61", alias = "6.1")]
    V6_1,
    #[value(name = "62", alias = "6.2")]
    V6_2,
    #[value(name = "63", alias = "6.3")]
    V6_3,
    #[value(name = "64", alias = "6.4")]
    V6_4,
    #[value(name = "65", alias = "6.5")]
    V6_5,
    #[value(name = "66", alias = "6.6")]
    V6_6,
    #[value(name = "67", alias = "6.7")]
    V6_7,
    #[value(name = "68", alias = "6.8")]
    V6_8,
    #[value(name = "69", alias = "6.9")]
    V6_9,
}

impl HlslShaderModelArg {
    pub fn to_model(self) -> naga::back::hlsl::ShaderModel {
        use naga::back::hlsl::ShaderModel as M;
        match self {
            Self::V5_0 => M::V5_0,
            Self::V5_1 => M::V5_1,
            Self::V6_0 => M::V6_0,
            Self::V6_1 => M::V6_1,
            Self::V6_2 => M::V6_2,
            Self::V6_3 => M::V6_3,
            Self::V6_4 => M::V6_4,
            Self::V6_5 => M::V6_5,
            Self::V6_6 => M::V6_6,
            Self::V6_7 => M::V6_7,
            Self::V6_8 => M::V6_8,
            Self::V6_9 => M::V6_9,
        }
    }
}

const SPIRV_VERSIONS: &[&str] = &["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"];

const METAL_VERSIONS: &[&str] = &[
    "1.0", "1.1", "1.2", "2.0", "2.1", "2.2", "2.3", "2.4", "3.0", "3.1", "3.2", "4.0",
];

/// A parser for `MAJOR.MINOR` that only accepts (and lists in `--help`) the given versions.
fn version_parser(
    versions: &'static [&'static str],
) -> impl clap::builder::TypedValueParser<Value = (u8, u8)> {
    use clap::builder::TypedValueParser as _;
    clap::builder::PossibleValuesParser::new(versions).map(|s| {
        let (major, minor) = s.split_once('.').unwrap();
        (major.parse().unwrap(), minor.parse().unwrap())
    })
}

fn parse_overrides(s: &str) -> Result<Overrides, String> {
    let mut pairs = vec![];
    for pair in s.split(',') {
        let Some((name, value)) = pair.split_once('=') else {
            return Err(format!("value needs a `=`: {pair:?}"));
        };
        let value = value
            .trim()
            .parse::<f64>()
            .map_err(|err| format!("{err}: {value:?}"))?;
        pairs.push((name.trim().to_string(), value));
    }
    Ok(Overrides { pairs })
}

fn parse_defines(s: &str) -> Result<Defines, String> {
    let mut pairs = vec![];
    for pair in s.split(',') {
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        pairs.push((name.trim().to_string(), value.trim().to_string()));
    }
    Ok(Defines { pairs })
}

fn parse_capabilities(s: &str) -> Result<naga::valid::Capabilities, String> {
    use naga::valid::Capabilities;
    let s = s.to_uppercase();
    if s == "NONE" {
        Ok(Capabilities::empty())
    } else if s == "ALL" {
        Ok(Capabilities::all())
    } else if let Ok(bits) = s.parse::<u64>() {
        Capabilities::from_bits(bits)
            .ok_or_else(|| format!("Invalid capabilities bitflags value: {bits}"))
    } else {
        s.split(',').try_fold(Capabilities::empty(), |acc, name| {
            Capabilities::from_name(name.trim())
                .map(|cap| acc | cap)
                .ok_or_else(|| format!("Unknown capability {}", name.trim()))
        })
    }
}

fn parse_task_limits(s: &str) -> Result<naga::back::TaskDispatchLimits, String> {
    let (x, y) = s
        .split_once(',')
        .ok_or_else(|| format!("No comma present for --task-limits value: {s}"))?;
    Ok(naga::back::TaskDispatchLimits {
        max_mesh_workgroups_per_dim: x
            .parse()
            .map_err(|e: core::num::ParseIntError| e.to_string())?,
        max_mesh_workgroups_total: y
            .parse()
            .map_err(|e: core::num::ParseIntError| e.to_string())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_core_flags() {
        let args = Args::try_parse_from([
            "naga",
            "--validate",
            "1",
            "--entry-point",
            "main",
            "--index-bounds-check-policy",
            "restrict",
            "in.wgsl",
            "out.spv",
        ])
        .unwrap();
        assert_eq!(args.validate, Some(1));
        assert_eq!(args.entry_point.as_deref(), Some("main"));
        assert_eq!(
            args.index_bounds_check_policy,
            Some(BoundsCheckPolicyArg::Restrict)
        );
        assert_eq!(
            args.files,
            vec!["in.wgsl".to_string(), "out.spv".to_string()]
        );
    }

    #[test]
    fn parses_repeated_overrides() {
        let args = Args::try_parse_from([
            "naga",
            "--override",
            "a=1,b=2",
            "--override",
            "c=3",
            "x.wgsl",
        ])
        .unwrap();
        let flat: Vec<_> = args
            .overrides
            .iter()
            .flat_map(|o| o.pairs.clone())
            .collect();
        assert_eq!(
            flat,
            vec![
                ("a".to_string(), 1.0),
                ("b".to_string(), 2.0),
                ("c".to_string(), 3.0)
            ]
        );
    }
}
