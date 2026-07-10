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
  --config-json <string>), never both: passing ANY flag alongside --config is an error. In
  config mode only the positional input/output file paths are accepted on the command line;
  everything else lives in the JSON. Run `naga --print-config-schema` to see every config key
  and its type — every flag has a corresponding config key, so config is a complete single
  source. Reading stdin / writing stdout via `-` works in config mode too: set the format with
  the `input_kind` / `output_kind` (and `shader_stage`) config keys, since the matching flags
  are exclusive with --config.

STRUCTURED OUTPUT
  --format json emits one JSON document on stdout with `diagnostics`
  (severity/message/location/labels/notes) and `reflection` (entry points, resources,
  overrides) — intended for editor integrations and CI.

EXTERNAL TOOLS (must be on PATH)
  --spirv-val and --spirv-opt validate / optimize SPIR-V output; --dxc compiles each HLSL
  entry point to `<hlsl-stem>.<entry-point>.dxil`.

NOTES
  * Coordinate-space conversion is controlled by --keep-coordinate-space, not by spv_out.flags.
  * --task-limits and --validate-mesh-output fan out to all applicable backends; in a config,
    set the flat per-backend keys `task_dispatch_limits` / `mesh_shader_primitive_indices_clamp`.
  * --print-config-schema omits `spv_out.capabilities` (its type has no JsonSchema impl), but
    --config / --config-json still accept it.
  * In --format json every diagnostic is `error` or `warning`, and SPIR-V parse errors have no
    source location.
  * `--zero-initialize-workgroup-memory native` is honored only by the SPIR-V backend; the other
    backends treat it as `polyfill`.";

/// Translate shaders to different formats.
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = LONG_ABOUT)]
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

    /// Directory to dump the SPIR-V block context dump to.
    #[arg(long, group = "options")]
    pub block_ctx_dir: Option<String>,

    /// The shader entrypoint. With `--compact`, anything unreachable from it is dropped.
    #[arg(long, group = "options")]
    pub entry_point: Option<String>,

    /// GLSL profile to target, e.g. `es`, `core`, `es330`.
    #[arg(long, value_parser = parse_glsl_profile, group = "options")]
    pub profile: Option<GlslProfile>,

    /// HLSL shader model, e.g. `50`, `51`, `60`..`67`.
    #[arg(long, value_parser = parse_shader_model, group = "options")]
    pub shader_model: Option<naga::back::hlsl::ShaderModel>,

    /// SPIR-V version, e.g. `1.0`, `1.4`.
    #[arg(long, value_parser = parse_spirv_version, group = "options")]
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

    /// Metal language version, e.g. `1.0`, `1.1`, `1.2`.
    #[arg(long, value_parser = parse_metal_version, group = "options")]
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

    /// After writing SPIR-V output, validate it with `spirv-val` (must be on PATH).
    #[arg(long, group = "options")]
    pub spirv_val: bool,

    /// After writing SPIR-V output, optimize it in place with `spirv-opt -O` (must be on PATH).
    #[arg(long, group = "options")]
    pub spirv_opt: bool,

    /// After writing HLSL output, compile each entry point to DXIL with `dxc` (must be on PATH).
    #[arg(long, group = "options")]
    pub dxc: bool,

    /// Output format for diagnostics and reflection: `text` (human, default) or `json`.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text, group = "options")]
    pub format: OutputFormat,

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

/// Re-exported so `params.rs` can name the parsed GLSL version.
pub use naga::back::glsl::Version as GlslProfile;

fn parse_glsl_profile(s: &str) -> Result<GlslProfile, String> {
    use naga::back::glsl::Version;
    if let Some(rest) = s.strip_prefix("core") {
        Ok(Version::Desktop(rest.parse().unwrap_or(330)))
    } else if let Some(rest) = s.strip_prefix("es") {
        Ok(Version::new_gles(rest.parse().unwrap_or(310)))
    } else {
        Err(format!("Unknown profile: {s}"))
    }
}

fn parse_shader_model(s: &str) -> Result<naga::back::hlsl::ShaderModel, String> {
    use naga::back::hlsl::ShaderModel as M;
    Ok(match s.to_lowercase().as_str() {
        "50" => M::V5_0,
        "51" => M::V5_1,
        "60" => M::V6_0,
        "61" => M::V6_1,
        "62" => M::V6_2,
        "63" => M::V6_3,
        "64" => M::V6_4,
        "65" => M::V6_5,
        "66" => M::V6_6,
        "67" => M::V6_7,
        _ => return Err(format!("Invalid value for --shader-model: {s}")),
    })
}

fn parse_spirv_version(s: &str) -> Result<(u8, u8), String> {
    let dot = s
        .find('.')
        .ok_or_else(|| "Missing dot separator".to_owned())?;
    let major = s[..dot].parse::<u8>().map_err(|e| e.to_string())?;
    let minor = s[dot + 1..].parse::<u8>().map_err(|e| e.to_string())?;
    Ok((major, minor))
}

fn parse_metal_version(s: &str) -> Result<(u8, u8), String> {
    let mut iter = s.split('.');
    let next = |iter: &mut core::str::Split<char>| {
        iter.next()
            .ok_or_else(|| format!("Invalid value for --metal-version: {s}"))?
            .parse::<u8>()
            .map_err(|err| format!("Invalid value for --metal-version: '{s}': {err}"))
    };
    let major = next(&mut iter)?;
    let minor = next(&mut iter)?;
    Ok((major, minor))
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
