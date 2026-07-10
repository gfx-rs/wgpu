//! Command-line argument definitions.

use clap::{Parser, ValueEnum};

/// Translate shaders to different formats.
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Bitmask of the ValidationFlags to be used; use 0 to disable validation.
    #[arg(long)]
    pub validate: Option<u8>,

    /// Policy for index bounds checking of arrays, vectors, and matrices.
    #[arg(long)]
    pub index_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Bounds-check policy for arrays/vectors/matrices in `storage`/`uniform` globals.
    /// Defaults to the index bounds check policy.
    #[arg(long)]
    pub buffer_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Bounds-check policy for texture loads. Defaults to the index bounds check policy.
    #[arg(long)]
    pub image_load_bounds_check_policy: Option<BoundsCheckPolicyArg>,

    /// Directory to dump the SPIR-V block context dump to.
    #[arg(long)]
    pub block_ctx_dir: Option<String>,

    /// The shader entrypoint. With `--compact`, anything unreachable from it is dropped.
    #[arg(long)]
    pub entry_point: Option<String>,

    /// GLSL profile to target, e.g. `es`, `core`, `es330`.
    #[arg(long, value_parser = parse_glsl_profile)]
    pub profile: Option<GlslProfile>,

    /// HLSL shader model, e.g. `50`, `51`, `60`..`67`.
    #[arg(long, value_parser = parse_shader_model)]
    pub shader_model: Option<naga::back::hlsl::ShaderModel>,

    /// SPIR-V version, e.g. `1.0`, `1.4`.
    #[arg(long, value_parser = parse_spirv_version)]
    pub spirv_version: Option<(u8, u8)>,

    /// Shader stage; derived from the file extension if unspecified.
    #[arg(long)]
    pub shader_stage: Option<ShaderStageArg>,

    /// Kind of input: `glsl`, `wgsl`, `spv`, or `bin`.
    #[arg(long)]
    pub input_kind: Option<InputKind>,

    /// Metal language version, e.g. `1.0`, `1.1`, `1.2`.
    #[arg(long, value_parser = parse_metal_version)]
    pub metal_version: Option<(u8, u8)>,

    /// Disable coordinate-space conversions where the frontend/backend supports them.
    #[arg(long)]
    pub keep_coordinate_space: bool,

    /// In dot output, include only the control flow graph.
    #[arg(long)]
    pub dot_cfg_only: bool,

    /// Treat STDIN as if it were this file path (needed for extension-based detection).
    #[arg(long)]
    pub stdin_file_path: Option<String>,

    /// Generate debug symbols (spv-out only, for now).
    #[arg(short = 'g', long)]
    pub generate_debug_symbols: bool,

    /// Compact the module's IR and revalidate.
    #[arg(long)]
    pub compact: bool,

    /// Write the module's IR before compaction to the given file. Implies `--compact`.
    #[arg(long)]
    pub before_compaction: Option<String>,

    /// Bulk validation mode: all filenames are inputs to read and validate.
    #[arg(long)]
    pub bulk_validate: bool,

    /// Pipeline-constant override, of the form "foo=N,bar=M"; repeatable.
    #[arg(long = "override", value_parser = parse_overrides)]
    pub overrides: Vec<Overrides>,

    /// Preprocessor defines for the GLSL frontend, "KEY=VALUE"; repeatable.
    #[arg(short = 'D', long = "defines", value_parser = parse_defines)]
    pub defines: Vec<Defines>,

    /// Capabilities filter: comma-separated names, a numeric bitflags value, "none", or "all".
    #[arg(long, default_value = "all", value_parser = parse_capabilities)]
    pub capabilities: naga::valid::Capabilities,

    /// Mesh shader task dispatch limits, as "X,Y".
    #[arg(long, value_parser = parse_task_limits)]
    pub task_limits: Option<naga::back::TaskDispatchLimits>,

    /// Whether the mesh shader output should be validated.
    #[arg(long, default_value_t = true)]
    pub validate_mesh_output: bool,

    /// Input file (stdin if omitted), then output files. In bulk mode, all are inputs.
    pub files: Vec<String>,
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

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Bin,
    Glsl,
    Spv,
    Wgsl,
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
    let dot = s.find('.').ok_or_else(|| "Missing dot separator".to_owned())?;
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
        max_mesh_workgroups_per_dim: x.parse().map_err(|e: core::num::ParseIntError| e.to_string())?,
        max_mesh_workgroups_total: y.parse().map_err(|e: core::num::ParseIntError| e.to_string())?,
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
        assert_eq!(args.index_bounds_check_policy, Some(BoundsCheckPolicyArg::Restrict));
        assert_eq!(args.files, vec!["in.wgsl".to_string(), "out.spv".to_string()]);
    }

    #[test]
    fn parses_repeated_overrides() {
        let args =
            Args::try_parse_from(["naga", "--override", "a=1,b=2", "--override", "c=3", "x.wgsl"])
                .unwrap();
        let flat: Vec<_> = args.overrides.iter().flat_map(|o| o.pairs.clone()).collect();
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
