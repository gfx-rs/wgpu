//! JSON config struct for naga-cli: a serde-deserializable mirror of the
//! translation options, loadable via `--config <path>` or `--config-json <string>`.

use std::collections::BTreeMap;

/// All translation options that can be set via `--config` / `--config-json`.
///
/// Every field is `Option<…>` so that partial configs ("just set `spv_out`")
/// work without specifying everything else. Unknown JSON keys are rejected
/// (`deny_unknown_fields`) to catch typos early.
#[derive(Debug, Default, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Bitmask of [`naga::valid::ValidationFlags`]. `0` disables validation.
    pub validate: Option<u8>,

    /// Capabilities filter passed to the validator and WGSL frontend.
    pub capabilities: Option<naga::valid::Capabilities>,

    /// Bounds-check policies for generated code.
    pub bounds_check_policies: Option<naga::proc::BoundsCheckPolicies>,

    /// The shader entry point to target.
    pub entry_point: Option<String>,

    /// If `true`, suppress coordinate-space conversions in frontends/backends.
    pub keep_coordinate_space: Option<bool>,

    /// Preprocessor defines for the GLSL frontend (`KEY → VALUE`).
    pub defines: Option<BTreeMap<String, String>>,

    /// Pipeline-constant overrides (`name → value`).
    pub overrides: Option<BTreeMap<String, f64>>,

    /// Options for the SPIR-V frontend.
    pub spv_in: Option<naga::front::spv::Options>,

    /// Options for the SPIR-V backend.
    pub spv_out: Option<naga::back::spv::Options<'static>>,

    /// Options for the MSL backend.
    pub msl: Option<naga::back::msl::Options>,

    /// Options for the GLSL backend.
    pub glsl_out: Option<naga::back::glsl::Options>,

    /// Options for the HLSL backend.
    pub hlsl: Option<naga::back::hlsl::Options>,

    /// HLSL: the pipeline renders to multisampled attachments (disables conservative depth).
    pub hlsl_multisampled: Option<bool>,

    /// Per-pipeline MSL options.
    pub msl_pipeline: Option<MslPipeline>,

    /// GLSL: number of views to render to with multiview.
    pub glsl_multiview: Option<core::num::NonZeroU32>,

    /// WGSL output: always write explicit types instead of relying on inference.
    pub wgsl_explicit_types: Option<bool>,

    /// Shader stages in which the validator allows subgroup operations, as bitflag names
    /// (e.g. `"COMPUTE | FRAGMENT"`). Defaults to all stages.
    #[schemars(with = "Option<String>")]
    pub subgroup_stages: Option<naga::valid::ShaderStages>,

    /// Subgroup operations the validator allows, as bitflag names (e.g. `"BASIC | VOTE"`).
    /// Defaults to all operations.
    #[schemars(with = "Option<String>")]
    pub subgroup_operations: Option<naga::valid::SubgroupOperationSet>,

    /// Options for the dot backend.
    pub dot: Option<naga::back::dot::Options>,

    /// Compact the module's IR and revalidate before output.
    pub compact: Option<bool>,

    /// Generate debug symbols (spv-out only, for now).
    pub generate_debug_symbols: Option<bool>,

    /// After writing SPIR-V output, validate it with `spirv-val`.
    pub spirv_val: Option<bool>,

    /// After writing SPIR-V output, optimize it in place with `spirv-opt`.
    pub spirv_opt: Option<bool>,

    /// After writing HLSL output, compile each entry point to DXIL with `dxc`.
    pub dxc: Option<bool>,

    /// Path to the `spirv-val` executable (default: search PATH).
    pub spirv_val_path: Option<std::path::PathBuf>,

    /// Path to the `spirv-opt` executable (default: search PATH).
    pub spirv_opt_path: Option<std::path::PathBuf>,

    /// Path to the `dxc` executable (default: search PATH).
    pub dxc_path: Option<std::path::PathBuf>,

    /// Input format (overrides extension detection; required to read stdin `-`).
    pub input_kind: Option<crate::cli::InputKind>,

    /// Output format when writing to stdout `-` (which has no extension to infer from).
    pub output_kind: Option<crate::cli::OutputKind>,

    /// Shader stage for GLSL input.
    pub shader_stage: Option<crate::cli::ShaderStageArg>,

    /// Output format for diagnostics and reflection: `text` or `json`.
    pub format: Option<crate::cli::OutputFormat>,

    /// Write the JSON document to this file instead of stdout (implies `format: "json"`).
    pub json_out: Option<String>,

    /// Write the module's IR before compaction to this path (implies compaction).
    pub before_compaction: Option<String>,

    /// Bulk-validation mode: treat all positional files as inputs to validate.
    pub bulk_validate: Option<bool>,
}

/// The per-pipeline MSL options naga-cli can set. Mirrors the corresponding fields of
/// `naga::back::msl::PipelineOptions`; its `entry_point` comes from `entry_point`.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct MslPipeline {
    /// Allow `BuiltIn::PointSize` and inject it if it doesn't exist.
    pub allow_and_force_point_size: bool,
    /// Receive vertex buffers as arguments and fetch attributes manually ("vertex pulling").
    pub vertex_pulling_transform: bool,
    /// Vertex buffer layouts used by the vertex pulling transform.
    pub vertex_buffer_mappings: Vec<naga::back::msl::VertexBufferMapping>,
}

/// Apply a loaded [`Config`] on top of already-built [`crate::params::Parameters`].
///
/// This mirrors the logic in [`crate::params::build_parameters`] so that the
/// same options expressed via JSON produce identical output to the equivalent
/// CLI flags.
///
/// `raw` is the same document as untyped JSON, used to tell which nested keys were given.
pub fn apply_config(
    config: Config,
    raw: &serde_json::Value,
    params: &mut crate::params::Parameters<'static>,
) {
    // Validation flags.
    if let Some(bits) = config.validate {
        if let Some(flags) = naga::valid::ValidationFlags::from_bits(bits) {
            params.validation_flags = flags;
        }
    }

    // Capabilities.
    if let Some(caps) = config.capabilities {
        params.capabilities = caps;
    }

    // Bounds-check policies.
    if let Some(policies) = config.bounds_check_policies {
        params.bounds_check_policies = policies;
    }

    // Entry point.
    if let Some(ep) = config.entry_point {
        params.entry_point = Some(ep);
    }

    // Keep-coordinate-space flag; also re-applies the writer flags below.
    let keep_coordinate_space = config.keep_coordinate_space.unwrap_or(false);
    if config.keep_coordinate_space.is_some() {
        params.keep_coordinate_space = keep_coordinate_space;
    }

    // Defines.
    if let Some(defines) = config.defines {
        params.defines = defines.into_iter().collect();
    }

    // Pipeline-constant overrides.
    if let Some(overrides) = config.overrides {
        params.overrides = overrides.into_iter().collect();
    }

    // SPIR-V frontend options.
    if let Some(spv_in) = config.spv_in {
        params.spv_in = spv_in;
        // naga-cli defaults `strict_capabilities` to false in both flag and config mode,
        // unlike `naga::front::spv::Options::default()`.
        if raw["spv_in"].get("strict_capabilities").is_none() {
            params.spv_in.strict_capabilities = false;
        }
        // If keep_coordinate_space was also set in the config, apply it on top of the
        // wholesale spv_in replacement (mirrors build_parameters' relationship:
        // adjust_coordinate_space = !keep_coordinate_space).
        if config.keep_coordinate_space.is_some() {
            params.spv_in.adjust_coordinate_space = !keep_coordinate_space;
        }
    } else {
        // Mirror build_parameters: spv_in.adjust_coordinate_space follows keep_coordinate_space.
        params.spv_in.adjust_coordinate_space = !keep_coordinate_space;
    }

    // SPIR-V backend options.
    if let Some(spv_out) = config.spv_out {
        params.spv_out = spv_out;
    }
    // Re-apply the ADJUST_COORDINATE_SPACE writer flag exactly as build_parameters does,
    // so that config output == equivalent-flag output.
    params.spv_out.flags.set(
        naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE,
        !keep_coordinate_space,
    );
    // Also propagate bounds-check policies into spv_out (mirrors build_parameters).
    params.spv_out.bounds_check_policies = params.bounds_check_policies;

    // MSL backend options.
    if let Some(msl) = config.msl {
        params.msl = msl;
    }

    // GLSL backend options.
    if let Some(glsl) = config.glsl_out {
        params.glsl = glsl;
    }
    // Re-apply the ADJUST_COORDINATE_SPACE writer flag for GLSL (mirrors build_parameters).
    params.glsl.writer_flags.set(
        naga::back::glsl::WriterFlags::ADJUST_COORDINATE_SPACE,
        !keep_coordinate_space,
    );

    // HLSL backend options.
    if let Some(hlsl) = config.hlsl {
        params.hlsl = hlsl;
    }
    if let Some(multisampled) = config.hlsl_multisampled {
        params.hlsl_multisampled = multisampled;
    }
    if let Some(msl_pipeline) = config.msl_pipeline {
        params.msl_pipeline = msl_pipeline;
    }
    if let Some(views) = config.glsl_multiview {
        params.glsl_multiview = Some(views);
    }
    if let Some(explicit) = config.wgsl_explicit_types {
        params.wgsl_explicit_types = explicit;
    }
    if let Some(stages) = config.subgroup_stages {
        params.subgroup_stages = Some(stages);
    }
    if let Some(ops) = config.subgroup_operations {
        params.subgroup_operations = Some(ops);
    }

    // Dot backend options.
    if let Some(dot) = config.dot {
        params.dot = dot;
    }

    // Processing actions. (--before-compaction stays a CLI-only I/O flag and can still
    // force compaction; hence the `|| params.compact` to preserve an already-set value.)
    if let Some(compact) = config.compact {
        params.compact = compact || params.compact;
    }
    if let Some(g) = config.generate_debug_symbols {
        params.generate_debug_symbols = g;
    }
    if let Some(v) = config.spirv_val {
        params.spirv_val = v;
    }
    if let Some(o) = config.spirv_opt {
        params.spirv_opt = o;
    }
    if let Some(d) = config.dxc {
        params.dxc = d;
    }
    if let Some(path) = config.spirv_val_path {
        params.spirv_val_path = Some(path);
    }
    if let Some(path) = config.spirv_opt_path {
        params.spirv_opt_path = Some(path);
    }
    if let Some(path) = config.dxc_path {
        params.dxc_path = Some(path);
    }

    // I/O format specifiers (let stdin/stdout `-` work in config mode).
    if let Some(kind) = config.input_kind {
        params.input_kind = Some(kind);
    }
    if let Some(kind) = config.output_kind {
        params.output_kind = Some(kind);
    }
    if let Some(stage) = config.shader_stage {
        params.shader_stage = Some(stage);
    }
    if let Some(format) = config.format {
        params.format = format;
    }
    if let Some(path) = config.json_out {
        params.json_out = Some(path);
    }
    if let Some(path) = config.before_compaction {
        params.before_compaction = Some(path);
        // --before-compaction implies compaction.
        params.compact = true;
    }
    if let Some(bulk) = config.bulk_validate {
        params.bulk_validate = bulk;
    }
}
