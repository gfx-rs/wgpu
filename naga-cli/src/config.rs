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

    /// Options for the dot backend.
    pub dot: Option<naga::back::dot::Options>,

    /// Compact the module's IR and revalidate before output.
    pub compact: Option<bool>,

    /// Generate debug symbols (spv-out only, for now).
    pub generate_debug_symbols: Option<bool>,

    /// After writing SPIR-V output, validate it with `spirv-val` (must be on PATH).
    pub spirv_val: Option<bool>,

    /// After writing SPIR-V output, optimize it in place with `spirv-opt` (must be on PATH).
    pub spirv_opt: Option<bool>,

    /// After writing HLSL output, compile each entry point to DXIL with `dxc` (must be on PATH).
    pub dxc: Option<bool>,
}

/// Apply a loaded [`Config`] on top of already-built [`crate::params::Parameters`].
///
/// This mirrors the logic in [`crate::params::build_parameters`] so that the
/// same options expressed via JSON produce identical output to the equivalent
/// CLI flags.
pub fn apply_config(config: Config, params: &mut crate::params::Parameters<'static>) {
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
}
