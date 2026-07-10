//! Translation parameters assembled from parsed CLI arguments.

use crate::cli::{Args, InputKind, ShaderStageArg};
use anyhow::anyhow;
use naga::FastHashMap;

#[derive(Default)]
pub struct Parameters<'a> {
    pub validation_flags: naga::valid::ValidationFlags,
    pub bounds_check_policies: naga::proc::BoundsCheckPolicies,
    pub entry_point: Option<String>,
    pub keep_coordinate_space: bool,
    pub overrides: naga::back::PipelineConstants,
    pub spv_in: naga::front::spv::Options,
    pub spv_out: naga::back::spv::Options<'a>,
    pub dot: naga::back::dot::Options,
    pub msl: naga::back::msl::Options,
    pub glsl: naga::back::glsl::Options,
    pub hlsl: naga::back::hlsl::Options,
    pub input_kind: Option<InputKind>,
    pub shader_stage: Option<ShaderStageArg>,
    pub defines: FastHashMap<String, String>,
    pub capabilities: naga::valid::Capabilities,
    /// Whether to pass the entry point to `process_overrides` (drops unreachable items).
    pub compact: bool,
}

pub fn build_parameters(args: &Args) -> anyhow::Result<Parameters<'static>> {
    let mut params = Parameters::default();

    if let Some(bits) = args.validate {
        params.validation_flags = naga::valid::ValidationFlags::from_bits(bits)
            .ok_or_else(|| anyhow!("Invalid validation flags: {bits}"))?;
    }

    if let Some(policy) = args.index_bounds_check_policy {
        params.bounds_check_policies.index = policy.to_policy();
    }
    params.bounds_check_policies.buffer = match args.buffer_bounds_check_policy {
        Some(p) => p.to_policy(),
        None => params.bounds_check_policies.index,
    };
    params.bounds_check_policies.image_load = match args.image_load_bounds_check_policy {
        Some(p) => p.to_policy(),
        None => params.bounds_check_policies.index,
    };

    params.overrides = args
        .overrides
        .iter()
        .flat_map(|o| o.pairs.iter().cloned())
        .collect();
    params.defines = args
        .defines
        .iter()
        .flat_map(|o| o.pairs.iter().cloned())
        .collect();

    params.spv_in = naga::front::spv::Options {
        adjust_coordinate_space: !args.keep_coordinate_space,
        strict_capabilities: false,
        block_ctx_dump_prefix: args.block_ctx_dir.clone(),
    };

    params.entry_point.clone_from(&args.entry_point);
    if let Some(version) = args.profile {
        params.glsl.version = version;
    }
    if let Some(ref model) = args.shader_model {
        params.hlsl.shader_model = *model;
    }
    if let Some(version) = args.metal_version {
        params.msl.lang_version = version;
    }
    if let Some(version) = args.spirv_version {
        params.spv_out.lang_version = version;
    }
    params.keep_coordinate_space = args.keep_coordinate_space;
    params.dot.cfg_only = args.dot_cfg_only;

    params.spv_out.bounds_check_policies = params.bounds_check_policies;
    params.spv_out.flags.set(
        naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE,
        !params.keep_coordinate_space,
    );
    params.glsl.writer_flags.set(
        naga::back::glsl::WriterFlags::ADJUST_COORDINATE_SPACE,
        !params.keep_coordinate_space,
    );

    params.compact = args.compact || args.before_compaction.is_some();
    params.capabilities = args.capabilities;

    params.spv_out.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.spv_out.task_dispatch_limits = args.task_limits;
    params.msl.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.msl.task_dispatch_limits = args.task_limits;
    params.hlsl.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.hlsl.task_dispatch_limits = args.task_limits;

    params.input_kind = args.input_kind;
    params.shader_stage = args.shader_stage;

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Args;
    use clap::Parser;

    #[test]
    fn buffer_policy_defaults_to_index_policy() {
        let args = Args::try_parse_from([
            "naga",
            "--index-bounds-check-policy",
            "restrict",
            "in.wgsl",
        ])
        .unwrap();
        let params = build_parameters(&args).unwrap();
        assert_eq!(
            params.bounds_check_policies.buffer,
            naga::proc::BoundsCheckPolicy::Restrict
        );
        assert_eq!(
            params.bounds_check_policies.image_load,
            naga::proc::BoundsCheckPolicy::Restrict
        );
    }

    #[test]
    fn invalid_validate_bits_error() {
        let args = Args::try_parse_from(["naga", "--validate", "255", "in.wgsl"]).unwrap();
        // 255 has bits outside ValidationFlags (all() = 0x3F = 63); build should error rather than panic.
        assert!(build_parameters(&args).is_err());
    }
}
