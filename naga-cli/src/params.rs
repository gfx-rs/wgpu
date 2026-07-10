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

    // Apply CommonBackendOptions (fake_missing_bindings, force_loop_bounding,
    // ray_query_initialization_tracking) to the three backends that embed it.
    params.spv_out.common = args.common.clone();
    params.msl.common = args.common.clone();
    params.hlsl.common = args.common.clone();

    // Apply --zero-initialize-workgroup-memory to all four backends.
    if let Some(mode) = args.zero_initialize_workgroup_memory {
        params.spv_out.zero_initialize_workgroup_memory = mode;
        params.msl.zero_initialize_workgroup_memory = mode;
        params.hlsl.zero_initialize_workgroup_memory = mode;
        params.glsl.zero_initialize_workgroup_memory = mode;
    }

    // Bespoke overrides for fields in CommonBackendOptions that have their own CLI flags
    // (--task-limits, --validate-mesh-output). These run AFTER the .common copy so they win.
    params.spv_out.common.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.spv_out.common.task_dispatch_limits = args.task_limits;
    params.msl.common.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.msl.common.task_dispatch_limits = args.task_limits;
    params.hlsl.common.mesh_shader_primitive_indices_clamp = args.validate_mesh_output;
    params.hlsl.common.task_dispatch_limits = args.task_limits;

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
        let args =
            Args::try_parse_from(["naga", "--index-bounds-check-policy", "restrict", "in.wgsl"])
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

    #[test]
    fn validate_mesh_output_false_disables_clamp() {
        // --validate-mesh-output false must be accepted as a valued option and propagate to all
        // three backend options.
        let args =
            Args::try_parse_from(["naga", "--validate-mesh-output", "false", "in.wgsl"]).unwrap();
        assert!(!args.validate_mesh_output);
        let params = build_parameters(&args).unwrap();
        assert!(!params.spv_out.common.mesh_shader_primitive_indices_clamp);
        assert!(!params.msl.common.mesh_shader_primitive_indices_clamp);
        assert!(!params.hlsl.common.mesh_shader_primitive_indices_clamp);

        // default (omitted) stays true
        let args_default = Args::try_parse_from(["naga", "in.wgsl"]).unwrap();
        assert!(args_default.validate_mesh_output);
        let params_default = build_parameters(&args_default).unwrap();
        assert!(
            params_default
                .spv_out
                .common
                .mesh_shader_primitive_indices_clamp
        );
        assert!(
            params_default
                .msl
                .common
                .mesh_shader_primitive_indices_clamp
        );
        assert!(
            params_default
                .hlsl
                .common
                .mesh_shader_primitive_indices_clamp
        );

        // explicit true also works
        let args_true =
            Args::try_parse_from(["naga", "--validate-mesh-output", "true", "in.wgsl"]).unwrap();
        assert!(args_true.validate_mesh_output);
    }

    #[test]
    fn force_loop_bounding_propagates() {
        // --force-loop-bounding false must propagate to all three backends.
        let args =
            Args::try_parse_from(["naga", "--force-loop-bounding", "false", "in.wgsl"]).unwrap();
        let params = build_parameters(&args).unwrap();
        assert!(!params.spv_out.common.force_loop_bounding);
        assert!(!params.msl.common.force_loop_bounding);
        assert!(!params.hlsl.common.force_loop_bounding);

        // default (omitted) must stay true on all three backends.
        let args_default = Args::try_parse_from(["naga", "in.wgsl"]).unwrap();
        let params_default = build_parameters(&args_default).unwrap();
        assert!(params_default.spv_out.common.force_loop_bounding);
        assert!(params_default.msl.common.force_loop_bounding);
        assert!(params_default.hlsl.common.force_loop_bounding);
    }

    #[test]
    fn fake_missing_bindings_propagates() {
        // --fake-missing-bindings false must propagate to all three backends.
        let args =
            Args::try_parse_from(["naga", "--fake-missing-bindings", "false", "in.wgsl"]).unwrap();
        let params = build_parameters(&args).unwrap();
        assert!(!params.spv_out.common.fake_missing_bindings);
        assert!(!params.msl.common.fake_missing_bindings);
        assert!(!params.hlsl.common.fake_missing_bindings);

        // default (omitted) must stay true on all three backends.
        let args_default = Args::try_parse_from(["naga", "in.wgsl"]).unwrap();
        let params_default = build_parameters(&args_default).unwrap();
        assert!(params_default.spv_out.common.fake_missing_bindings);
        assert!(params_default.msl.common.fake_missing_bindings);
        assert!(params_default.hlsl.common.fake_missing_bindings);
    }

    #[test]
    fn ray_query_initialization_tracking_propagates() {
        // --ray-query-initialization-tracking false must propagate to all three backends.
        let args = Args::try_parse_from([
            "naga",
            "--ray-query-initialization-tracking",
            "false",
            "in.wgsl",
        ])
        .unwrap();
        let params = build_parameters(&args).unwrap();
        assert!(!params.spv_out.common.ray_query_initialization_tracking);
        assert!(!params.msl.common.ray_query_initialization_tracking);
        assert!(!params.hlsl.common.ray_query_initialization_tracking);

        // default (omitted) must stay true on all three backends.
        let args_default = Args::try_parse_from(["naga", "in.wgsl"]).unwrap();
        let params_default = build_parameters(&args_default).unwrap();
        assert!(
            params_default
                .spv_out
                .common
                .ray_query_initialization_tracking
        );
        assert!(params_default.msl.common.ray_query_initialization_tracking);
        assert!(params_default.hlsl.common.ray_query_initialization_tracking);
    }

    #[test]
    fn task_limits_propagates() {
        // --task-limits X,Y must propagate to all three backends as Some(TaskDispatchLimits{..}).
        let args = Args::try_parse_from(["naga", "--task-limits", "8,16", "in.wgsl"]).unwrap();
        let params = build_parameters(&args).unwrap();
        let expected = Some(naga::back::TaskDispatchLimits {
            max_mesh_workgroups_per_dim: 8,
            max_mesh_workgroups_total: 16,
        });
        assert_eq!(params.spv_out.common.task_dispatch_limits, expected);
        assert_eq!(params.msl.common.task_dispatch_limits, expected);
        assert_eq!(params.hlsl.common.task_dispatch_limits, expected);

        // default (omitted) must stay None on all three backends.
        let args_default = Args::try_parse_from(["naga", "in.wgsl"]).unwrap();
        let params_default = build_parameters(&args_default).unwrap();
        assert_eq!(params_default.spv_out.common.task_dispatch_limits, None);
        assert_eq!(params_default.msl.common.task_dispatch_limits, None);
        assert_eq!(params_default.hlsl.common.task_dispatch_limits, None);
    }
}
