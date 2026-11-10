mod example_wgsl;
#[cfg(all(feature = "serialize", feature = "deserialize", feature = "wgsl-in"))]
mod function_control_serialization;
mod snapshots;
mod spirv_capabilities;
mod spirv_debug_info;
mod spirv_roundtrip;
mod validation;
mod wgsl_errors;

#[cfg(all(
    feature = "wgsl-in",
    feature = "spv-in",
    spv_out,
    hlsl_out,
    glsl_out,
    msl_out,
    wgsl_out
))]
mod exp2_control;

mod hlsl_inout;
