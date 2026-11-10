mod example_wgsl;
#[cfg(all(feature = "serialize", feature = "deserialize", feature = "wgsl-in"))]
mod function_control_serialization;
mod snapshots;
mod spirv_capabilities;
mod spirv_debug_info;
mod spirv_roundtrip;
mod validation;
mod wgsl_errors;

mod hlsl_inout;
