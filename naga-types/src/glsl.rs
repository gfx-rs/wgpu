// Must match code in glsl_built_in
pub const FIRST_INSTANCE_BINDING: &str = "naga_vs_first_instance";

/// List of supported `core` GLSL versions.
pub const SUPPORTED_CORE_VERSIONS: &[u16] = &[140, 150, 330, 400, 410, 420, 430, 440, 450, 460];
/// List of supported `es` GLSL versions.
pub const SUPPORTED_ES_VERSIONS: &[u16] = &[300, 310, 320];
