//! Code common to the front and backends for specific languages.

mod diagnostic_debug;
mod diagnostic_display;
mod external_texture;
pub mod predeclared;
pub mod wgsl;

pub use diagnostic_debug::{DiagnosticDebug, ForDebug, ForDebugWithTypes};
pub use diagnostic_display::DiagnosticDisplay;
pub(crate) use external_texture::ExternalTextureParameter;

// Re-exported here for backwards compatibility
pub use super::proc::vector_size_str;
