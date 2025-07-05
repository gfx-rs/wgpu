//! Code common to the front and backends for specific languages.

mod diagnostic_debug;
mod diagnostic_display;
pub mod predeclared;
pub mod wgsl;

pub use diagnostic_debug::{DiagnosticDebug, ForDebug, ForDebugWithTypes};
pub use diagnostic_display::DiagnosticDisplay;

/// Helper function that returns the string corresponding to the [`VectorSize`](crate::VectorSize)
pub const fn vector_size_str(size: crate::VectorSize) -> &'static str {
    match size {
        crate::VectorSize::Bi => "2",
        crate::VectorSize::Tri => "3",
        crate::VectorSize::Quad => "4",
    }
}
