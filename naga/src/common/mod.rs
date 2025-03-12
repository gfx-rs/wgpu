//! Code common to the front and backends for specific languages.

pub mod wgsl;

/// Helper function that returns the string corresponding to the [`VectorSize`](crate::VectorSize)
pub const fn vector_size_str(size: crate::VectorSize) -> &'static str {
    match size {
        crate::VectorSize::Bi => "2",
        crate::VectorSize::Tri => "3",
        crate::VectorSize::Quad => "4",
    }
}
