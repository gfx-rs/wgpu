//! Functions which export shader modules into binary and text formats.

#[cfg(feature = "glsl-out")]
pub mod glsl;
#[cfg(feature = "msl-out")]
pub mod msl;
#[cfg(feature = "spv-out")]
pub mod spv;

impl crate::Expression {
    /// Returns the ref count, upon reaching which this expression
    /// should be considered for baking.
    #[allow(dead_code)]
    fn bake_ref_count(&self) -> usize {
        match *self {
            // accesses are never cached, only loads are
            crate::Expression::Access { .. } | crate::Expression::AccessIndex { .. } => !0,
            // image operations look better when isolated
            crate::Expression::ImageSample { .. } | crate::Expression::ImageLoad { .. } => 1,
            // cache expressions that are referenced multiple times
            _ => 2,
        }
    }
}
