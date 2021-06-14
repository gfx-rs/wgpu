//! Functions which export shader modules into binary and text formats.

#[cfg(feature = "dot-out")]
pub mod dot;
#[cfg(feature = "glsl-out")]
pub mod glsl;
#[cfg(feature = "hlsl-out")]
pub mod hlsl;
#[cfg(feature = "msl-out")]
pub mod msl;
#[cfg(feature = "spv-out")]
pub mod spv;
#[cfg(feature = "wgsl-out")]
pub mod wgsl;

impl crate::Expression {
    /// Returns the ref count, upon reaching which this expression
    /// should be considered for baking.
    ///
    /// Note: we have to cache any expressions that depend on the control flow,
    /// or otherwise they may be moved into a non-uniform contol flow, accidentally.
    #[allow(dead_code)]
    fn bake_ref_count(&self) -> usize {
        match *self {
            // accesses are never cached, only loads are
            crate::Expression::Access { .. } | crate::Expression::AccessIndex { .. } => !0,
            // sampling may use the control flow, and image ops look better by themselves
            crate::Expression::ImageSample { .. } | crate::Expression::ImageLoad { .. } => 1,
            // derivatives use the control flow
            crate::Expression::Derivative { .. } => 1,
            // TODO: We need a better fix for named `Load` expressions
            // More info - https://github.com/gfx-rs/naga/pull/914
            // And https://github.com/gfx-rs/naga/issues/910
            crate::Expression::Load { .. } => 1,
            // cache expressions that are referenced multiple times
            _ => 2,
        }
    }
}

/// Helper function that returns the string corresponding to the [`BinaryOperator`](crate::BinaryOperator)
/// # Notes
/// Used by `glsl-out`, `msl-out`, `wgsl-out`, `hlsl-out`.
#[allow(dead_code)]
fn binary_operation_str(op: crate::BinaryOperator) -> &'static str {
    use crate::BinaryOperator as Bo;
    match op {
        Bo::Add => "+",
        Bo::Subtract => "-",
        Bo::Multiply => "*",
        Bo::Divide => "/",
        Bo::Modulo => "%",
        Bo::Equal => "==",
        Bo::NotEqual => "!=",
        Bo::Less => "<",
        Bo::LessEqual => "<=",
        Bo::Greater => ">",
        Bo::GreaterEqual => ">=",
        Bo::And => "&",
        Bo::ExclusiveOr => "^",
        Bo::InclusiveOr => "|",
        Bo::LogicalAnd => "&&",
        Bo::LogicalOr => "||",
        Bo::ShiftLeft => "<<",
        Bo::ShiftRight => ">>",
    }
}

/// Helper function that returns the string corresponding to the [`VectorSize`](crate::VectorSize)
/// # Notes
/// Used by `msl-out`, `wgsl-out`, `hlsl-out`.
#[allow(dead_code)]
fn vector_size_str(size: crate::VectorSize) -> &'static str {
    match size {
        crate::VectorSize::Bi => "2",
        crate::VectorSize::Tri => "3",
        crate::VectorSize::Quad => "4",
    }
}
