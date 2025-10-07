use alloc::vec;

use super::Error;
use crate::arena::Handle;

/// Create a default value for an output built-in.
pub fn generate_default_built_in(
    module: &mut crate::Module,
    global_expression_kind_tracker: &mut crate::proc::ExpressionKindTracker,
    built_in: Option<crate::BuiltIn>,
    ty: Handle<crate::Type>,
    span: crate::Span,
) -> Result<Handle<crate::Expression>, Error> {
    let expr = match built_in {
        Some(crate::BuiltIn::Position { .. }) => {
            let zero = super::append_global_expression(
                module,
                global_expression_kind_tracker,
                crate::Expression::Literal(crate::Literal::F32(0.0)),
                crate::proc::ExpressionKind::Const,
                span,
            );
            let one = super::append_global_expression(
                module,
                global_expression_kind_tracker,
                crate::Expression::Literal(crate::Literal::F32(1.0)),
                crate::proc::ExpressionKind::Const,
                span,
            );
            crate::Expression::Compose {
                ty,
                components: vec![zero, zero, zero, one],
            }
        }
        Some(crate::BuiltIn::PointSize) => crate::Expression::Literal(crate::Literal::F32(1.0)),
        Some(crate::BuiltIn::FragDepth) => crate::Expression::Literal(crate::Literal::F32(0.0)),
        Some(crate::BuiltIn::SampleMask) => {
            crate::Expression::Literal(crate::Literal::U32(u32::MAX))
        }
        // Note: `crate::BuiltIn::ClipDistance` is intentionally left for the default path
        _ => crate::Expression::ZeroValue(ty),
    };
    Ok(super::append_global_expression(
        module,
        global_expression_kind_tracker,
        expr,
        crate::proc::ExpressionKind::Const,
        span,
    ))
}
