//! Uniformity tags of functions and of IR operations.
//!
//! [`FunctionSummary`] holds the tags the WGSL spec computes for user-defined
//! functions. [`expression_intrinsic`] and [`statement_intrinsic`] give the
//! tags of IR operations that the spec describes as built-in functions with
//! special uniformity properties. A new operation with uniformity requirements
//! needs a new arm in one of those two functions.

use alloc::{vec, vec::Vec};

use super::Operation;
use crate::{diagnostic_filter::Severity, Expression, Handle, Statement};

bitflags::bitflags! {
    /// A WGSL spec potential-trigger-set.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub(super) struct Triggers: u8 {
        const DERIVATIVE_UNIFORMITY = 1 << 0;
        const SUBGROUP_UNIFORMITY = 1 << 1;
        /// The triggering rule for synchronization functions, which cannot be filtered.
        const UNNAMED = 1 << 2;
    }
}

/// A uniformity requirement that triggers a diagnostic of `severity`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Requirement {
    pub severity: Severity,
    pub triggers: Triggers,
}

/// The tags the WGSL spec computes for a function.
#[derive(Clone, Debug, Default)]
pub(in crate::valid) struct FunctionSummary {
    /// `CallSiteRequiredToBeUniform.S`.
    pub(super) call_site: Option<Requirement>,
    /// `ReturnValueMayBeNonUniform`.
    pub(super) result_may_be_non_uniform: bool,
    pub(super) params: Vec<ParamSummary>,
}

impl FunctionSummary {
    /// A summary that assumes the worst about a function.
    pub(super) fn conservative(params: usize) -> Self {
        let requirement = Requirement {
            severity: Severity::Error,
            triggers: Triggers::UNNAMED,
        };
        FunctionSummary {
            call_site: Some(requirement),
            result_may_be_non_uniform: true,
            params: vec![
                ParamSummary {
                    required: Some(requirement),
                    contents_required: Some(requirement),
                    pointer_out: Some(PointerOut {
                        may_be_non_uniform: true,
                        ..PointerOut::default()
                    }),
                    ..ParamSummary::default()
                };
                params
            ],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct ParamSummary {
    /// `ParameterRequiredToBeUniform.S`.
    pub required: Option<Requirement>,
    /// `ParameterContentsRequiredToBeUniform.S`.
    pub contents_required: Option<Requirement>,
    /// `ParameterReturnContentsRequiredToBeUniform`, for the parameter value.
    pub result_depends_on_value: bool,
    /// `ParameterReturnContentsRequiredToBeUniform`, for the contents of a
    /// `function` address space pointer parameter.
    pub result_depends_on_contents: bool,
    /// What the memory behind a `function` address space pointer parameter
    /// holds when the function returns.
    pub pointer_out: Option<PointerOut>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct PointerOut {
    /// `PointerParameterMayBeNonUniform`.
    pub may_be_non_uniform: bool,
    /// The stored value depends on the control flow at the call site.
    pub depends_on_call_site: bool,
    pub depends_on: Vec<ParamDependency>,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ParamDependency {
    Value(u32),
    Contents(u32),
}

/// The uniformity of an operation's result.
#[derive(Clone, Copy, Debug)]
pub(super) enum IntrinsicResult {
    /// Uniform if the control flow and every operand are uniform.
    Operands,
    /// Uniform if the control flow is uniform.
    Uniform,
    /// May be non-uniform.
    NonUniform,
}

/// The uniformity tags of an IR operation.
#[derive(Clone, Debug)]
pub(super) struct Intrinsic {
    pub op: Operation,
    /// The operation must only execute in uniform control flow.
    pub call_site: Option<Triggers>,
    /// Operands that must be uniform values, with their argument index in WGSL.
    pub uniform_operands: Vec<(u32, Handle<Expression>, Triggers)>,
    pub result: IntrinsicResult,
    /// The expression that holds the result of a statement.
    pub result_expr: Option<Handle<Expression>>,
}

impl Intrinsic {
    const fn new(op: Operation, call_site: Option<Triggers>, result: IntrinsicResult) -> Self {
        Intrinsic {
            op,
            call_site,
            uniform_operands: Vec::new(),
            result,
            result_expr: None,
        }
    }
}

/// The tags of expressions that differ from the default, which is a result
/// that is uniform if its operands are.
///
/// The spec doesn't cover ray queries, hit objects, or cooperative matrices.
/// Their results are treated as non-uniform, and cooperative matrix
/// operations require uniform control flow without a way to filter the
/// diagnostic.
///
/// `is_read_write_storage_image` tells whether an image expression has a
/// read-write storage texture type.
pub(super) fn expression_intrinsic(
    expr: &Expression,
    is_read_write_storage_image: impl FnOnce(Handle<Expression>) -> bool,
) -> Option<Intrinsic> {
    use IntrinsicResult as R;
    Some(match *expr {
        Expression::Derivative { .. } => Intrinsic::new(
            Operation::Derivative,
            Some(Triggers::DERIVATIVE_UNIFORMITY),
            R::NonUniform,
        ),
        Expression::ImageSample { ref level, .. } if level.implicit_derivatives() => {
            Intrinsic::new(
                Operation::ImplicitLodSample,
                Some(Triggers::DERIVATIVE_UNIFORMITY),
                R::NonUniform,
            )
        }
        Expression::ImageLoad { image, .. } if is_read_write_storage_image(image) => {
            Intrinsic::new(Operation::StorageTextureLoad, None, R::NonUniform)
        }
        Expression::CooperativeLoad { .. } | Expression::CooperativeMultiplyAdd { .. } => {
            Intrinsic::new(
                Operation::CooperativeMatrix,
                Some(Triggers::UNNAMED),
                R::Operands,
            )
        }
        Expression::RayQueryGetIntersection { .. } | Expression::RayQueryVertexPositions { .. } => {
            Intrinsic::new(Operation::RayQuery, None, R::NonUniform)
        }
        Expression::HitObjectQuery { .. } => {
            Intrinsic::new(Operation::HitObject, None, R::NonUniform)
        }
        _ => return None,
    })
}

/// The tags of statements that have uniformity requirements or produce a
/// result.
///
/// [`Statement::Call`] is handled with the callee's [`FunctionSummary`]
/// instead.
pub(super) fn statement_intrinsic(statement: &Statement) -> Option<Intrinsic> {
    use IntrinsicResult as R;
    let subgroup = |result| {
        let mut intrinsic = Intrinsic::new(
            Operation::Subgroup,
            Some(Triggers::SUBGROUP_UNIFORMITY),
            R::NonUniform,
        );
        intrinsic.result_expr = Some(result);
        intrinsic
    };
    Some(match *statement {
        Statement::ControlBarrier(barrier) => {
            // `subgroupBarrier` isn't in the spec. Treat it as a subgroup
            // built-in function.
            let triggers = if barrier == crate::Barrier::SUB_GROUP {
                Triggers::SUBGROUP_UNIFORMITY
            } else {
                Triggers::UNNAMED
            };
            Intrinsic::new(Operation::Barrier(barrier), Some(triggers), R::Uniform)
        }
        Statement::WorkGroupUniformLoad { pointer, result } => Intrinsic {
            uniform_operands: vec![(0, pointer, Triggers::UNNAMED)],
            result_expr: Some(result),
            ..Intrinsic::new(
                Operation::WorkGroupUniformLoad,
                Some(Triggers::UNNAMED),
                R::Uniform,
            )
        },
        Statement::Atomic {
            result: Some(result),
            ..
        } => Intrinsic {
            result_expr: Some(result),
            ..Intrinsic::new(Operation::Atomic, None, R::NonUniform)
        },
        Statement::SubgroupBallot { result, .. }
        | Statement::SubgroupCollectiveOperation { result, .. }
        | Statement::SubgroupBallotFindBit { result, .. } => subgroup(result),
        Statement::SubgroupGather {
            ref mode, result, ..
        } => {
            use crate::GatherMode as Gm;
            let mut intrinsic = subgroup(result);
            match *mode {
                Gm::ShuffleDown(delta) | Gm::ShuffleUp(delta) | Gm::ShuffleXor(delta) => {
                    intrinsic
                        .uniform_operands
                        .push((1, delta, Triggers::SUBGROUP_UNIFORMITY));
                }
                Gm::QuadBroadcast(_) | Gm::QuadSwap(_) => intrinsic.op = Operation::Quad,
                Gm::BroadcastFirst | Gm::Broadcast(_) | Gm::Shuffle(_) => {}
            }
            intrinsic
        }
        Statement::CooperativeStore { .. } => Intrinsic::new(
            Operation::CooperativeMatrix,
            Some(Triggers::UNNAMED),
            R::Uniform,
        ),
        Statement::RayQuery {
            fun: crate::RayQueryFunction::Proceed { result },
            ..
        } => Intrinsic {
            result_expr: Some(result),
            ..Intrinsic::new(Operation::RayQuery, None, R::NonUniform)
        },
        _ => return None,
    })
}
