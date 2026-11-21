//! The uniformity analysis of the WGSL specification, run on Naga IR.
//!
//! See <https://www.w3.org/TR/WGSL/#uniformity>. Each function is analyzed
//! after its callees. The analysis builds a graph whose nodes stand for
//! "this point of the program is in uniform control flow" or "this value is
//! uniform", and whose edges are implications between them. A uniformity
//! failure is a path from a `RequiredToBeUniform.S` node to the
//! `MayBeNonUniform` node.
//!
//! - [`graph`] holds the graph and its reachability queries.
//! - [`tags`] holds the per-function summaries that callers use, and the
//!   uniformity properties of IR operations. New operations with uniformity
//!   requirements are added there.
//! - [`walk`] builds the graph of one function.

mod graph;
mod tags;
mod walk;

use alloc::string::String;
use core::fmt;

use crate::span::WithSpan;
use crate::valid::FunctionInfo;
use crate::{Function, Module};

pub(super) use tags::FunctionSummary;

#[derive(Clone, Copy, Debug)]
pub(super) enum FunctionKind {
    Regular,
    EntryPoint(crate::ShaderStage),
}

/// Analyze the uniformity of `function`.
///
/// `summaries` holds the result of this function for every function in
/// `module.functions` that precedes `function`.
pub(super) fn analyze_function(
    module: &Module,
    function: &Function,
    info: &FunctionInfo,
    summaries: &[FunctionSummary],
    kind: FunctionKind,
) -> Result<FunctionSummary, WithSpan<UniformityError>> {
    walk::Analyzer::new(module, function, info, summaries, kind).run()
}

/// An operation that requires uniformity, or that produces a value that may
/// be non-uniform.
#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Operation {
    Barrier(crate::Barrier),
    WorkGroupUniformLoad,
    Derivative,
    ImplicitLodSample,
    StorageTextureLoad,
    Atomic,
    Subgroup,
    Quad,
    CooperativeMatrix,
    RayQuery,
    HitObject,
    Call { name: String },
    Unknown,
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::Barrier as B;
        match *self {
            Self::Barrier(B::WORK_GROUP) => f.write_str("`workgroupBarrier`"),
            Self::Barrier(B::STORAGE) => f.write_str("`storageBarrier`"),
            Self::Barrier(B::TEXTURE) => f.write_str("`textureBarrier`"),
            Self::Barrier(B::SUB_GROUP) => f.write_str("`subgroupBarrier`"),
            Self::Barrier(_) => f.write_str("a control barrier"),
            Self::WorkGroupUniformLoad => f.write_str("`workgroupUniformLoad`"),
            Self::Derivative => f.write_str("a derivative"),
            Self::ImplicitLodSample => {
                f.write_str("a texture sample with an implicit level of detail")
            }
            Self::StorageTextureLoad => f.write_str("a load from a read-write storage texture"),
            Self::Atomic => f.write_str("an atomic operation"),
            Self::Subgroup => f.write_str("a subgroup operation"),
            Self::Quad => f.write_str("a quad operation"),
            Self::CooperativeMatrix => f.write_str("a cooperative matrix operation"),
            Self::RayQuery => f.write_str("a ray query"),
            Self::HitObject => f.write_str("a hit object query"),
            Self::Call { ref name } => write!(f, "a call to `{name}`"),
            Self::Unknown => f.write_str("an operation"),
        }
    }
}

/// Where a value that may be non-uniform comes from.
#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum NonUniformCause {
    MutableGlobal,
    PrivatePointerArgument,
    BuiltIn(crate::BuiltIn),
    UserDefinedInput,
    Result(Operation),
    PointerArgument { function: String, index: u32 },
    Unknown,
}

impl fmt::Display for NonUniformCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::common::wgsl::TryToWgsl;
        match *self {
            Self::MutableGlobal => f.write_str("a load from a mutable module-scope variable"),
            Self::PrivatePointerArgument => {
                f.write_str("a load through a pointer argument in the `private` address space")
            }
            Self::BuiltIn(builtin) => {
                write!(
                    f,
                    "the built-in value `{}`",
                    builtin.to_wgsl_for_diagnostics()
                )
            }
            Self::UserDefinedInput => f.write_str("a user-defined shader input"),
            Self::Result(ref op) => write!(f, "the result of {op}"),
            Self::PointerArgument {
                ref function,
                index,
            } => write!(
                f,
                "the value that `{function}` stores through its argument {index}"
            ),
            Self::Unknown => f.write_str("an unknown value"),
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
enum RequirementTarget {
    ControlFlow,
    Argument(u32),
    ArgumentContents(u32),
}

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum UniformityError {
    #[error(
        "Uniform control flow is required for {op}, but control flow may be non-uniform because of {cause}"
    )]
    NonUniformControlFlow {
        op: Operation,
        cause: NonUniformCause,
    },
    #[error(
        "Argument {index} of {op} must be a uniform value, but it may be non-uniform because of {cause}"
    )]
    NonUniformArgument {
        op: Operation,
        index: u32,
        cause: NonUniformCause,
    },
    #[error("The value that argument {index} of {op} points to must be uniform, but it may be non-uniform because of {cause}")]
    NonUniformArgumentContents {
        op: Operation,
        index: u32,
        cause: NonUniformCause,
    },
}
