use crate::{id, Label};
use alloc::borrow::Cow;
use hashbrown::HashMap;

/// Describes a programmable pipeline stage.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
/// cbindgen:ignore
pub struct ProgrammableStageDescriptor<'a> {
    /// The compiled shader module for this stage.
    pub module: id::ShaderModuleId,

    /// The name of the entry point in `module` that this stage should use.
    ///
    /// - If this is `Some(name)`, `module` must contain an entry point with the
    ///   given name.
    ///
    /// - If this is `None`, `module` must have only one entry point for this
    ///   stage; we use that one.
    pub entry_point: Option<Cow<'a, str>>,

    /// Values for pipeline-overridable constants in `module` that this stage
    /// should use.
    ///
    /// If an `@id` attribute was specified on the declaration,
    /// the key must be the pipeline constant ID as a decimal ASCII number; if not,
    /// the key must be the constant's identifier name.
    ///
    /// The value may represent any of WGSL's concrete scalar types.
    pub constants: HashMap<String, f64>,
}

/// Describes a compute pipeline.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
/// cbindgen:ignore
pub struct ComputePipelineDescriptor<'a> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<id::PipelineLayoutId>,
    /// The compiled compute stage and its entry point.
    pub stage: ProgrammableStageDescriptor<'a>,
}
