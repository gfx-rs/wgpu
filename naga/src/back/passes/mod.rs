/*!
IR passes that backends run on a module before writing it.

Backends don't take a [`ModuleInfo`]. Each backend runs `run` on the module
it is given, which works on one copy of the module: it keeps only the entry
point being written, compacts the module, replaces its overrides with the
pipeline's constants, runs the passes the backend asks for, puts the
expressions of the functions they changed back in order, and validates the
result. The backend then writes that
module with its own [`ModuleInfo`].

The module given to a backend must already have been validated: this
validation only checks the result of the passes.
*/
#![cfg_attr(
    not(any(glsl_out, hlsl_out, msl_out, spv_out)),
    allow(
        dead_code,
        reason = "shared helpers can be dead if none of the enabled backends need it"
    )
)]

use alloc::{boxed::Box, vec::Vec};
use core::mem;

use crate::{
    back::{
        pipeline_constants::{self, PipelineConstantError},
        PipelineConstants,
    },
    compact::{compact_with_global_origins, KeepUnused},
    valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags, Validator},
    Function, GlobalVariable, Handle, Module, ShaderStage, WithSpan,
};

/// An error produced while running the passes a backend requested.
#[derive(Clone, Debug, thiserror::Error)]
pub enum PassError {
    #[error(transparent)]
    Overrides(Box<PipelineConstantError>),
    #[error("module failed revalidation after backend passes")]
    Validation(#[from] Box<WithSpan<ValidationError>>),
}

impl From<PipelineConstantError> for PassError {
    fn from(error: PipelineConstantError) -> Self {
        Self::Overrides(Box::new(error))
    }
}

/// The passes a backend requests.
#[derive(Clone, Debug)]
pub(crate) struct Passes<'a> {
    /// Keep only this entry point.
    pub entry_point: Option<(ShaderStage, &'a str)>,
    /// Values for the pipeline-overridable constants.
    pub constants: Option<&'a PipelineConstants>,
    /// Which calls to inline. This runs before the other passes.
    pub inline: crate::inline::InlineStrategy,
}

/// A module prepared for a backend by [`run`].
pub(crate) struct Prepared {
    pub module: Module,
    pub info: ModuleInfo,
    /// For each global variable of `module`, its handle in the module given to
    /// [`run`], or `None` if the handles are the same.
    #[cfg_attr(
        not(glsl_out),
        allow(dead_code, reason = "only GLSL reports global variable handles")
    )]
    pub global_origins: Option<Vec<Handle<GlobalVariable>>>,
}

/// Functions that a pass may have left with unused expressions, or with
/// expressions that refer to later ones.
#[derive(Default)]
pub(crate) struct Dirty {
    pub functions: crate::FastHashSet<Handle<Function>>,
    /// Indices of entry points.
    pub entry_points: crate::FastHashSet<usize>,
}

/// Remove unused expressions from the `dirty` functions of `module`, and put
/// the others in order.
fn reorder(module: &mut Module, dirty: &Dirty) {
    for &handle in &dirty.functions {
        let mut function = mem::take(&mut module.functions[handle]);
        crate::compact::reorder_function(module, &mut function);
        module.functions[handle] = function;
    }
    for &index in &dirty.entry_points {
        let mut function = mem::take(&mut module.entry_points[index].function);
        crate::compact::reorder_function(module, &mut function);
        module.entry_points[index].function = function;
    }
}

/// Prepare `module` for a backend: keep only [`Passes::entry_point`], remove
/// what it doesn't use, replace its overrides with constants, and validate it.
///
/// `module` must be valid. Its overrides are replaced using
/// [`Passes::constants`] and the overrides' defaults.
pub(crate) fn run(module: &Module, passes: &Passes) -> Result<Prepared, PassError> {
    let mut module = module.clone();
    let global_count = module.global_variables.len();

    if let Some((stage, name)) = passes.entry_point {
        module
            .entry_points
            .retain(|ep| ep.stage == stage && ep.name == name);
    }

    let empty = PipelineConstants::default();
    let constants = passes.constants.unwrap_or(&empty);
    pipeline_constants::check_constants_exist(&module, constants)?;

    // Remove other entry points' items, which may use overrides without a value.
    let keep_unused = if passes.entry_point.is_some() || !module.overrides.is_empty() {
        KeepUnused::No
    } else {
        KeepUnused::Yes
    };
    let global_origins = compact_with_global_origins(&mut module, keep_unused);

    pipeline_constants::apply_overrides(&mut module, None, constants)?;

    // Passes add the functions they leave with unused or out-of-order
    // expressions to `dirty`.
    let mut dirty = Dirty::default();
    let inlined = crate::inline::inline_in_place(&mut module, passes.inline);
    dirty.functions.extend(inlined.functions);
    dirty.entry_points.extend(inlined.entry_points);
    reorder(&mut module, &dirty);

    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate_resolved_overrides(&module)?;
    Ok(Prepared {
        module,
        info,
        global_origins: (global_origins.len() != global_count).then_some(global_origins),
    })
}
