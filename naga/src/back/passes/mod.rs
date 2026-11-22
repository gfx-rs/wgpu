/*!
IR passes that backends run on a module before writing it.

Each backend calls `run` on the module it is given, which the caller may pass
by reference or by value. The caller decides which entry points the module
contains, must replace its overrides with constants first, for example with
[`process_overrides`](crate::back::pipeline_constants::process_overrides), and
must have validated it. Backends don't take a [`ModuleInfo`]: `run` validates
the module after the passes and returns its info.

A pass that changes a borrowed module works on a copy; an owned module is
changed in place. A pass may leave unused items, forward references among a
function's expressions, and calls to functions that come later in the arena:
if any pass changed the module, the runner compacts it, which removes the
unused items and puts the rest in order, before validating it.
*/
#![cfg_attr(
    not(any(glsl_out, hlsl_out, msl_out, spv_out)),
    allow(
        dead_code,
        reason = "shared helpers can be dead if none of the enabled backends need it"
    )
)]

mod emit;

use alloc::{borrow::Cow, boxed::Box};

use crate::{
    compact::{try_compact, KeepUnused, ReorderError},
    valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags, Validator},
    Module, WithSpan,
};

/// An error produced while running the passes a backend requested.
#[derive(Clone, Debug, thiserror::Error)]
pub enum PassError {
    #[error(transparent)]
    Reorder(#[from] ReorderError),
    #[error("module failed revalidation after backend passes")]
    Validation(#[from] Box<WithSpan<ValidationError>>),
}

/// The passes a backend requests.
#[derive(Clone, Debug, Default)]
pub(crate) struct Passes {}

/// Run `passes` on `module`, and validate the result.
///
/// Return the module the backend should write, which is `module` itself if no
/// pass changed anything, and its [`ModuleInfo`].
pub(crate) fn run<'a>(
    module: Cow<'a, Module>,
    passes: &Passes,
) -> Result<(Cow<'a, Module>, ModuleInfo), PassError> {
    let Passes {} = *passes;
    // Each pass calls `Cow::to_mut` only if it will change the module, which
    // copies a borrowed module and is free for an owned one, and reports
    // whether it changed it.
    let changed = false;
    finish(module, changed)
}

/// Compact `module` if a pass `changed` it, and validate it.
///
/// Named expressions count as used, so a pass that leaves an expression
/// unused should not name it. Forward references must follow the rule
/// described in [`ReorderError`].
fn finish(
    mut module: Cow<'_, Module>,
    changed: bool,
) -> Result<(Cow<'_, Module>, ModuleInfo), PassError> {
    if changed {
        // The caller chose what the module contains, so keep what's unused.
        try_compact(module.to_mut(), KeepUnused::Yes)?;
    }
    let info = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)?;
    Ok((module, info))
}
