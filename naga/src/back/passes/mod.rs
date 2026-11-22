/*!
IR passes that backends run on a module before writing it.

Each backend calls `run` on the module and [`ModuleInfo`] it is given, which
the caller may pass by reference or by value. The caller decides which entry
points the module contains, and must replace its overrides with constants
first, for example with
[`process_overrides`](crate::back::pipeline_constants::process_overrides).

A pass that changes a borrowed module works on a copy; an owned module is
changed in place. It may leave unused items,
forward references among a function's expressions, and calls to functions
that come later in the arena: once the passes are done, the runner compacts the
copy, which removes the unused items and puts the rest in order, and validates
it. If no pass changes anything, the backend writes the module and
[`ModuleInfo`] it was given, and doesn't revalidate.
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

/// Run `passes` on `module`, whose [`ModuleInfo`] is `info`.
///
/// Return the module the backend should write and its [`ModuleInfo`]. They are
/// `module` and `info` themselves if no pass changed anything.
pub(crate) fn run<'a>(
    module: Cow<'a, Module>,
    info: Cow<'a, ModuleInfo>,
    passes: &Passes,
) -> Result<(Cow<'a, Module>, Cow<'a, ModuleInfo>), PassError> {
    let Passes {} = *passes;
    // Each pass calls `Cow::to_mut` only if it will change the module, which
    // copies a borrowed module and is free for an owned one, and reports
    // whether it changed it.
    let changed = false;
    finish(module, info, changed)
}

/// Compact and validate `module` if a pass `changed` it.
///
/// Named expressions count as used, so a pass that leaves an expression
/// unused should not name it. Forward references must follow the rule
/// described in [`ReorderError`].
fn finish<'a>(
    module: Cow<'a, Module>,
    info: Cow<'a, ModuleInfo>,
    changed: bool,
) -> Result<(Cow<'a, Module>, Cow<'a, ModuleInfo>), PassError> {
    if !changed {
        return Ok((module, info));
    }
    let mut module = module.into_owned();
    // The caller chose what the module contains, so keep what's unused.
    try_compact(&mut module, KeepUnused::Yes)?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)?;
    Ok((Cow::Owned(module), Cow::Owned(info)))
}
