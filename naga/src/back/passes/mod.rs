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
unused items and puts the rest in order, before validating it. With
[`PassOptions::compact`], the runner always compacts, and removes everything
the entry points don't use.
*/
#![cfg_attr(
    not(any(glsl_out, hlsl_out, msl_out, spv_out)),
    allow(
        dead_code,
        reason = "shared helpers can be dead if none of the enabled backends need it"
    )
)]

mod emit;
pub mod workgroup_zero_init;

use alloc::{borrow::Cow, boxed::Box};

use crate::{
    compact::{try_compact, Dropped, KeepUnused, ReorderError},
    valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags, Validator},
    Module, WithSpan,
};

/// An error produced while running the passes a backend requested.
#[derive(Clone, Debug, thiserror::Error)]
pub enum PassError {
    #[error(transparent)]
    Reorder(#[from] ReorderError),
    #[error(transparent)]
    WorkgroupZeroInit(#[from] workgroup_zero_init::WorkgroupZeroInitError),
    #[error("module failed revalidation after backend passes")]
    Validation(#[from] Box<WithSpan<ValidationError>>),
}

/// Options for the IR passes a backend runs before writing a module.
///
/// The SPIR-V, HLSL, MSL and GLSL backends' `Options` each hold one.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default))]
pub struct PassOptions {
    /// After the passes, remove everything the module's entry points don't
    /// use, so the output doesn't depend on what unused items the module had.
    ///
    /// If `false`, the module is compacted only if a pass changed it, and then
    /// unused items the caller left in are kept.
    pub compact: bool,
}

/// The passes a backend requests.
#[derive(Clone, Debug, Default)]
pub(crate) struct Passes {
    /// See [`PassOptions::compact`].
    pub compact: bool,
    /// Run [`workgroup_zero_init`].
    pub workgroup_zero_init: bool,
}

impl From<PassOptions> for Passes {
    fn from(options: PassOptions) -> Self {
        let PassOptions { compact } = options;
        Self {
            compact,
            workgroup_zero_init: false,
        }
    }
}

/// Run `passes` on `module`, and validate the result.
///
/// Return the module the backend should write, which is `module` itself if no
/// pass changed anything, and its [`ModuleInfo`].
pub(crate) fn run<'a>(
    mut module: Cow<'a, Module>,
    passes: &Passes,
) -> Result<(Cow<'a, Module>, ModuleInfo), PassError> {
    let Passes {
        compact,
        workgroup_zero_init,
    } = *passes;
    // Each pass calls `Cow::to_mut` only if it will change the module, which
    // copies a borrowed module and is free for an owned one, reports whether
    // it changed it, and adds what it left for compaction to remove to
    // `dropped`.
    let mut changed = false;
    let dropped = Dropped::default();
    if workgroup_zero_init {
        changed |= workgroup_zero_init::run(&mut module)?;
    }
    finish(module, changed, compact, &dropped)
}

/// Compact `module` if a pass `changed` it or `compact` is set, and validate
/// it.
///
/// Compaction removes the items in `dropped`, and with `compact` everything
/// else unused too. Named expressions count as used, so a pass that leaves an
/// expression unused should not name it. Forward references must follow the
/// rule described in [`ReorderError`].
fn finish<'a>(
    mut module: Cow<'a, Module>,
    changed: bool,
    compact: bool,
    dropped: &Dropped,
) -> Result<(Cow<'a, Module>, ModuleInfo), PassError> {
    if compact {
        try_compact(module.to_mut(), KeepUnused::No, dropped)?;
    } else if changed {
        // The caller chose what the module contains, so keep what's unused.
        try_compact(module.to_mut(), KeepUnused::Yes, dropped)?;
    }
    let info = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)?;
    Ok((module, info))
}

/// Parse and validate `source`, run `pass` on it, and then compact and
/// validate the result as [`run`] does.
///
/// Return `None` if the pass reported that it left the module unchanged, in
/// which case it must not have copied it either.
#[cfg(all(test, feature = "wgsl-in"))]
pub(crate) fn run_pass_on_wgsl<E>(
    source: &str,
    pass: impl FnOnce(&mut Cow<'_, Module>) -> Result<bool, E>,
) -> Result<Option<Module>, PassError>
where
    PassError: From<E>,
{
    let module = crate::front::wgsl::parse_str(source).unwrap();
    let mut prepared = Cow::Borrowed(&module);
    let changed = pass(&mut prepared)?;
    if !changed {
        assert!(
            matches!(prepared, Cow::Borrowed(_)),
            "unchanged module was copied"
        );
        return Ok(None);
    }
    let (prepared, _) = finish(prepared, changed, false, &Dropped::default())?;
    Ok(Some(prepared.into_owned()))
}
