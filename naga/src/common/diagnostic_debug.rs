//! Displaying Naga IR terms in debugging output.

#[cfg(any(feature = "wgsl-in", feature = "wgsl-out"))]
use crate::common::wgsl::TypeContext;

use crate::proc::TypeResolution;
use crate::{Handle, Scalar, Type, TypeInner, UniqueArena};

use core::fmt;

/// A wrapper for displaying Naga IR terms in debugging output.
///
/// This is like [`DiagnosticDisplay`], but requires weaker context
/// and produces correspondingly lower-fidelity output. For example,
/// this cannot show the override names for override-sized array
/// lengths.
///
/// [`DiagnosticDisplay`]: super::DiagnosticDisplay
pub struct DiagnosticDebug<T>(pub T);

impl fmt::Debug for DiagnosticDebug<(Handle<Type>, &UniqueArena<Type>)> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (handle, ctx) = self.0;

        #[cfg(any(feature = "wgsl-in", feature = "wgsl-out"))]
        ctx.write_type(handle, f)?;

        #[cfg(not(any(feature = "wgsl-in", feature = "wgsl-out")))]
        {
            let _ = ctx;
            write!(f, "{handle:?}")?;
        }

        Ok(())
    }
}

impl fmt::Debug for DiagnosticDebug<(&TypeInner, &UniqueArena<Type>)> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (inner, ctx) = self.0;

        #[cfg(any(feature = "wgsl-in", feature = "wgsl-out"))]
        ctx.write_type_inner(inner, f)?;

        #[cfg(not(any(feature = "wgsl-in", feature = "wgsl-out")))]
        {
            let _ = ctx;
            write!(f, "{inner:?}")?;
        }

        Ok(())
    }
}

impl fmt::Debug for DiagnosticDebug<(&TypeResolution, &UniqueArena<Type>)> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (resolution, ctx) = self.0;

        #[cfg(any(feature = "wgsl-in", feature = "wgsl-out"))]
        ctx.write_type_resolution(resolution, f)?;

        #[cfg(not(any(feature = "wgsl-in", feature = "wgsl-out")))]
        {
            let _ = ctx;
            write!(f, "{resolution:?}")?;
        }

        Ok(())
    }
}

impl fmt::Debug for DiagnosticDebug<Scalar> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scalar = self.0;

        #[cfg(any(feature = "wgsl-in", feature = "wgsl-out"))]
        f.write_str(&crate::common::wgsl::TryToWgsl::to_wgsl_for_diagnostics(
            scalar,
        ))?;

        #[cfg(not(any(feature = "wgsl-in", feature = "wgsl-out")))]
        write!(f, "{scalar:?}")?;

        Ok(())
    }
}
