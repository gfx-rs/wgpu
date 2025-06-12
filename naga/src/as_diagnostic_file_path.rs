//! [`AsDiagnosticFilePath`] and its supporting items.

use alloc::borrow::Cow;

#[cfg(feature = "std")]
use std::path::Path;

#[cfg(not(feature = "std"))]
use alloc::string::String;

mod sealed {
    pub trait Sealed {}
}

/// A trait that abstracts over types accepted for conversion to the most
/// featureful path representation possible; that is:
///
/// - When `no_std` is active, this is implemented for [`String`], [`str`], and [`Cow`] (i.e.,
///   `Cow<'_, str>`).
/// - Otherwise, types that implement `AsRef<Path>` (to extract a `&Path`).
///
/// This type is used as the type bounds for various diagnostic rendering methods, i.e.,
/// [`WithSpan::emit_to_string_with_path`](crate::span::WithSpan::emit_to_string_with_path).
///
/// [`String`]: alloc::string::String
pub trait AsDiagnosticFilePath: sealed::Sealed {
    fn to_string_lossy(&self) -> Cow<'_, str>;
}

#[cfg(feature = "std")]
impl<T: AsRef<Path> + ?Sized> AsDiagnosticFilePath for T {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        self.as_ref().to_string_lossy()
    }
}

#[cfg(feature = "std")]
impl<T: AsRef<Path> + ?Sized> sealed::Sealed for T {}

#[cfg(not(feature = "std"))]
impl AsDiagnosticFilePath for String {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.as_str())
    }
}

#[cfg(not(feature = "std"))]
impl sealed::Sealed for String {}

#[cfg(not(feature = "std"))]
impl AsDiagnosticFilePath for str {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        Cow::Borrowed(self)
    }
}

#[cfg(not(feature = "std"))]
impl sealed::Sealed for str {}

#[cfg(not(feature = "std"))]
impl AsDiagnosticFilePath for Cow<'_, str> {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        use core::borrow::Borrow;
        Cow::Borrowed(self.borrow())
    }
}

#[cfg(not(feature = "std"))]
impl sealed::Sealed for Cow<'_, str> {}

#[cfg(not(feature = "std"))]
impl<T: AsDiagnosticFilePath + ?Sized> AsDiagnosticFilePath for &T {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        (*self).to_string_lossy()
    }
}

#[cfg(not(feature = "std"))]
impl<T: AsDiagnosticFilePath + ?Sized> sealed::Sealed for &T {}
