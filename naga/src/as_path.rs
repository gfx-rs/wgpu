//! [`AsPath`] and its supporting items.

use alloc::borrow::Cow;

#[cfg(feature = "std")]
use std::path::Path;

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// A trait that abstracts over types accepted for conversion to the most
/// featureful path representation possible; that is:
///
/// - When `no_std` is active, this represents types can be converted to `Cow<'_, str>`.
/// - Otherwise, types that implement `AsRef<Path>` (to extract a `&Path`).
///
/// This type is used as the type bounds for various diagnostic rendering methods, i.e.,
/// [`WithSpan::emit_to_string_with_path`](crate::span::WithSpan::emit_to_string_with_path).
pub trait AsPath {
    fn to_string_lossy(&self) -> Cow<'_, str>;
}

#[cfg(feature = "std")]
impl<T: AsRef<Path> + ?Sized> AsPath for T {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        self.as_ref().to_string_lossy()
    }
}

#[cfg(not(feature = "std"))]
impl AsPath for String {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.as_str())
    }
}

#[cfg(not(feature = "std"))]
impl AsPath for str {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        Cow::Borrowed(self)
    }
}

#[cfg(not(feature = "std"))]
impl AsPath for Cow<'_, str> {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        use core::borrow::Borrow;
        Cow::Borrowed(self.borrow())
    }
}

#[cfg(not(feature = "std"))]
impl<T: AsPath + ?Sized> AsPath for &T {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        (*self).to_string_lossy()
    }
}
