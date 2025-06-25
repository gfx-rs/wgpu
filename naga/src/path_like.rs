//! [`PathLike`] and its supporting items, such as [`PathLikeRef`] and [`PathLikeOwned`].
//! This trait and these types provide a common denominator API for `Path`-like
//! types and operations in `std` and `no_std` contexts.
//!
//! # Usage
//!
//! - Store a [`PathLikeRef<'a>`] instead of a `&'a Path` in structs and enums.
//! - Store a [`PathLikeOwned`] instead of a `PathBuf` in structs and enums.
//! - Accept `impl PathLike` instead of `impl AsRef<Path>` for methods which directly
//!   work with `Path`-like values.
//! - Accept `Into<PathLikeRef<'_>>` and/or `Into<PathLikeOwned>` in methods which
//!   will store a `Path`-like value.

use alloc::{borrow::Cow, string::String};
use core::fmt;

mod sealed {
    /// Seal for [`PathLike`](super::PathLike).
    pub trait Sealed {}
}

/// A trait that abstracts over types accepted for conversion to the most
/// featureful path representation possible; that is:
///
/// - When `no_std` is active, this is implemented for:
///   - [`str`],
///   - [`String`],
///   - [`Cow<'_, str>`], and
///   - [`PathLikeRef`]
/// - Otherwise, types that implement `AsRef<Path>` (to extract a `&Path`).
///
/// This type is used as the type bounds for various diagnostic rendering methods, i.e.,
/// [`WithSpan::emit_to_string_with_path`](crate::span::WithSpan::emit_to_string_with_path).
pub trait PathLike: sealed::Sealed {
    fn to_string_lossy(&self) -> Cow<'_, str>;
}

/// Abstraction over `Path` which falls back to [`str`] for `no_std` compatibility.
///
/// This type should be used for _storing_ a reference to a [`PathLike`].
/// Functions which accept a `Path` should prefer to use `impl PathLike`
/// or `impl Into<PathLikeRef<'_>>`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathLikeRef<'a>(&'a path_like_impls::PathInner);

impl fmt::Debug for PathLikeRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

impl<'a> From<&'a str> for PathLikeRef<'a> {
    fn from(value: &'a str) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(std)] {
                Self(std::path::Path::new(value))
            } else {
                Self(value)
            }
        }
    }
}

/// Abstraction over `PathBuf` which falls back to [`String`] for `no_std` compatibility.
///
/// This type should be used for _storing_ an owned [`PathLike`].
/// Functions which accept a `PathBuf` should prefer to use `impl PathLike`
/// or `impl Into<PathLikeOwned>`.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathLikeOwned(<path_like_impls::PathInner as alloc::borrow::ToOwned>::Owned);

impl fmt::Debug for PathLikeOwned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl From<String> for PathLikeOwned {
    fn from(value: String) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(std)] {
                Self(value.into())
            } else {
                Self(value)
            }
        }
    }
}

#[cfg(std)]
mod path_like_impls {
    //! Implementations of [`PathLike`] within an `std` context.
    //!
    //! Since `std` is available, we blanket implement [`PathLike`] for all types
    //! implementing [`AsRef<Path>`].

    use alloc::borrow::Cow;
    use std::path::Path;

    use super::{sealed, PathLike};

    pub(super) type PathInner = Path;

    impl<T: AsRef<Path> + ?Sized> PathLike for T {
        fn to_string_lossy(&self) -> Cow<'_, str> {
            self.as_ref().to_string_lossy()
        }
    }

    impl<T: AsRef<Path> + ?Sized> sealed::Sealed for T {}
}

#[cfg(no_std)]
mod path_like_impls {
    //! Implementations of [`PathLike`] within a `no_std` context.
    //!
    //! Without `std`, we cannot blanket implement on [`AsRef<Path>`].
    //! Instead, we manually implement for a subset of types which are known
    //! to implement [`AsRef<Path>`] when `std` is available.
    //!
    //! Implementing [`PathLike`] for a type which does _not_ implement [`AsRef<Path>`]
    //! with `std` enabled breaks the additive requirement of Cargo features.

    use alloc::{borrow::Cow, string::String};
    use core::borrow::Borrow;

    use super::{sealed, PathLike, PathLikeOwned, PathLikeRef};

    pub(super) type PathInner = str;

    impl PathLike for String {
        fn to_string_lossy(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.as_str())
        }
    }

    impl sealed::Sealed for String {}

    impl PathLike for str {
        fn to_string_lossy(&self) -> Cow<'_, str> {
            Cow::Borrowed(self)
        }
    }

    impl sealed::Sealed for str {}

    impl PathLike for Cow<'_, str> {
        fn to_string_lossy(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.borrow())
        }
    }

    impl sealed::Sealed for Cow<'_, str> {}

    impl<T: PathLike + ?Sized> PathLike for &T {
        fn to_string_lossy(&self) -> Cow<'_, str> {
            (*self).to_string_lossy()
        }
    }

    impl<T: PathLike + ?Sized> sealed::Sealed for &T {}

    impl PathLike for PathLikeRef<'_> {
        fn to_string_lossy(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0)
        }
    }

    impl sealed::Sealed for PathLikeRef<'_> {}

    impl PathLike for PathLikeOwned {
        fn to_string_lossy(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0.borrow())
        }
    }

    impl sealed::Sealed for PathLikeOwned {}
}

#[cfg(std)]
mod path_like_owned_std_impls {
    //! Traits which can only be implemented for [`PathLikeOwned`] with `std`.

    use std::path::{Path, PathBuf};

    use super::PathLikeOwned;

    impl AsRef<Path> for PathLikeOwned {
        fn as_ref(&self) -> &Path {
            self.0.as_ref()
        }
    }

    impl From<PathBuf> for PathLikeOwned {
        fn from(value: PathBuf) -> Self {
            Self(value)
        }
    }

    impl From<PathLikeOwned> for PathBuf {
        fn from(value: PathLikeOwned) -> Self {
            value.0
        }
    }

    impl AsRef<PathBuf> for PathLikeOwned {
        fn as_ref(&self) -> &PathBuf {
            &self.0
        }
    }
}

#[cfg(std)]
mod path_like_ref_std_impls {
    //! Traits which can only be implemented for [`PathLikeRef`] with `std`.

    use std::path::Path;

    use super::PathLikeRef;

    impl AsRef<Path> for PathLikeRef<'_> {
        fn as_ref(&self) -> &Path {
            self.0
        }
    }

    impl<'a> From<&'a Path> for PathLikeRef<'a> {
        fn from(value: &'a Path) -> Self {
            Self(value)
        }
    }

    impl<'a> From<PathLikeRef<'a>> for &'a Path {
        fn from(value: PathLikeRef<'a>) -> Self {
            value.0
        }
    }
}
