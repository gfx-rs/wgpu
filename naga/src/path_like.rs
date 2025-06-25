//! [`PathLike`] and its supporting items, such as [`PathLikeRef`] and [`PathLikeOwned`].

use alloc::borrow::Cow;
use core::fmt;

mod sealed {
    pub trait Sealed {}
}

/// A trait that abstracts over types accepted for conversion to the most
/// featureful path representation possible; that is:
///
/// - When `no_std` is active, this is implemented for:
///   - [`str`],
///   - [`String`](alloc::string::String),
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
/// Functions which accept a `Path` should prefer to use `impl PathLike`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathLikeRef<'a>(&'a impls::PathInner);

impl fmt::Debug for PathLikeRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

/// Abstraction over `PathBuf` which falls back to [`String`](alloc::string::String)
/// for `no_std` compatibility.
///
/// This type should be used for _storing_ an owned [`PathLike`].
/// Functions which accept a `PathBuf` should prefer to use `impl PathLike`.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathLikeOwned(<impls::PathInner as alloc::borrow::ToOwned>::Owned);

impl fmt::Debug for PathLikeOwned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

#[cfg(std)]
mod impls {
    use alloc::{borrow::Cow, string::String};
    use std::path::{Path, PathBuf};

    use super::{sealed, PathLike, PathLikeOwned, PathLikeRef};

    pub(super) type PathInner = Path;

    impl<T: AsRef<Path> + ?Sized> PathLike for T {
        fn to_string_lossy(&self) -> Cow<'_, str> {
            self.as_ref().to_string_lossy()
        }
    }

    impl<T: AsRef<Path> + ?Sized> sealed::Sealed for T {}

    impl AsRef<Path> for PathLikeRef<'_> {
        fn as_ref(&self) -> &Path {
            self.0
        }
    }

    impl AsRef<Path> for PathLikeOwned {
        fn as_ref(&self) -> &Path {
            self.0.as_ref()
        }
    }

    impl<'a> From<&'a str> for PathLikeRef<'a> {
        fn from(value: &'a str) -> Self {
            Self(Path::new(value))
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

    impl From<String> for PathLikeOwned {
        fn from(value: String) -> Self {
            Self(PathBuf::from(value))
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

#[cfg(no_std)]
mod impls {
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

    impl<'a> From<&'a str> for PathLikeRef<'a> {
        fn from(value: &'a str) -> Self {
            Self(value)
        }
    }

    impl From<String> for PathLikeOwned {
        fn from(value: String) -> Self {
            Self(value)
        }
    }
}
