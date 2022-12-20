//! Macros for validation internal to the wgpu.
//!
//! This module defines assertion macros that respect `wgpu-type`'s
//! `"strict_asserts"` feature.
//!
//! Because `wgpu-core`'s public APIs validate their arguments in all
//! types of builds, for performance, the `track` module skips some of
//! Rust's usual run-time checks on its internal operations in release
//! builds. However, some `wgpu-core` applications have a strong
//! preference for robustness over performance. To accommodate them,
//! `wgpu-core`'s `"strict_asserts"` feature enables that validation
//! in both debug and release builds.

/// This is equivalent to [`std::assert`] if the `strict_asserts` feature is activated, otherwise equal to [`std::debug_assert`].
#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert {
    ( $( $arg:tt )* ) => {
        assert!( $( $arg )* )
    }
}

/// This is equivalent to [`std::assert_eq`] if the `strict_asserts` feature is activated, otherwise equal to [`std::debug_assert_eq`].
#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert_eq {
    ( $( $arg:tt )* ) => {
        assert_eq!( $( $arg )* )
    }
}

/// This is equivalent to [`std::assert_ne`] if the `strict_asserts` feature is activated, otherwise equal to [`std::debug_assert_ne`].
#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert_ne {
    ( $( $arg:tt )* ) => {
        assert_ne!( $( $arg )* )
    }
}

/// This is equivalent to [`std::assert`] if the `strict_asserts` feature is activated, otherwise equal to [`std::debug_assert`]
#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert {
    ( $( $arg:tt )* ) => {
        debug_assert!( $( $arg )* )
    };
}

/// This is equivalent to [`std::assert_eq`] if the `strict_asserts` feature is activated, otherwise equal to [`std::debug_assert_eq`]
#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert_eq {
    ( $( $arg:tt )* ) => {
        debug_assert_eq!( $( $arg )* )
    };
}

/// This is equivalent to [`std::assert_ne`] if the `strict_asserts` feature is activated, otherwise equal to [`std::debug_assert_ne`]
#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert_ne {
    ( $( $arg:tt )* ) => {
        debug_assert_ne!( $( $arg )* )
    };
}

/// Unwrapping using strict_asserts
pub trait StrictAssertUnwrapExt<T> {
    /// Unchecked unwrap, with a [`strict_assert`] backed assertion of validitly.
    ///
    /// # Safety
    ///
    /// It _must_ be valid to call unwrap_unchecked on this value.
    unsafe fn strict_unwrap_unchecked(self) -> T;
}

impl<T> StrictAssertUnwrapExt<T> for Option<T> {
    unsafe fn strict_unwrap_unchecked(self) -> T {
        strict_assert!(self.is_some(), "Called strict_unwrap_unchecked on None");
        // SAFETY: Checked by above assert, or by assertion by unsafe.
        unsafe { self.unwrap_unchecked() }
    }
}

impl<T, E> StrictAssertUnwrapExt<T> for Result<T, E> {
    unsafe fn strict_unwrap_unchecked(self) -> T {
        strict_assert!(self.is_ok(), "Called strict_unwrap_unchecked on Err");
        // SAFETY: Checked by above assert, or by assertion by unsafe.
        unsafe { self.unwrap_unchecked() }
    }
}
