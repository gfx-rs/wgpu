//! Macros for validation internal to the resource tracker.
//!
//! This module defines assertion macros that respect `wgpu-core`'s
//! `"strict_asserts"` feature.
//!
//! Because `wgpu-core`'s public APIs validate their arguments in all
//! types of builds, for performance, the `track` module skips some of
//! Rust's usual run-time checks on its internal operations in release
//! builds. However, some `wgpu-core` applications have a strong
//! preference for robustness over performance. To accommodate them,
//! `wgpu-core`'s `"strict_asserts"` feature enables that validation
//! in both debug and release builds.

#[cfg(feature = "strict_asserts")]
macro_rules! strict_assert {
    ( $( $arg:tt )* ) => {
        assert!( $( $arg )* )
    }
}

#[cfg(feature = "strict_asserts")]
macro_rules! strict_assert_eq {
    ( $( $arg:tt )* ) => {
        assert_eq!( $( $arg )* )
    }
}

#[cfg(feature = "strict_asserts")]
macro_rules! strict_assert_ne {
    ( $( $arg:tt )* ) => {
        assert_ne!( $( $arg )* )
    }
}

#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert {
    ( $( $arg:tt )* ) => {};
}

#[cfg(not(feature = "strict_asserts"))]
macro_rules! strict_assert_eq {
    ( $( $arg:tt )* ) => {};
}

#[cfg(not(feature = "strict_asserts"))]
macro_rules! strict_assert_ne {
    ( $( $arg:tt )* ) => {};
}
