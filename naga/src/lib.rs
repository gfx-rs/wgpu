/*! Universal shader translator.


*/

#![allow(
    clippy::new_without_default,
    clippy::unneeded_field_pattern,
    clippy::match_like_matches_macro,
    clippy::collapsible_if,
    clippy::derive_partial_eq_without_eq,
    clippy::needless_borrowed_reference,
    clippy::single_match,
    clippy::enum_variant_names
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    clippy::pattern_type_mismatch,
    clippy::missing_const_for_fn,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::match_wildcard_for_single_variants
)]
#![deny(clippy::exit)]
#![cfg_attr(
    not(test),
    warn(
        clippy::dbg_macro,
        clippy::panic,
        clippy::print_stderr,
        clippy::print_stdout,
        clippy::todo
    )
)]
#![no_std]

#[cfg(any(
    test,
    spv_out,

    // Need OnceLock
    hlsl_out,
    msl_out,
    wgsl_out,

    feature = "spv-in",
    feature = "wgsl-in"
))]
extern crate std;

extern crate alloc;

mod arena;
pub mod back;
pub mod common;
#[cfg(feature = "compact")]
pub mod compact;
pub mod diagnostic_filter;
pub mod error;
pub mod front;
pub mod ir;
pub mod keywords;
mod non_max_u32;
pub mod proc;
mod span;
pub mod valid;

use alloc::string::String;

pub use crate::arena::{Arena, Handle, Range, UniqueArena};
pub use crate::span::{SourceLocation, Span, SpanContext, WithSpan};

// TODO: Eliminate this re-export and migrate uses of `crate::Foo` to `use crate::ir; ir::Foo`.
pub use ir::*;

/// Width of a boolean type, in bytes.
pub const BOOL_WIDTH: Bytes = 1;

/// Width of abstract types, in bytes.
pub const ABSTRACT_WIDTH: Bytes = 8;

/// Hash map that is faster but not resilient to DoS attacks.
/// (Similar to rustc_hash::FxHashMap but using hashbrown::HashMap instead of alloc::collections::HashMap.)
/// To construct a new instance: `FastHashMap::default()`
pub type FastHashMap<K, T> =
    hashbrown::HashMap<K, T, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// Hash set that is faster but not resilient to DoS attacks.
/// (Similar to rustc_hash::FxHashSet but using hashbrown::HashSet instead of alloc::collections::HashMap.)
pub type FastHashSet<K> =
    hashbrown::HashSet<K, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// Insertion-order-preserving hash set (`IndexSet<K>`), but with the same
/// hasher as `FastHashSet<K>` (faster but not resilient to DoS attacks).
pub type FastIndexSet<K> =
    indexmap::IndexSet<K, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// Insertion-order-preserving hash map (`IndexMap<K, V>`), but with the same
/// hasher as `FastHashMap<K, V>` (faster but not resilient to DoS attacks).
pub type FastIndexMap<K, V> =
    indexmap::IndexMap<K, V, core::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// Map of expressions that have associated variable names
pub(crate) type NamedExpressions = FastIndexMap<Handle<Expression>, String>;
