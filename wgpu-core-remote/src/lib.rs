#![allow(
    // It is much clearer to assert negative conditions with eq! false
    clippy::bool_assert_comparison,
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
    // Redundant matching is more explicit.
    clippy::redundant_pattern_matching,
    // Explicit lifetimes are often easier to reason about.
    clippy::needless_lifetimes,
    // No need for defaults in the internal types.
    clippy::new_without_default,
    // Needless updates are more scalable, easier to play with features.
    clippy::needless_update,
    // Need many arguments for some core functions to be able to re-use code in many situations.
    clippy::too_many_arguments,
    // It gets in the way a lot and does not prevent bugs in practice.
    clippy::pattern_type_mismatch,
    // `wgpu-core` isn't entirely user-facing, so it's useful to document internal items.
    rustdoc::private_intra_doc_links,
)]
#![expect(missing_debug_implementations, reason = "TODO")]
#![warn(
    clippy::alloc_instead_of_core,
    clippy::ptr_as_ptr,
    clippy::std_instead_of_alloc,
    clippy::std_instead_of_core,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_op_in_unsafe_fn,
    unused_extern_crates,
    unused_qualifications
)]

extern crate alloc;
extern crate wgpu_hal as hal;
extern crate wgpu_types as wgt;

pub type TexelCopyBufferInfo = wgt::TexelCopyBufferInfo<id::BufferId>;
pub type TexelCopyTextureInfo = wgt::TexelCopyTextureInfo<id::TextureId>;
pub type CopyExternalImageDestInfo = wgt::CopyExternalImageDestInfo<id::TextureId>;

pub type Command = wgpu_core::command::Command<id::IdReferences>;

pub mod global;
pub mod hub;
pub mod id;
pub mod identity;
pub mod registry;
pub mod storage;

type Index = u32;
type Epoch = u32;
