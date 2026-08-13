#![allow(
    // Need many arguments for some core functions to be able to re-use code in many situations.
    clippy::too_many_arguments,
)]
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
