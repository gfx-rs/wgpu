/*!
This crate is a remoting version of [`wgpu_core`] for browser implementing WebGPU.

Modern browsers have content processes that are untrusted and cannot directly access the GPU.
Instead, they communicate with a trusted GPU process that has access to the GPU.

wgpu-core-remote provides a remoting wrapper around [`wgpu_core`] named [`Global`](crate::global::Global),
that is used in the GPU process to execute commands from the content process.
All stuff that is needed in the content process is provided in the [`wgpu_core_remote_types`] crate.

The key part of remoting are [`Id`](crate::id::Id) which are described in [`crate::hub`].
*/

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

pub mod global;
pub mod hub;
pub mod id;
pub mod registry;
pub mod storage;
