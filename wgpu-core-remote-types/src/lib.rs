//! This crate contains types for the remoting version of wgpu-core for browser implementing WebGPU.
//!
//! It contains types that are used for IPC communication between the browser's untrusted content process
//! and the browser's trusted GPU process and [`IdentityHub`](crate::identity::IdentityHub) for Id generation.
//! All in all it contains all types needed for content process.
//!
//! All IPC types implement `serde::Serialize` and `serde::Deserialize` so that they can be sent over IPC.
//!
//! Types are defined entirely separately from wgpu-core's (and eventually wgpu-types's) public types,
//! so that even as experimental features like raytracing and native wpgu features that are not standard WebGPU,
//! cannot be expressed in untrusted (thus potentially malicious) content processes,
//! as Serde's standard deserialization rejects invalid values of the IPC types.
//!
//! For more information about the remoting architecture, see the wgpu-core-remote crate's documentation.
extern crate alloc;
extern crate wgpu_types as wgt;

use alloc::borrow::Cow;

pub type Index = u32;
pub type Epoch = u32;
pub type SubmissionIndex = u64;
pub type SubmittedWorkDoneClosure = Box<dyn FnOnce() + Send + 'static>;

pub mod id;
pub mod identity;

pub mod binding_model;
pub mod encoders;

pub type Label<'a> = Option<Cow<'a, str>>;
