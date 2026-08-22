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
