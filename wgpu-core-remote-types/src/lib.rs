extern crate alloc;
extern crate wgpu_types as wgt;

use alloc::borrow::Cow;

pub type Index = u32;
pub type Epoch = u32;

pub mod id;
pub mod identity;

pub mod binding_model;

pub type Label<'a> = Option<Cow<'a, str>>;
