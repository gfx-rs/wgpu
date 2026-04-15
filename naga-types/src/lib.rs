#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod glsl;
pub mod hlsl;
pub mod msl;
pub mod spv;

use alloc::string::String;
