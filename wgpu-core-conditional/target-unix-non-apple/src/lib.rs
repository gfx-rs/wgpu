//! No code. See README.md for details.

#[cfg(all(unix, not(target_os = "emscripten"), not(target_vendor = "apple")))]
pub use wgpu_core::*;
