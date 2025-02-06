#![cfg(target_vendor = "apple")]

#[cfg(feature = "angle")]
pub use bytemuck;
#[cfg(feature = "angle")]
pub use glow;
#[cfg(feature = "angle")]
pub use khronos_egl;
#[cfg(feature = "angle")]
pub use libloading;

#[cfg(feature = "vulkan-portability")]
pub use ash;
#[cfg(feature = "vulkan-portability")]
pub use gpu_alloc;
#[cfg(feature = "vulkan-portability")]
pub use gpu_descriptor;
#[cfg(feature = "vulkan-portability")]
pub use libc;
#[cfg(feature = "vulkan-portability")]
pub use ordered_float;
#[cfg(feature = "vulkan-portability")]
pub use smallvec;
