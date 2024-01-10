#[cfg(webgpu)]
mod webgpu;
#[cfg(webgpu)]
pub(crate) use webgpu::Context;

#[cfg(not(webgpu))]
mod wgpu_core;
#[cfg(not(webgpu))]
pub(crate) use wgpu_core::Context;
