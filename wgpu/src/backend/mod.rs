#[cfg(webgpu)]
mod webgpu;
#[cfg(webgpu)]
pub(crate) use webgpu::ContextWebGpu;

#[cfg(wgpu_core)]
mod wgpu_core;
#[cfg(wgpu_core)]
pub(crate) use wgpu_core::ContextWgpuCore;
