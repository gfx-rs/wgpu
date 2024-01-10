#[cfg(webgpu)]
mod webgpu;
#[cfg(webgpu)]
pub(crate) type Context = webgpu::ContextWebGpu;

#[cfg(not(webgpu))]
mod wgpu_core;
#[cfg(not(webgpu))]
pub(crate) type Context = wgpu_core::ContextWgpuCore;
