#[cfg(webgpu)]
mod webgpu;
#[cfg(webgpu)]
pub(crate) use webgpu::{get_browser_gpu_property, ContextWebGpu};

#[cfg(wgpu_core)]
mod wgpu_core;
#[cfg(wgpu_core)]
pub(crate) use wgpu_core::ContextWgpuCore;

#[cfg(wasi_webgpu)]
mod wasi_webgpu;
#[cfg(wasi_webgpu)]
pub(crate) use wasi_webgpu::ContextWasiWebgpu;
