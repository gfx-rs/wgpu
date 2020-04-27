#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub(crate) use web::Context;

#[cfg(not(target_arch = "wasm32"))]
mod direct;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use direct::Context;

#[cfg(not(target_arch = "wasm32"))]
mod native_gpu_future;
