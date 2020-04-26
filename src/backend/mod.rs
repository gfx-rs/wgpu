#[cfg(target_arch = "wasm32")]
mod web;
//#[cfg(target_arch = "wasm32")]
//pub use web::*;

#[cfg(not(target_arch = "wasm32"))]
mod direct;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use direct::{
    buffer_map_read, buffer_map_write, request_adapter, request_device_and_queue, Context,
};

#[cfg(not(target_arch = "wasm32"))]
mod native_gpu_future;
