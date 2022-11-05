#[cfg(all(target_arch = "wasm32", not(feature = "webgl")))]
mod web;
#[cfg(all(target_arch = "wasm32", not(feature = "webgl")))]
pub(crate) use web::{BufferMappedRange, Context, QueueWriteBuffer};

#[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]
mod direct;
#[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]
pub(crate) use direct::{BufferMappedRange, Context, QueueWriteBuffer};
