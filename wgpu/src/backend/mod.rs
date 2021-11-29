#[cfg(all(
    target_arch = "wasm32",
    not(target_os = "emscripten"),
    not(feature = "webgl")
))]
mod web;
#[cfg(all(
    target_arch = "wasm32",
    not(target_os = "emscripten"),
    not(feature = "webgl")
))]
pub(crate) use web::{BufferMappedRange, Context};

#[cfg(any(
    not(target_arch = "wasm32"),
    target_os = "emscripten",
    feature = "webgl"
))]
mod direct;
#[cfg(any(
    not(target_arch = "wasm32"),
    target_os = "emscripten",
    feature = "webgl"
))]
pub(crate) use direct::{BufferMappedRange, Context};

#[cfg(any(
    not(target_arch = "wasm32"),
    target_os = "emscripten",
    feature = "webgl"
))]
mod native_gpu_future;
