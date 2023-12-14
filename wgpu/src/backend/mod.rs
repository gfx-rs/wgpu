#[cfg(all(
    target_arch = "wasm32",
    not(any(target_os = "emscripten", feature = "webgl"))
))]
mod web;
#[cfg(all(
    target_arch = "wasm32",
    not(any(target_os = "emscripten", feature = "webgl"))
))]
pub(crate) use web::Context;

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
pub(crate) use direct::Context;
