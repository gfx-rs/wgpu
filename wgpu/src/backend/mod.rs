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
    all(
        not(target_arch = "wasm32"),
        any(
            not(any(target_os = "macos", target_os = "ios")),
            feature = "metal",
            feature = "vulkan-portability",
            feature = "angle"
        )
    ),
    target_os = "emscripten",
    feature = "webgl"
))]
mod direct;
#[cfg(any(
    all(
        not(target_arch = "wasm32"),
        any(
            not(any(target_os = "macos", target_os = "ios")),
            feature = "metal",
            feature = "vulkan-portability",
            feature = "angle"
        )
    ),
    target_os = "emscripten",
    feature = "webgl"
))]
pub(crate) use direct::Context;

#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    not(any(feature = "metal", feature = "vulkan-portability", feature = "angle"))
))]
mod dummy;
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    not(any(feature = "metal", feature = "vulkan-portability", feature = "angle"))
))]
pub(crate) use dummy::Context;
