fn main() {
    cfg_aliases::cfg_aliases! {
        native: { not(target_arch = "wasm32") },
        webgl: { all(target_arch = "wasm32", not(target_os = "emscripten"), feature = "webgl") },
        webgpu: { all(target_arch = "wasm32", not(target_os = "emscripten"), feature = "webgpu") },
        Emscripten: { all(target_arch = "wasm32", target_os = "emscripten") },
        wgpu_core: { any(native, webgl, Emscripten) },
        send_sync: { any(
            not(target_arch = "wasm32"),
            all(feature = "fragile-send-sync-non-atomic-wasm", not(target_feature = "atomics"))
        ) },
        dx12: { all(target_os = "windows", feature = "dx12") },
        metal: { all(any(target_os = "ios", target_os = "macos"), feature = "metal") },
        // This alias is _only_ if _we_ need naga in the wrapper. wgpu-core provides
        // its own re-export of naga, which can be used in other situations
        naga: { any(feature = "naga-ir", feature = "spirv", feature = "glsl") },
    }
}
