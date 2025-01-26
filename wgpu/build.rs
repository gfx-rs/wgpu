fn main() {
    cfg_aliases::cfg_aliases! {
        native: { not(target_arch = "wasm32") },

        unix_non_apple: { all(native, not(target_vendor = "apple"), unix) },
        Emscripten: { all(target_arch = "wasm32", target_os = "emscripten") },
        send_sync: { any(
            not(target_arch = "wasm32"),
            all(feature = "fragile-send-sync-non-atomic-wasm", not(target_feature = "atomics"))
        ) },

        webgl: { all(target_arch = "wasm32", not(target_os = "emscripten"), feature = "webgl") },
        webgpu: { all(target_arch = "wasm32", not(target_os = "emscripten"), feature = "webgpu") },
        dx12: { all(target_os = "windows", feature = "dx12") },
        metal: { all(target_vendor = "apple", feature = "metal") },
        vulkan: {
            any(all(feature = "vulkan-portability", target_vendor = "apple"),
                all(feature = "vulkan", unix_non_apple)
            )},
        gles: {
            any(
                all(feature = "angle", target_vendor = "apple"),
                all(feature = "gles", unix_non_apple)
            )
        },

        wgpu_core: { any(webgl, dx12, metal, vulkan, gles) },

        // This alias is _only_ if _we_ need naga in the wrapper. wgpu-core provides
        // its own re-export of naga, which can be used in other situations
        naga: { any(feature = "naga-ir", feature = "spirv", feature = "glsl") },
        // ⚠️ Keep in sync with target.cfg() definition in wgpu-hal/Cargo.toml and cfg_alias in `wgpu-hal` crate ⚠️
        static_dxc: { all(target_os = "windows", feature = "static-dxc", not(target_arch = "aarch64")) },
    }
}
