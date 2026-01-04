fn main() {
    cfg_aliases::cfg_aliases! {
        // platforms
        android_platform: { target_os = "android" },
        ohos_platform: { target_env = "ohos" },
        wasm_platform: { target_family = "wasm" },
        emuscripten_platform: { target_os = "emscripten" },
        macos_platform: { target_os = "macos" },
        ios_platform: { target_os = "ios" },
        apple: { any(ios_platform, macos_platform) },
        free_unix: { all(unix, not(apple), not(android_platform), not(ohos_platform)) },

        native: { not(wasm_platform) },
        send_sync: { any(
            not(wasm_platform),
            all(feature = "fragile-send-sync-non-atomic-wasm", not(target_feature = "atomics"))
        ) },
        webgl: { all(wasm_platform, not(emuscripten_platform), gles) },
        Emscripten: { all(emuscripten_platform, gles) },
        dx12: { all(windows, feature = "dx12") },
        gles: { all(feature = "gles") },
        // Within the GL ES backend, use `std` and be Send + Sync only if we are using a target
        // that, among the ones where the GL ES backend is supported, has `std`.
        gles_with_std: { all(
            feature = "gles",
            any(
                not(wasm_platform),
                // Accept wasm32-unknown-unknown, which uniquely has a stub `std`
                all(target_vendor = "unknown", target_os = "unknown"),
                // Accept wasm32-unknown-emscripten and similar, which has a real `std`
                emuscripten_platform
            )
        ) },
        // GLES_Backends.
        gles_egl_backend: { all(feature = "gles", any(windows, unix), not(apple), not(wasm_platform)) },
        // Not Support GLX, TODO: Add support for glx
        // gles_glx_backend: { all(feature = "gles", feature = "glx", x11_platform, not(wasm_platform)) },
        gles_wgl_backend: { all(feature = "gles", windows, not(wasm_platform)) },
        gles_cgl_backend: { all(feature = "gles", macos_platform, not(wasm_platform)) },

        metal: { all(apple, feature = "metal") },
        vulkan: { all(not(wasm_platform), feature = "vulkan") },
        any_backend: { any(dx12, metal, vulkan, gles) },
        // ⚠️ Keep in sync with target.cfg() definition in Cargo.toml and cfg_alias in `wgpu` crate ⚠️
        static_dxc: { all(windows, feature = "static-dxc", not(target_arch = "aarch64"), target_env = "msvc") },
        supports_64bit_atomics: { target_has_atomic = "64" },
        supports_ptr_atomics: { target_has_atomic = "ptr" }
    }
}
