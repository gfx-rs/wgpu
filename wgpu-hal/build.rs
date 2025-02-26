fn main() {
    cfg_aliases::cfg_aliases! {
        native: { not(target_arch = "wasm32") },
        send_sync: { any(
            not(target_arch = "wasm32"),
            all(feature = "fragile-send-sync-non-atomic-wasm", not(target_feature = "atomics"))
        ) },
        webgl: { all(target_arch = "wasm32", not(target_os = "emscripten"), gles) },
        Emscripten: { all(target_os = "emscripten", gles) },
        dx12: { all(target_os = "windows", feature = "dx12") },
        gles: { all(feature = "gles") },
        metal: { all(target_vendor = "apple", feature = "metal") },
        vulkan: { all(not(target_arch = "wasm32"), feature = "vulkan") },
        // ⚠️ Keep in sync with target.cfg() definition in Cargo.toml and cfg_alias in `wgpu` crate ⚠️
        static_dxc: { all(target_os = "windows", feature = "static-dxc", not(target_arch = "aarch64")) },
        supports_64bit_atomics: { target_has_atomic = "64" }
    }
}
