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
        gles: {
            any(
                all(not(target_vendor = "apple"), feature = "gles"), // Standard GLES on non-Apple platforms
                all(target_vendor = "apple", feature = "angle") // ANGLE on Apple platforms
            )
        },
        metal: { all(target_vendor = "apple", feature = "metal") },
        vulkan: {
            any(
                all(not(target_vendor = "apple"), feature = "vulkan"), // Standard Vulkan on non-Apple platforms
                all(target_vendor = "apple", feature = "vulkan-portability") // Vulkan Portability on Apple platforms
            )
        },
        // ⚠️ Keep in sync with target.cfg() definition in Cargo.toml and cfg_alias in `wgpu` crate ⚠️
        static_dxc: { all(target_os = "windows", feature = "static-dxc", not(target_arch = "aarch64")) }
    }
}
