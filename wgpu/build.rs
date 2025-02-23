fn main() {
    cfg_aliases::cfg_aliases! {
        native: { not(target_arch = "wasm32") },
        Emscripten: { all(target_arch = "wasm32", target_os = "emscripten") },

        send_sync: { any(
            native,
            all(feature = "fragile-send-sync-non-atomic-wasm", not(target_feature = "atomics"))
        ) },

        // Backends - keep this in sync with `wgpu-core/Cargo.toml` & docs in `wgpu/Cargo.toml`
        webgpu: { all(not(native), not(Emscripten), feature = "webgpu") },
        webgl: { all(not(native), not(Emscripten), feature = "webgl") },
        dx12: { all(target_os = "windows", feature = "dx12") },
        metal: { all(target_vendor = "apple", feature = "metal") },
        vulkan: { any(
            all(any(windows, target_os = "linux", target_os = "android"), feature = "vulkan"),
            all(target_vendor = "apple", feature = "vulkan-portability")
        ) },
        gles: { any(
            all(any(windows, target_os = "linux", target_os = "android", Emscripten), feature = "gles"),
            all(target_vendor = "apple", feature = "angle")
        ) },
        noop: { feature = "noop" },

        wgpu_core: { any(
                // On native, wgpu_core is currently always enabled, even if there's no backend enabled at all.
                native,
                // `wgpu_core` is implied if any backend other than WebGPU is enabled.
                // (this is redundant except for `gles` and `noop`)
                webgl, dx12, metal, vulkan, gles, noop
            )
        },

        // This alias is _only_ if _we_ need naga in the wrapper. wgpu-core provides
        // its own re-export of naga, which can be used in other situations
        naga: { any(feature = "naga-ir", feature = "spirv", feature = "glsl") },
        // ⚠️ Keep in sync with target.cfg() definition in wgpu-hal/Cargo.toml and cfg_alias in `wgpu-hal` crate ⚠️
        static_dxc: { all(target_os = "windows", feature = "static-dxc", not(target_arch = "aarch64")) },
    }
}
