fn main() {
    cfg_aliases::cfg_aliases! {
        send_sync: { any(
            not(target_arch = "wasm32"),
            all(feature = "fragile-send-sync-non-atomic-wasm", not(target_feature = "atomics"))
        ) },
        webgl: { all(target_arch = "wasm32", not(target_os = "emscripten"), gles) },
        dx12: { all(target_os = "windows", feature = "dx12") },
        gles: { all(feature = "gles") },
        metal: { all(any(target_os = "ios", target_os = "macos"), feature = "metal") },
        vulkan: { all(not(target_arch = "wasm32"), feature = "vulkan") }
    }
}
