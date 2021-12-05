fn main() {
    // Setup cfg aliases
    cfg_aliases::cfg_aliases! {
        // Vendors/systems
        wasm: { target_arch = "wasm32" },
        apple: { any(target_os = "ios", target_os = "macos") },
        unix_wo_apple: {all(unix, not(apple))},

        // Backends
        vulkan: { all(not(wasm), any(windows, unix_wo_apple)) },
        metal: { all(not(wasm), apple) },
        dx12: { all(not(wasm), windows) },
        dx11: { all(false, not(wasm), windows) },
        gl: {
            any(
                unix_wo_apple,
                feature = "angle",
                wasm
            )
        },
    }
}
