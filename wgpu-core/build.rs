fn main() {
    // Setup cfg aliases
    cfg_aliases::cfg_aliases! {
        // Vendors/systems
        apple: { any(target_os = "ios", target_os = "macos") },

        // Backends
        vulkan: { any(windows, all(unix, not(apple)), feature = "gfx-backend-vulkan") },
        metal: { apple },
        dx12: { windows },
        dx11: { windows },
        gl: { unix },
    }
}
