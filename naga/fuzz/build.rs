fn main() {
    cfg_aliases::cfg_aliases! {
        enable_fuzzing: { not(any(target_arch = "wasm32", target_os = "ios", all(windows, target_arch = "aarch64"))) },
    }
}
