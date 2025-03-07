fn main() {
    cfg_aliases::cfg_aliases! {
        fuzzable_platform: { not(any(target_arch = "wasm32", target_os = "ios", all(windows, target_arch = "aarch64"))) },
    }
    // This cfg provided by cargo-fuzz
    println!("cargo::rustc-check-cfg=cfg(fuzzing)");
}
