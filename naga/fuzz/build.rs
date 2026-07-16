#![allow(
    semicolon_in_expressions_from_macros,
    reason = "work around <https://github.com/katharostech/cfg_aliases/issues/16>"
)]

fn main() {
    cfg_aliases::cfg_aliases! {
        fuzzable_platform: { not(any(target_arch = "wasm32", target_os = "ios", target_os = "openbsd", all(windows, target_arch = "aarch64"))) },
    }
    // This cfg provided by cargo-fuzz
    println!("cargo::rustc-check-cfg=cfg(fuzzing)");
}
