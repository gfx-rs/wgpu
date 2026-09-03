fn main() {
    cfg_aliases::cfg_aliases! {
        native: { not(target_family = "wasm") },
        Emscripten: { all(target_family = "wasm", target_os = "emscripten") },
        web: { all(target_family = "wasm", not(Emscripten), feature = "web") },

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
            // The `vulkan` feature enables the Vulkan backend only on "native Vulkan" platforms, i.e. Windows/Linux/Android
            all(any(windows, target_os = "linux", target_os = "android", target_os = "freebsd"), feature = "vulkan"),
            // On Apple platforms, however, we require the `vulkan-portability` feature
            // to explicitly opt-in to Vulkan since it's meant to be used with MoltenVK.
            all(target_vendor = "apple", feature = "vulkan-portability")
        ) },
        drm: { all(
            feature = "drm",
            any(target_os = "linux", target_os = "freebsd", target_os = "netbsd", target_os = "openbsd")
        ) },
        gles: { any(
            // The `gles` feature enables the OpenGL/GLES backend only on "native OpenGL" platforms, i.e. Windows, Linux, Android, and Emscripten.
            // (Note that WebGL is also not included here!)
            all(any(windows, target_os = "linux", target_os = "android", target_os = "freebsd", Emscripten), feature = "gles"),
            // On Apple platforms, however, we require the `angle` feature to explicitly opt-in to OpenGL
            // since it's meant to be used with ANGLE.
            all(target_vendor = "apple", feature = "angle")
        ) },
        noop: { feature = "noop" },

        wgpu_core: {
            any(
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
        static_dxc: { all(target_os = "windows", feature = "static-dxc", not(target_arch = "aarch64"), target_env = "msvc") },
        custom: {any(feature = "custom")},
        std: { any(
            feature = "std",
            // TODO: Remove this when an alternative Mutex implementation is available for `no_std`.
            // send_sync requires an appropriate Mutex implementation, which is only currently
            // possible with `std` enabled.
            send_sync,
            // Unwinding panics necessitate access to `std` to determine if a thread is panicking
            panic = "unwind"
        ) },
        no_std: { not(std) }
    }

    // Expose a `file://` base URL pointing at the crate's
    // `src/documentation/images/` directory so that local documentation builds
    // resolve images from disk. Because these images live inside the crate, they
    // are packaged on publish and this path also resolves when a downstream user
    // builds the docs of `wgpu` as a dependency. `docsrs` builds ignore this and
    // use an HTTP URL instead; see the `doc_image!` macro in `src/macros/mod.rs`.
    // Resolving the path here lets us normalize to forward slashes and pick the
    // right number of leading slashes for the host.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let docs_dir = std::path::Path::new(&manifest_dir)
        .join("src")
        .join("documentation")
        .join("images");
    let docs_dir = docs_dir.to_string_lossy().replace('\\', "/");
    let docs_url = if docs_dir.starts_with('/') {
        format!("file://{docs_dir}")
    } else {
        format!("file:///{docs_dir}")
    };
    println!("cargo::rustc-env=WGPU_DOCS_URL_BASE={docs_url}");

    // Pin hosted documentation image URLs (docs.rs and the self-hosted trunk
    // docs) to the exact commit being built rather than the moving `trunk`
    // branch, so the images never drift from the prose. Only the `docsrs` arm of
    // the `doc_image!` macro reads this; local builds use the on-disk path above
    // and ignore it, so we deliberately don't add a `rerun-if-changed` on the
    // git HEAD — recomputing it on the build script's normal re-runs is enough.
    println!(
        "cargo::rustc-env=WGPU_DOCS_COMMIT={}",
        docs_commit(&manifest_dir)
    );
}

/// Best-effort lookup of the commit this build corresponds to, for pinning
/// documentation image URLs.
///
/// Tries `git` first (a local checkout, and the self-hosted trunk docs CI), then
/// the `.cargo_vcs_info.json` that `cargo publish` embeds in the packaged crate
/// (the docs.rs case, where there is no `.git`), and finally falls back to
/// `trunk` so the URLs still resolve.
fn docs_commit(manifest_dir: &str) -> String {
    commit_from_git(manifest_dir)
        .or_else(|| commit_from_vcs_info(manifest_dir))
        .unwrap_or_else(|| "trunk".to_owned())
}

fn commit_from_git(manifest_dir: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(manifest_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!sha.is_empty()).then_some(sha)
}

fn commit_from_vcs_info(manifest_dir: &str) -> Option<String> {
    let path = std::path::Path::new(manifest_dir).join(".cargo_vcs_info.json");
    let contents = std::fs::read_to_string(path).ok()?;
    // Minimal extraction of `"sha1": "<hex>"`, to avoid pulling a JSON parser
    // into the build script.
    let sha = contents
        .split_once("\"sha1\"")?
        .1
        .split_once('"')?
        .1
        .split_once('"')?
        .0;
    (!sha.is_empty()).then(|| sha.to_owned())
}
