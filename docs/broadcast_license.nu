# Quick maintenance script which broadcasts the license files to all crates
# that are released on crates.io

# Change to the root of the repository
cd ($env.FILE_PWD | path dirname)

let crates = [
    "wgpu",
    "wgpu-core",
    "wgpu-core/platform-deps/apple",
    "wgpu-core/platform-deps/emscripten",
    "wgpu-core/platform-deps/wasm",
    "wgpu-core/platform-deps/windows-linux-android",
    "wgpu-hal",
    "wgpu-info",
    "wgpu-naga-bridge",
    "wgpu-types",
    "naga",
    "naga-cli",
]

for crate in $crates {
    cp LICENSE.APACHE LICENSE.MIT $"./($crate)/"
}
