# Naga

[![Matrix](https://img.shields.io/badge/Matrix-%23naga%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#naga:matrix.org)
[![Crates.io](https://img.shields.io/crates/v/naga.svg?label=naga)](https://crates.io/crates/naga)
[![Docs.rs](https://docs.rs/naga/badge.svg)](https://docs.rs/naga)
[![Build Status](https://github.com/gfx-rs/naga/workflows/pipeline/badge.svg)](https://github.com/gfx-rs/naga/actions)

This is an experimental shader translation library for the needs of gfx-rs project and WebGPU.

## Supported end-points

Everything is still work-in-progress, but some end-points are usable:

Front-end       |       Status       | Feature | Notes |
--------------- | ------------------ | ------- | ----- |
SPIR-V (binary) | :white_check_mark: | spv-in  |       |
WGSL            | :white_check_mark: | wgsl-in |       |
GLSL            | :ok:               | glsl-in | Vulkan flavor is expected |
Rust            |                    |         |       |

Back-end        |       Status       | Feature  | Notes |
--------------- | ------------------ | -------- | ----- |
SPIR-V          | :white_check_mark: | spv-out  |       |
WGSL            |                    |          |       |
Metal           | :white_check_mark: | msl-out  |       |
HLSL            | :construction:     | hlsl-out |       |
GLSL            | :ok:               | glsl-out |       |
AIR             |                    |          |       |
DXIL/DXIR       |                    |          |       |
DXBC            |                    |          |       |
DOT (GraphViz)  | :ok:               | dot-out  | Not a shading language |

:white_check_mark: = Primary support — :ok: = Secondary support — :construction: = Unsupported, but support in progress

## Conversion tool

Naga includes a default binary target "convert", which allows to test the conversion of different code paths.
```bash
cargo run --features spv-in -- my_shader.spv # dump the IR module to debug output
cargo run --features spv-in,msl-out -- my_shader.spv my_shader.metal --flow-dir flow-dir # convert the SPV to Metal, also dump the SPIR-V flow graph to `flow-dir`
cargo run --features wgsl-in,glsl-out -- my_shader.wgsl my_shader.vert --profile es310 # convert the WGSL to GLSL vertex stage under ES 3.20 profile
```

## Development workflow

The main instrument aiding the development is the good old `cargo test --all-features`,
which will run the snapshot tests as well as the unit tests.
Any changes in the snapshots would then have to be reviewed with `cargo insta review`
before being accepted into the code.

If working on a particular front-end or back-end, it may be convenient to
enable the relevant features in `Cargo.toml`, e.g.
```toml
default = ["spv-out"] #TEMP!
```
This allows IDE basic checks to report errors there, unless your IDE is sufficiently configurable already.

Finally, when changes to the snapshots are made, we should verify that the produced shaders
are indeed valid for the target platforms they are compiled for. We automate this with `Makefile`:
```bash
make validate-spv # for Vulkan shaders, requires SPIRV-Tools installed
make validate-msl # for Metal shaders, requires XCode command-line tools installed
make validate-glsl # for OpenGL shaders, requires GLSLang installed
```
