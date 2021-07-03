<img align="right" width="25%" src="logo.png">

# wgpu

[![Matrix](https://img.shields.io/badge/Dev_Matrix-%23wgpu%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#wgpu:matrix.org) [![Matrix](https://img.shields.io/badge/User_Matrix-%23wgpu--users%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#wgpu-users:matrix.org)
[![Build Status](https://github.com/gfx-rs/wgpu/workflows/CI/badge.svg)](https://github.com/gfx-rs/wgpu/actions)
[![codecov.io](https://codecov.io/gh/gfx-rs/wgpu/branch/master/graph/badge.svg?token=84qJTesmeS)](https://codecov.io/gh/gfx-rs/wgpu)

This is an implementation of [WebGPU](https://www.w3.org/community/gpu/) API in Rust, targeting both native and the Web.
See the upstream [WebGPU specification](https://gpuweb.github.io/gpuweb/) (work in progress).

The repository hosts the following parts:

  - [![Crates.io](https://img.shields.io/crates/v/wgpu.svg?label=wgpu)](https://crates.io/crates/wgpu) [![docs.rs](https://docs.rs/wgpu/badge.svg)](https://docs.rs/wgpu/) - public Rust API for users
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-core.svg?label=wgpu-core)](https://crates.io/crates/wgpu-core) [![docs.rs](https://docs.rs/wgpu-core/badge.svg)](https://docs.rs/wgpu-core/) - internal Rust API for WebGPU implementations to use
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-hal.svg?label=wgpu-hal)](https://crates.io/crates/wgpu-hal) [![docs.rs](https://docs.rs/wgpu-hal/badge.svg)](https://docs.rs/wgpu-hal/) - internal unsafe GPU abstraction API
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-info.svg?label=wgpu-info)](https://crates.io/crates/wgpu-info) - program that prints out information about all the adapters on the system or invokes a command for every adapter.
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-types.svg?label=wgpu-types)](https://crates.io/crates/wgpu-types) [![docs.rs](https://docs.rs/wgpu-types/badge.svg)](https://docs.rs/wgpu-types/) - Rust types shared between `wgpu-core` and `wgpu-rs`
  - `player` - standalone application for replaying the API traces, uses `winit`

Rust examples can be found at `wgpu/examples`. `wgpu` is a default member, so you can run the examples directly from the root, e.g. `cargo run --example boids`.

If you are looking for the native implementation or bindings to the API in other languages, you need [wgpu-native](https://github.com/gfx-rs/wgpu-native).

## Supported Platforms

   API   |    Windows 7/10    |  Linux & Android   |    macOS & iOS     |
  -----  | ------------------ | ------------------ | ------------------ |
  DX11   | :construction:     |                    |                    |
  DX12   | :construction:     |                    |                    |
  Vulkan | :white_check_mark: | :white_check_mark: |                    |
  Metal  |                    |                    | :white_check_mark: |
  GLes3  |                    | :ok:               |                    |

:white_check_mark: = Primary support — :ok: = Secondary support — :construction: = Unsupported, but support in progress

## Testing Infrastructure

wgpu features a set of unit, integration, and example based tests. All framework based examples are automatically reftested against the screenshot in the example directory. The `wgpu-info` example contains the logic which can automatically run the tests multiple times for all the adapters present on the system. These tests are also run on CI on windows and linux over Vulkan/DX12/DX11/GL on software adapters.

To run the test suite, run the following command:

```
cargo run --bin wgpu-info -- cargo test --no-fail-fast
```

To run any individual test on a specific adapter, populate the following environment variables:
- `WGPU_ADAPTER_NAME` with a substring of the name of the adapter you want to use (ex. "1080" will match "NVIDIA GeForce 1080ti").
- `WGPU_BACKEND` with the name of the backend you want to use (`vulkan`, `metal`, `dx12`, `dx11`, or `gl`).

Then to run an example's reftests, run:

```
cargo test --example <example-name> --no-fail-fast
```

Or run a part of the integration test suite:

```
cargo test -p wgpu -- <name-of-test>
```

If you are a user and want a way to help contribute to wgpu, we always need more help writing test cases. 
