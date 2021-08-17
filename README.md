<img align="right" width="25%" src="logo.png">

# wgpu

[![Matrix](https://img.shields.io/badge/Dev_Matrix-%23wgpu%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#wgpu:matrix.org) [![Matrix](https://img.shields.io/badge/User_Matrix-%23wgpu--users%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#wgpu-users:matrix.org)
[![Build Status](https://github.com/gfx-rs/wgpu/workflows/CI/badge.svg)](https://github.com/gfx-rs/wgpu/actions)
[![codecov.io](https://codecov.io/gh/gfx-rs/wgpu/branch/master/graph/badge.svg?token=84qJTesmeS)](https://codecov.io/gh/gfx-rs/wgpu)

This is an implementation of [WebGPU](https://www.w3.org/community/gpu/) API in Rust, targeting both native and the Web.
It's written in Rust and is based on [gfx-hal](https://github.com/gfx-rs/gfx) with help of [gpu-alloc](https://github.com/zakarumych/gpu-alloc) and [gpu-descriptor](https://github.com/zakarumych/gpu-descriptor). See the upstream [WebGPU specification](https://gpuweb.github.io/gpuweb/) (work in progress).

The repository hosts the following parts:

  - [![Crates.io](https://img.shields.io/crates/v/wgpu.svg?label=wgpu)](https://crates.io/crates/wgpu) [![docs.rs](https://docs.rs/wgpu/badge.svg)](https://docs.rs/wgpu/) - public Rust API for users
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-core.svg?label=wgpu-core)](https://crates.io/crates/wgpu-core) [![docs.rs](https://docs.rs/wgpu-core/badge.svg)](https://docs.rs/wgpu-core/) - internal Rust API for WebGPU implementations to use
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-types.svg?label=wgpu-types)](https://crates.io/crates/wgpu-types) [![docs.rs](https://docs.rs/wgpu-types/badge.svg)](https://docs.rs/wgpu-types/) - Rust types shared between `wgpu-core` and `wgpu-rs`
  - `player` - standalone application for replaying the API traces, uses `winit`

Rust examples can be found at `wgpu/examples`. `wgpu` is a default member, so you can run the examples directly from the root, e.g. `cargo run --example boids`.

If you are looking for the native implementation or bindings to the API in other languages, you need [wgpu-native](https://github.com/gfx-rs/wgpu-native).

## Supported Platforms

   API   |    Windows 7/10    |  Linux & Android   |    macOS & iOS     |
  -----  | ------------------ | ------------------ | ------------------ |
  DX11   | :ok:               |                    |                    |
  DX12   | :white_check_mark: |                    |                    |
  Vulkan | :white_check_mark: | :white_check_mark: |                    |
  Metal  |                    |                    | :white_check_mark: |
  GL ES3 |                    | :construction:     |                    |

:white_check_mark: = Primary support — :ok: = Secondary support — :construction: = Unsupported, but support in progress
