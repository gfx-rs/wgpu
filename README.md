<img align="right" width="25%" src="logo.png">

This is an active GitHub mirror of the WebGPU implementation in Rust, which now lives in "gfx/wgpu" of [Mozilla-central](https://hg.mozilla.org/mozilla-central/file/tip/gfx/wgpu). Issues and pull requests are accepted, but some bidirectional synchronization may be involved.

# WebGPU

[![Matrix](https://img.shields.io/badge/Matrix-%23wgpu%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#wgpu:matrix.org)
[![Build Status](https://travis-ci.org/gfx-rs/wgpu.svg?branch=master)](https://travis-ci.org/gfx-rs/wgpu)
[![Crates.io](https://img.shields.io/crates/v/wgpu-core.svg?label=wgpu-core)](https://crates.io/crates/wgpu-core)
[![Crates.io](https://img.shields.io/crates/v/wgpu-native.svg?label=wgpu-native)](https://crates.io/crates/wgpu-native)

This is the core logic of an experimental [WebGPU](https://www.w3.org/community/gpu/) implementation. It's written in Rust and is based on [gfx-hal](https://github.com/gfx-rs/gfx) with help of [gfx-extras](https://github.com/gfx-rs/gfx-extras). See the upstream [WebGPU specification](https://gpuweb.github.io/gpuweb/) (work in progress).

The implementation consists of the following parts:

  - `wgpu-core` - internal Rust API for WebGPU implementations to use
  - `wgpu-types` - Rust types shared between `wgpu-core`, `wgpu-native`, and `wgpu-rs`

This repository is not meant for direct use by applications.
If you are looking for the user-facing Rust API, you need [wgpu-rs](https://github.com/gfx-rs/wgpu-rs).
If you are looking for the native implementation or bindings to the API in other languages, you need [wgpu-native](https://github.com/gfx-rs/wgpu-native).

## Supported Platforms

   API   |       Windows      |       Linux        |    macOS & iOS     |
  -----  | ------------------ | ------------------ | ------------------ |
  DX11   | :white_check_mark: |                    |                    |
  DX12   | :heavy_check_mark: |                    |                    |
  Vulkan | :heavy_check_mark: | :heavy_check_mark: |                    |
  Metal  |                    |                    | :heavy_check_mark: |
  OpenGL | :construction:     | :construction:     | :construction:     |
  
:heavy_check_mark: = Primary support — :white_check_mark: = Secondary support — :construction: = Unsupported, but support in progress
