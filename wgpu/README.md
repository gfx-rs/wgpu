## wgpu-rs
[![Build Status](https://travis-ci.org/gfx-rs/wgpu-rs.svg)](https://travis-ci.org/gfx-rs/wgpu-rs)
[![Crates.io](https://img.shields.io/crates/v/wgpu.svg)](https://crates.io/crates/wgpu)
[![Gitter](https://badges.gitter.im/gfx-rs/webgpu.svg)](https://gitter.im/gfx-rs/webgpu)

This is an idiomatic Rust wrapper over [wgpu-native](https://github.com/gfx-rs/wgpu). It's designed to be suitable for general purpose graphics and computation needs of Rust community. It currently only works for the native platform, in the future aims to support WASM/Emscripten platforms as well.

## Gallery

![Cube](etc/example-cube.png) ![Shadow](etc/example-shadow.png) ![MipMap](etc/example-mipmap.png)
![vange-rs](etc/vange-rs.png) ![Brawl](etc/brawl-attack.gif) ![GLX map](etc/glx-map.png)

## Usage

The library requires one of the following features enabled in order to run any of the examples:
  - "vulkan"
  - "metal"
  - "dx12"
  - "dx11"

These examples assume that necessary dependencies for the graphics backend are already installed. For more information about installation and usage, refer to the [Getting Started](https://github.com/gfx-rs/gfx/blob/master/info/getting_started.md) gfx-rs guide.

## Friends

Shout out to the following projects that work best with wgpu-rs:
  - [wgpu_glyph](https://github.com/hecrj/wgpu_glyph) - for your text-y rendering needs
  - [coffee](https://github.com/hecrj/coffee) - a whole 2D engine
  - [imgui-wgpu](https://github.com/unconed/imgui-wgpu-rs) - Dear ImGui interfacing
