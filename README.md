## wgpu-rs
[![Build Status](https://travis-ci.org/gfx-rs/wgpu-rs.svg)](https://travis-ci.org/gfx-rs/wgpu-rs)
[![Crates.io](https://img.shields.io/crates/v/wgpu.svg)](https://crates.io/crates/wgpu)
[![Gitter](https://badges.gitter.im/gfx-rs/webgpu.svg)](https://gitter.im/gfx-rs/webgpu)

This is an idiomatic Rust wrapper over [wgpu-native](https://github.com/gfx-rs/wgpu). It's designed to be suitable for general purpose graphics and computation needs of Rust community. It currently only works for the native platform, in the future aims to support WASM/Emscripten platforms as well.

## Gallery

![Cube](etc/example-cube.png) ![Shadow](etc/example-shadow.png) ![MipMap](etc/example-mipmap.png)
![vange-rs](etc/vange-rs.png) ![Brawl](etc/brawl-attack.gif) ![GLX map](etc/glx-map.png)

## Usage

### Running an example
All examples are located under the [examples](examples) directory. We are using the default syntax for running examples, as found in the [Cargo](https://doc.rust-lang.org/cargo/reference/manifest.html#examples) documentation.

```bash
cargo run --example cube
```

#### Hello Compute

`hello-*` examples show barebones setup without any helper code.

For "hello-compute", pass 4 numbers separated by spaces as arguments:
```bash
cargo run --example hello-compute 1 2 3 4
```

More examples can be found under the [examples](examples) directory.

## Friends

Shout out to the following projects that work best with wgpu-rs:
  - [wgpu_glyph](https://github.com/hecrj/wgpu_glyph) - for your text-y rendering needs
  - [coffee](https://github.com/hecrj/coffee) - a whole 2D engine
  - [rgx](https://github.com/cloudhead/rgx) - a 2D graphics library
  - [imgui-wgpu](https://github.com/unconed/imgui-wgpu-rs) - Dear ImGui interfacing

## Development

If you need to test local fixes to gfx-rs or other dependencies, the simplest way is to add a Cargo patch. For example, when working on DX12 backend on Windows, you can check out the "hal-0.2" branch of gfx-rs repo and add this to the end of "Cargo.toml":
```toml
[patch.crates-io]
gfx-backend-dx12 = { path = "../gfx/src/backend/dx12" }
gfx-hal = { path = "../gfx/src/hal" }
```
If a version needs to be changed, you need to to do `cargo update -p gfx-backend-dx12`.
