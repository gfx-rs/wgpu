# WebGPU
[![Build Status](https://travis-ci.org/gfx-rs/wgpu.svg)](https://travis-ci.org/gfx-rs/wgpu)
[![Crates.io](https://img.shields.io/crates/v/wgpu.svg)](https://crates.io/crates/wgpu)
[![Gitter](https://badges.gitter.im/gfx-rs/webgpu.svg)](https://gitter.im/gfx-rs/webgpu)

This is an experimental [WebGPU](https://www.w3.org/community/gpu/) implementation as a native static library. It's written in Rust and is based on [gfx-hal](https://github.com/gfx-rs/gfx) and [satellite](https://github.com/gfx-rs/gfx-memory) libraries. The corresponding WebIDL specification can be found at [gpuweb project](https://github.com/gpuweb/gpuweb/blob/master/design/sketch.webidl).

The implementation consists of the following parts:
  - `wgpu-native` - the native implementation of WebGPU as a C API library
  - `wgpu-bindings` - automatic generator of actual C headers
  - `wgpu-remote` - remoting layer to work with WebGPU across the process boundary
  - `wgpu-rs` - idiomatic Rust wrapper of the native library

## Example

To run an example, simply `cd` to the `examples` or `gfx-examples` directory, then use `cargo run` with `--features {backend}` to specify the backend (where `{backend}` is either `vulkan`, `dx12`, `dx11` or `metal`). For example:

```bash
# Clone the wgpu repository
git clone https://github.com/gfx-rs/wgpu
# Change directory to `examples`
cd wgpu/examples
# Vulkan (Linux/Windows)
cargo run --bin hello_triangle --features vulkan
# Metal (macOS/iOS)
cargo run --bin hello_triangle --features metal
# DirectX12 (Windows)
cargo run --bin hello_triangle --features dx12

cd ../gfx-examples
# Vulkan (Linux/Windows)
cargo run --bin cube --features vulkan
# Metal (macOS/iOS)
cargo run --bin cube --features metal
# DirectX12 (Windows)
cargo run --bin cube --features dx12
```

These examples assume that necessary dependencies for the graphics backend are already installed. For more information about installation and usage, refer to the [Getting Started](https://github.com/gfx-rs/gfx/blob/master/info/getting_started.md) guide.
