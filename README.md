# WebGPU
[![Build Status](https://travis-ci.org/gfx-rs/wgpu.svg)](https://travis-ci.org/gfx-rs/wgpu)
[![Gitter](https://badges.gitter.im/gfx-rs/webgpu.svg)](https://gitter.im/gfx-rs/webgpu)

This is an experimental [WebGPU](https://www.w3.org/community/gpu/) implementation as a native static library. It's written in Rust and is based on [gfx-hal](https://github.com/gfx-rs/gfx) and [satellite](https://github.com/gfx-rs/gfx-memory) libraries. The corresponding WebIDL specification can be found at [gpuweb project](https://github.com/gpuweb/gpuweb/blob/master/design/sketch.webidl).

The implementation consists of the following parts:
  - `wgpu-native` - the native implementation of WebGPU as a C API library
  - `wgpu-bindings` - automatic generator of actual C headers
  - `wgpu-remote` - remoting layer to work with WebGPU across the process boundary
  - `wgpu-rs` - idiomatic Rust wrapper of the native library
