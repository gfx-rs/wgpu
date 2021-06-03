<img align="right" width="25%" src="logo.png">

# wgpu-rs

[![Build Status](https://github.com/gfx-rs/wgpu-rs/workflows/CI/badge.svg?branch=master)](https://github.com/gfx-rs/wgpu-rs/actions)
[![Crates.io](https://img.shields.io/crates/v/wgpu.svg)](https://crates.io/crates/wgpu)
[![Docs.rs](https://docs.rs/wgpu/badge.svg)](https://docs.rs/wgpu)

[![Matrix](https://img.shields.io/badge/Dev_Matrix-%23wgpu%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#wgpu:matrix.org)
[![Matrix](https://img.shields.io/badge/User_Matrix-%23wgpu--users%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#wgpu-users:matrix.org)

wgpu-rs is an idiomatic Rust wrapper over [wgpu-core](https://github.com/gfx-rs/wgpu). It's designed to be suitable for general purpose graphics and computation needs of Rust community.

wgpu-rs can target both the natively supported backends and WASM directly.

See our [gallery](https://wgpu.rs/#showcase) and the [wiki page](https://github.com/gfx-rs/wgpu-rs/wiki/Applications-and-Libraries) for the list of libraries and applications using `wgpu-rs`.

## Usage

### How to Run Examples

All examples are located under the [examples](examples) directory.

These examples use the default syntax for running examples, as found in the [Cargo](https://doc.rust-lang.org/cargo/reference/manifest.html#examples) documentation. For example, to run the `cube` example:

```bash
cargo run --example cube
```

The `hello*` examples show bare-bones setup without any helper code. For `hello-compute`, pass 4 numbers separated by spaces as arguments:

```bash
cargo run --example hello-compute 1 2 3 4
```

The following environment variables can be used to configure how the framework examples run:

- `WGPU_BACKEND`

  Options: `vulkan`, `metal`, `dx11`, `dx12`, `gl`, `webgpu`

  If unset a default backend is chosen based on what is supported
  by your system.

- `WGPU_POWER_PREF`

  Options: `low`, `high`

  If unset a low power adapter is preferred.

#### Run Examples on the Web (`wasm32-unknown-unknown`)

See [wiki article](https://github.com/gfx-rs/wgpu-rs/wiki/Running-on-the-Web-with-WebGPU-and-WebGL).

## Logging

`wgpu-core` uses `tracing` for logging and `wgpu-rs` uses `log` for logging.

### Simple Setup

If you just want log messages to show up and to use the chrome tracing infrastructure,
take a dependency on the `wgpu-subscriber` crate then call `initialize_default_subscriber`. It will
set up logging to stdout/stderr based on the `RUST_LOG` environment variable.

### Manual Conversion

`tracing` also has tools available to convert all `tracing` events into `log` events and vise versa.

#### `log` events -> `tracing` events

The `tracing_log` crate has a `log` logger to translate all events into `tracing` events. Call:

```rust
tracing_log::LogTracer::init().unwrap()
```

#### `tracing` events -> `log` events

The `tracing` crate has a `log` feature which will automatically use `log` if no subscriber is added:

```toml
tracing = { version = "0.1", features = ["log"] }
```

If you want events to be handled both by `tracing` and `log`, enable the `log-always` feature of `tracing`:

```toml
tracing = { version = "0.1", features = ["log-always"] }
```

## Development

If you need to test local fixes to gfx or other dependencies, the simplest way is to add a Cargo patch. For example, when working on DX12 backend on Windows, you can check out the latest release branch in the [gfx-hal repository](https://github.com/gfx-rs/gfx) (e.g. currently `hal-0.8`) and add this patch to the end of `Cargo.toml`:

```toml
[patch."https://github.com/gfx-rs/gfx"]
gfx-backend-dx12 = { path = "../gfx/src/backend/dx12" }
gfx-hal = { path = "../gfx/src/hal" }
```

If a version needs to be changed, you need to do `cargo update -p gfx-backend-dx12`.
