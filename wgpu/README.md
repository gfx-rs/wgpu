<img align="right" width="25%" src="https://raw.githubusercontent.com/gfx-rs/wgpu/master/logo.png">

wgpu-rs is an idiomatic Rust wrapper over [wgpu-core](https://github.com/gfx-rs/wgpu). It's designed to be suitable for general purpose graphics and computation needs of Rust community.

wgpu-rs can target both the natively supported backends and WASM directly.

See our [gallery](https://wgpu.rs/#showcase) and the [wiki page](https://github.com/gfx-rs/wgpu/wiki/Users) for the list of libraries and applications using `wgpu-rs`.

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

- `WGPU_ADAPTER_NAME`

  Select a specific adapter by specifying a substring of the adapter name.

#### Run Examples on the Web (`wasm32-unknown-unknown`)

See [wiki article](https://github.com/gfx-rs/wgpu/wiki/Running-on-the-Web-with-WebGPU-and-WebGL).

## Shaders

[WGSL](https://gpuweb.github.io/gpuweb/wgsl/) is the main shading language of WebGPU.

Users can run the [naga](https://github.com/gfx-rs/naga) binary in the following way to convert their SPIR-V shaders to WGSL:
```bash
cargo run -- <input.spv> <output.wgsl>
```

In addition, SPIR-V can be used by enabling the `spirv` feature and GLSL can be enabled by enabling the `glsl` feature at the cost of slightly increased build times.
