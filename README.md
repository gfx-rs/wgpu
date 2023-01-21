<img align="right" width="25%" src="logo.png">

# wgpu

[![Matrix Space](https://img.shields.io/static/v1?label=Space&message=%23Wgpu&color=blue&logo=matrix)](https://matrix.to/#/#Wgpu:matrix.org)
[![Dev Matrix  ](https://img.shields.io/static/v1?label=devs&message=%23wgpu&color=blueviolet&logo=matrix)](https://matrix.to/#/#wgpu:matrix.org)
[![User Matrix ](https://img.shields.io/static/v1?label=users&message=%23wgpu-users&color=blueviolet&logo=matrix)](https://matrix.to/#/#wgpu-users:matrix.org)
[![Build Status](https://github.com/gfx-rs/wgpu/workflows/CI/badge.svg)](https://github.com/gfx-rs/wgpu/actions)
[![codecov.io](https://codecov.io/gh/gfx-rs/wgpu/branch/master/graph/badge.svg?token=84qJTesmeS)](https://codecov.io/gh/gfx-rs/wgpu)

`wgpu` is a cross-platform, safe, pure-rust graphics api. It runs natively on Vulkan, Metal, D3D12, D3D11, and OpenGLES; and on top of WebGPU on wasm.

The api is based on the [WebGPU standard](https://gpuweb.github.io/gpuweb/). It serves as the core of the WebGPU integration in Firefox, Servo, and Deno.

## Repo Overview

The repository hosts the following libraries:

  - [![Crates.io](https://img.shields.io/crates/v/wgpu.svg?label=wgpu)](https://crates.io/crates/wgpu) [![docs.rs](https://docs.rs/wgpu/badge.svg)](https://docs.rs/wgpu/) - User facing Rust API.
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-core.svg?label=wgpu-core)](https://crates.io/crates/wgpu-core) [![docs.rs](https://docs.rs/wgpu-core/badge.svg)](https://docs.rs/wgpu-core/) - Internal WebGPU implementation.
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-hal.svg?label=wgpu-hal)](https://crates.io/crates/wgpu-hal) [![docs.rs](https://docs.rs/wgpu-hal/badge.svg)](https://docs.rs/wgpu-hal/) - Internal unsafe GPU API abstraction layer.
  - [![Crates.io](https://img.shields.io/crates/v/wgpu-types.svg?label=wgpu-types)](https://crates.io/crates/wgpu-types) [![docs.rs](https://docs.rs/wgpu-types/badge.svg)](https://docs.rs/wgpu-types/) - Rust types shared between all crates.
  - [![Crates.io](https://img.shields.io/crates/v/deno_webgpu.svg?label=deno_webgpu)](https://crates.io/crates/deno_webgpu) - WebGPU implementation for the Deno JavaScript/TypeScript runtime

The following binaries:
  - `cts_runner` - WebGPU Conformance Test Suite runner using `deno_webgpu`.
  - `player` - standalone application for replaying the API traces.
  - `wgpu-info` - program that prints out information about all the adapters on the system or invokes a command for every adapter.

For an overview of all the components in the gfx-rs ecosystem, see [the big picture](./etc/big-picture.png).

### MSRV policy

Minimum Supported Rust Version is **1.64**.
It is enforced on CI (in "/.github/workflows/ci.yml") with `RUST_VERSION` variable.
This version can only be upgraded in breaking releases.

The `wgpu-core`, `wgpu-hal`, and `wgpu-types` crates should never
require an MSRV ahead of Firefox's MSRV for nightly builds, as
determined by the value of `MINIMUM_RUST_VERSION` in
[`python/mozboot/mozboot/util.py`][util]. However, Firefox uses `cargo
vendor` to extract only those crates it actually uses, so the
workspace's other crates can have more recent MSRVs.

*Note for Rust 1.64*: The workspace itself can even use a newer MSRV
than Firefox, as long as the vendoring step's `Cargo.toml` rewriting
removes any features Firefox's MSRV couldn't handle. For example,
`wgpu` can use manifest key inheritance, added in Rust 1.64, even
before Firefox reaches that MSRV, because `cargo vendor` copies
inherited values directly into the individual crates' `Cargo.toml`
files, producing 1.63-compatible files.

[util]: https://searchfox.org/mozilla-central/source/python/mozboot/mozboot/util.py

## Getting Started

### Rust

Rust examples can be found at `wgpu/examples`. You can run the examples with `cargo run --example name`. See the [list of examples](wgpu/examples). For detailed instructions, look at [Running the examples](https://github.com/gfx-rs/wgpu/wiki/Running-the-examples) on the wiki.

If you are looking for a wgpu tutorial, look at the following:
- https://sotrh.github.io/learn-wgpu/

### C/C++

To use wgpu in C/C++, you need [wgpu-native](https://github.com/gfx-rs/wgpu-native).

### Others

If you want to use wgpu in other languages, there are many bindings to wgpu-native from languages such as Python, D, Julia, Kotlin, and more. See [the list](https://github.com/gfx-rs/wgpu-native#bindings).

## Community


We have the Matrix space [![Matrix Space](https://img.shields.io/static/v1?label=Space&message=%23Wgpu&color=blue&logo=matrix)](https://matrix.to/#/#Wgpu:matrix.org) with a few different rooms that form the wgpu community:
- [![Dev Matrix](https://img.shields.io/static/v1?label=devs&message=%23wgpu&color=blueviolet&logo=matrix)](https://matrix.to/#/#wgpu:matrix.org) - discussion of the library's development.
- [![User Matrix](https://img.shields.io/static/v1?label=users&message=%23wgpu-users&color=blueviolet&logo=matrix)](https://matrix.to/#/#wgpu-users:matrix.org) - discussion of using the library and the surrounding ecosystem.
- [![Random Matrix](https://img.shields.io/static/v1?label=random&message=%23wgpu-random&color=blueviolet&logo=matrix)](https://matrix.to/#/#wgpu-random:matrix.org) - discussion of everything else.

## Wiki

We have a [wiki](https://github.com/gfx-rs/wgpu/wiki) that serves as a knowledge base.

## Supported Platforms

   API   |    Windows                     |  Linux & Android          |    macOS & iOS      |
  -----  | ------------------------------ | ------------------------- | ------------------- |
  Vulkan | :white_check_mark:             | :white_check_mark:        | :ok: (vulkan-portability) |
  Metal  |                                |                           | :white_check_mark:  |
  DX12   | :white_check_mark: (W10+ only) |                           |                     |
  DX11   | :hammer_and_wrench:            |                           |                     |
  GLES3  |                                | :ok:                      |                     |
  Angle  | :ok:                           | :ok:                      | :ok: (macOS only) |

:white_check_mark: = First Class Support — :ok: = Best Effort Support — :hammer_and_wrench: = Unsupported, but support in progress

### Shader Support

wgpu supports shaders in [WGSL](https://gpuweb.github.io/gpuweb/wgsl/), SPIR-V, and GLSL.
Both [HLSL](https://github.com/Microsoft/DirectXShaderCompiler) and [GLSL](https://github.com/KhronosGroup/glslang)
have compilers to target SPIR-V. All of these shader languages can be used with any backend, we
will handle all of the conversion. Additionally, support for these shader inputs is not going away.

While WebGPU does not support any shading language other than WGSL, we will automatically convert your
non-WGSL shaders if you're running on WebGPU.

WGSL is always supported by default, but GLSL and SPIR-V need features enabled to compile in support.

Note that the WGSL specification is still under development,
so the [draft specification][wgsl spec] does not exactly describe what `wgpu` supports.
See [below](#tracking-the-webgpu-and-wgsl-draft-specifications) for details.

To enable SPIR-V shaders, enable the `spirv` feature of wgpu.
To enable GLSL shaders, enable the `glsl` feature of wgpu.

### Angle

[Angle](http://angleproject.org) is a translation layer from GLES to other backends, developed by Google.
We support running our GLES3 backend over it in order to reach platforms with GLES2 or DX11 support, which aren't accessible otherwise.
In order to run with Angle, "angle" feature has to be enabled, and Angle libraries placed in a location visible to the application.
These binaries can be downloaded from [gfbuild-angle](https://github.com/DileSoft/gfbuild-angle) artifacts, [manual compilation](https://github.com/google/angle/blob/main/doc/DevSetup.md) may be required on Macs with Apple silicon.

On Windows, you generally need to copy them into the working directory, in the same directory as the executable, or somewhere in your path.
On Linux, you can point to them using `LD_LIBRARY_PATH` environment.

## Environment Variables

All testing and example infrastructure shares the same set of environment variables that determine which Backend/GPU it will run on.

- `WGPU_ADAPTER_NAME` with a substring of the name of the adapter you want to use (ex. `1080` will match `NVIDIA GeForce 1080ti`).
- `WGPU_BACKEND` with a comma separated list of the backends you want to use (`vulkan`, `metal`, `dx12`, `dx11`, or `gl`).
- `WGPU_POWER_PREF` with the power preference to choose when a specific adapter name isn't specified (`high` or `low`)
- `WGPU_DX12_COMPILER` with the DX12 shader compiler you wish to use (`dxc` or `fxc`, note that `dxc` requires `dxil.dll` and `dxcompiler.dll` to be in the working directory otherwise it will fall back to `fxc`)

When running the CTS, use the variables `DENO_WEBGPU_ADAPTER_NAME`, `DENO_WEBGPU_BACKEND`, `DENO_WEBGPU_POWER_PREFERENCE`.

## Testing

We have multiple methods of testing, each of which tests different qualities about wgpu. We automatically run our tests on CI if possible. The current state of CI testing:

| Backend/Platform | Tests              | CTS                 | Notes                                 |
| ---------------- | -------------------|---------------------|-------------------------------------- |
| DX12/Windows 10  | :heavy_check_mark: | :heavy_check_mark:  | using WARP                            |
| DX11/Windows 10  | :construction:     | —                   | using WARP                            |
| Metal/MacOS      | —                  | —                   | metal requires GPU                    |
| Vulkan/Linux     | :heavy_check_mark: | :x:                 | using lavapipe, [cts hangs][cts-hang] |
| GLES/Linux       | :heavy_check_mark: | —                   | using llvmpipe                        |

[cts-hang]: https://github.com/gfx-rs/wgpu/issues/1974

### Core Test Infrastructure

We use a tool called [`cargo nextest`](https://github.com/nextest-rs/nextest) to run our tests.
To install it, run `cargo install cargo-nextest`.

To run the test suite on the default device:

```
cargo nextest run --no-fail-fast
```

`wgpu-info` can run the tests once for each adapter on your system.

```
cargo run --bin wgpu-info -- cargo nextest run --no-fail-fast
```

Then to run an example's image comparison tests, run:

```
cargo nextest run <example-test-name> --no-fail-fast
```

Or run a part of the integration test suite:

```
cargo nextest run -p wgpu -- <name-of-test>
```

If you are a user and want a way to help contribute to wgpu, we always need more help writing test cases.

### WebGPU Conformance Test Suite

WebGPU includes a Conformance Test Suite to validate that implementations are working correctly. We can run this CTS against wgpu.

To run the CTS, first you need to check it out:

```
git clone https://github.com/gpuweb/cts.git
cd cts
# works in bash and powershell
git checkout $(cat ../cts_runner/revision.txt)
```

To run a given set of tests:

```
# Must be inside the cts folder we just checked out, else this will fail
cargo run --manifest-path ../cts_runner/Cargo.toml -- ./tools/run_deno --verbose "<test string>"
```

To find the full list of tests, go to the [online cts viewer](https://gpuweb.github.io/cts/standalone/?runnow=0&worker=0&debug=0&q=webgpu:*).

The list of currently enabled CTS tests can be found [here](./cts_runner/test.lst).

## Tracking the WebGPU and WGSL draft specifications

The `wgpu` crate is meant to be an idiomatic Rust translation of the [WebGPU API][webgpu spec].
That specification, along with its shading language, [WGSL][wgsl spec],
are both still in the "Working Draft" phase,
and while the general outlines are stable,
details change frequently.
Until the specification is stabilized, the `wgpu` crate and the version of WGSL it implements
will likely differ from what is specified,
as the implementation catches up.

Exactly which WGSL features `wgpu` supports depends on how you are using it:

- When running as native code, `wgpu` uses the [Naga][naga] crate
  to translate WGSL code into the shading language of your platform's native GPU API.
  Naga has [a milestone][naga wgsl milestone]
  for catching up to the WGSL specification,
  but in general there is no up-to-date summary
  of the differences between Naga and the WGSL spec.

- When running in a web browser (by compilation to WebAssembly)
  without the `"webgl"` feature enabled,
  `wgpu` relies on the browser's own WebGPU implementation.
  WGSL shaders are simply passed through to the browser,
  so that determines which WGSL features you can use.

- When running in a web browser with `wgpu`'s `"webgl"` feature enabled,
  `wgpu` uses Naga to translate WGSL programs into GLSL.
  This uses the same version of Naga as if you were running `wgpu` as native code.

[webgpu spec]: https://www.w3.org/TR/webgpu/
[wgsl spec]: https://gpuweb.github.io/gpuweb/wgsl/
[naga]: https://github.com/gfx-rs/naga/
[naga wgsl milestone]: https://github.com/gfx-rs/naga/milestone/4

## Coordinate Systems

wgpu uses the coordinate systems of D3D and Metal:

Render | Texture
-------|--------
![render_coordinates](./etc/render_coordinates.png) | ![texture_coordinates](./etc/texture_coordinates.png)
