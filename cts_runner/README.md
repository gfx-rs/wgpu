# cts_runner

This crate contains infrastructure for running the WebGPU conformance tests on
Deno's `wgpu`-based implementation of WebGPU.

Instructions for running the tests via the CTS `xtask` are in
[docs/testing.md](https://github.com/gfx-rs/wgpu/blob/trunk/docs/testing.md#webgpu-cts).
The file [revision.txt](./revision.txt) specifies the version of the CTS that
will be used by default.

`cts_runner` is somewhat misnamed at this point, in that it is useful for
things other than just running the CTS:

- The [tests](./tests) directory contains a few directed tests for
  Deno's bindings to `wgpu`.
- Standalone JavaScript snippets that use WebGPU can be run
  with a command like: `cargo run -p cts_runner -- test.js`.
