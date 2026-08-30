# `naga-cli`

The command-line front end for [`naga`](../naga), the shader translator and
validator used by [`wgpu`](https://github.com/gfx-rs/wgpu). It reads WGSL,
SPIR-V, and GLSL, and writes WGSL, SPIR-V, GLSL, MSL, HLSL, naga IR, and
Graphviz `dot` graphs.

## Installation

```bash
# release version
cargo install naga-cli

# development version
cargo install naga-cli --git https://github.com/gfx-rs/wgpu.git
```

The installed binary is called `naga`, not `naga-cli`.

## Usage

```bash
naga my_shader.wgsl # validate only
naga my_shader.spv my_shader.txt # dump the IR module into a file
naga my_shader.spv my_shader.metal --flow-dir flow-dir # convert the SPV to Metal, also dump the SPIR-V flow graph to `flow-dir`
naga my_shader.wgsl my_shader.vert --profile es310 # convert the WGSL to GLSL vertex stage under ES 3.20 profile
```

The output language is chosen from the output file's extension. Run
`naga --help` for the full option list.
