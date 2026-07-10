# naga — the naga shader translator CLI

`naga` is the command-line interface for [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga),
the shader translation and validation library that is part of the wgpu project.
It converts between shader representations (WGSL, SPIR-V, GLSL, MSL, HLSL, DOT) and can
validate shaders without producing any output. All options exposed by the naga library —
bounds-check policies, backend language versions, zero-initialisation modes, and more — are
reachable from the command line or via a JSON config file.

---

## Install / build

```sh
cargo build -p naga-cli
```

The binary is called `naga` and is placed in `target/debug/naga` (or `target/release/naga` with
`--release`).

---

## Basic usage

The input format is inferred from the file extension (`wgsl`, `spv`, `glsl`, `hlsl`, `bin`).
The output format is inferred from the output file extension. When no output file is given, the
shader is validated only.

```sh
# Validate a WGSL shader (no output file → validation only).
naga shader.wgsl

# Translate WGSL to SPIR-V.
naga shader.wgsl out.spv

# Translate SPIR-V back to WGSL.
naga in.spv out.wgsl

# Translate WGSL to MSL, HLSL, or GLSL.
naga shader.wgsl out.metal
naga shader.wgsl out.hlsl
naga shader.wgsl out.glsl

# Read from stdin (--stdin-file-path tells naga which extension to use for format detection).
cat shader.wgsl | naga --stdin-file-path shader.wgsl

# Bulk-validate every file listed.
naga --bulk-validate a.wgsl b.wgsl c.spv
```

The example shader used in this README lives at `examples/triangle.wgsl`.

---

## Options: flags vs `--config`

There are two mutually exclusive ways to set translation options:

**Flags** — quick, per-option overrides:

```sh
# Set the target SPIR-V version.
naga shader.wgsl out.spv --spirv-version 1.3

# Control zero-initialisation of workgroup memory.
naga shader.wgsl out.spv --zero-initialize-workgroup-memory polyfill

# Disable loop bounding.
naga shader.wgsl out.spv --force-loop-bounding false

# Generate debug symbols.
naga shader.wgsl out.spv -g

# Compact the IR before output.
naga shader.wgsl out.spv --compact

# Restrict array index bounds checking.
naga shader.wgsl out.spv --index-bounds-check-policy restrict

# Target a specific HLSL shader model.
naga shader.wgsl out.hlsl --shader-model 60

# Target a specific Metal version.
naga shader.wgsl out.metal --metal-version 2.0
```

**Config file** — provides access to the full set of backend options via a JSON document.
Flags and config are mutually exclusive; use one or the other.

```sh
# Use a config file.
naga shader.wgsl out.spv --config options.json

# Or pass the JSON inline.
naga shader.wgsl out.spv --config-json '{"spv_out":{"lang_version":[1,3]}}'
```

`examples/options.json` shows a partial config (only specified keys are applied; everything else
uses defaults):

```json
{
  "spv_out": {
    "lang_version": [1, 3],
    "force_loop_bounding": false
  },
  "msl": {
    "lang_version": [2, 0]
  }
}
```

To see every configurable key and its type, print the JSON Schema:

```sh
naga --print-config-schema
```

> **Note:** `--print-config-schema` omits one field that cannot be expressed in JSON Schema —
> the SPIR-V capabilities set (`spv_out.capabilities`, type `FastHashSet<spirv::Capability>`,
> the SPIR-V writer's capability allow-list — distinct from the top-level `capabilities`
> validator/WGSL-frontend filter, which IS present in the schema) — but it is still accepted
> by `--config` and `--config-json`. All other fields, including the top-level `defines` map,
> are present in the schema.

---

## Structured output (`--format json`)

By default diagnostics are printed as human-readable text. Pass `--format json` to receive a
machine-readable JSON document containing diagnostics **and** reflection data. This is useful for
editor integrations, CI pipelines, and tooling that needs to inspect entry points or resources.

```sh
naga examples/triangle.wgsl --format json
```

Real output (the full document is printed to stdout):

```json
{
  "success": true,
  "diagnostics": [],
  "reflection": {
    "entry_points": [
      {
        "name": "vs_main",
        "stage": "Vertex",
        "workgroup_size": [0, 0, 0]
      },
      {
        "name": "fs_main",
        "stage": "Fragment",
        "workgroup_size": [0, 0, 0]
      }
    ],
    "resources": [
      {
        "name": "tint",
        "group": 0,
        "binding": 0,
        "address_space": "Uniform"
      }
    ],
    "overrides": []
  }
}
```

**Diagnostics shape:** each entry in `diagnostics` carries `severity` (`"error"` or
`"warning"`), `message`, an optional `location` (`file`, `line`, `column`, `length`), zero or
more `labels` (each with `message` and `location`), and a `notes` array (may be empty for
validation errors). SPIR-V parse errors have no source location.

**Reflection shape:** `entry_points` lists every shader stage with its `name`, `stage`, and
`workgroup_size`; `resources` lists every global binding with its `name`, `group`, `binding`, and
`address_space`; `overrides` lists pipeline-constant overrides.

---

## External tool hooks

naga can hand off output files to external tools. The tools must be on `PATH`.

```sh
# Validate SPIR-V output with spirv-val (Khronos SPIRV-Tools).
naga shader.wgsl out.spv --spirv-val

# Optimize SPIR-V output in place with spirv-opt -O.
naga shader.wgsl out.spv --spirv-opt

# Compile HLSL output to DXIL with dxc.
# (Requires dxc on PATH, e.g. from the DirectX Shader Compiler release.)
naga shader.wgsl out.hlsl --dxc
```

When `--dxc` is used, naga compiles each entry point in the HLSL file separately and writes the
DXIL binary to `<hlsl-stem>.<entry-point-name>.dxil`. For example, a shader with entry points
`vs_main` and `fs_main` produced from `out.hlsl` will emit `out.vs_main.dxil` and
`out.fs_main.dxil`.

---

## Notes / gotchas

- **Coordinate space.** Coordinate-space conversions are controlled by the top-level
  `--keep-coordinate-space` flag (or config key `keep_coordinate_space`), **not** by the
  per-backend `spv_out.flags` bitmask. Setting `ADJUST_COORDINATE_SPACE` directly in
  `spv_out.flags` will be overwritten by the top-level flag.

- **Task/mesh shader limits.** `--task-limits` and `--validate-mesh-output` are convenience
  flags that set options across all applicable backends. The underlying config keys are flat
  fields directly under each backend (there is no `common` sub-object in the JSON):
  - `--task-limits` sets `task_dispatch_limits`, which is a struct with two required fields:
    `max_mesh_workgroups_per_dim` (u32) and `max_mesh_workgroups_total` (u32).
  - `--validate-mesh-output` controls `mesh_shader_primitive_indices_clamp` (bool).

  Example (set these directly under the backend key):
  ```json
  {
    "spv_out": {
      "task_dispatch_limits": {
        "max_mesh_workgroups_per_dim": 65535,
        "max_mesh_workgroups_total": 65535
      },
      "mesh_shader_primitive_indices_clamp": true
    }
  }
  ```

- **`--print-config-schema` omissions.** The SPIR-V capabilities set
  (`spv_out.capabilities`, type `FastHashSet<spirv::Capability>` — the SPIR-V writer's
  capability allow-list, distinct from the top-level `capabilities` validator/WGSL-frontend
  filter which IS in the schema) is excluded from the printed schema because its type lacks
  a `JsonSchema` impl. It is still accepted by `--config` and `--config-json`. The top-level
  `defines` map and all other fields are present in the schema.

- **JSON diagnostics detail.** In `--format json` output, every diagnostic has severity
  `"error"` or `"warning"`. The `notes` array may be empty for validation errors. SPIR-V
  parse errors carry no source location.

- **`ZeroInitializeWorkgroupMemoryMode::Native`** is honoured only by the SPIR-V backend
  (via `VK_KHR_zero_initialize_workgroup_memory` / Vulkan 1.3). Other backends fall back to
  the polyfill.
