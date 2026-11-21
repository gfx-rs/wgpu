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

The input format is inferred from the file extension (`wgsl`, `glsl`, `spv`, `bin` — there is no
HLSL or MSL frontend). The output format is inferred from the output file extension (`spv`,
`metal`, `hlsl`, `wgsl`, `dot`, `txt` for the IR, `bin` for the serialized IR, and `vert`/`frag`/
`comp` for GLSL). When no output file is given, the
shader is validated only. To write to **stdout** instead of a file, use `-` as the output path
and give the format explicitly with `--output-kind` (there is no extension to infer from):

```sh
naga shader.wgsl - --output-kind hlsl      # print HLSL to stdout
naga shader.wgsl - --output-kind spv > out.spv
```

Stdout output cannot be combined with `--format json` unless `--json-out <file>` sends the JSON
document to a file. The SPIR-V tool hooks below work with stdout output (they run on a temporary
file). `--dxc` with stdout output requires `--entry-point` and writes that entry point's DXIL to
stdout instead of the HLSL.

```sh
# Validate a WGSL shader (no output file → validation only).
naga shader.wgsl

# Translate WGSL to SPIR-V.
naga shader.wgsl out.spv

# Translate SPIR-V back to WGSL.
naga in.spv out.wgsl

# Translate WGSL to MSL or HLSL.
naga shader.wgsl out.metal
naga shader.wgsl out.hlsl

# Translate to GLSL — the output extension picks the stage (.vert / .frag / .comp),
# since the GLSL backend emits one stage/entry point at a time. The entry point defaults
# to `main`; pass --entry-point to select another.
naga shader.wgsl out.frag --entry-point fs_main

# Read from stdin: use `-` as the input path and name the format with --input-kind.
cat shader.wgsl | naga - --input-kind wgsl

# Bulk-validate every file listed.
naga --bulk-validate a.wgsl b.wgsl c.spv
```

The examples below use this small WGSL shader, `triangle.wgsl`:

```wgsl
@group(0) @binding(0) var<uniform> tint: vec4<f32>;

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(i), 0.0, 0.0, 1.0);
}

@fragment fn fs_main() -> @location(0) vec4<f32> { return tint; }
```

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
naga shader.wgsl out.hlsl --hlsl-shader-model 60

# HLSL for a pipeline with multisampled attachments.
naga shader.wgsl out.hlsl --hlsl-multisampled

# Target a specific Metal version.
naga shader.wgsl out.metal --metal-version 2.0

# Target a GLSL version (core140..core460, es300/es310/es320, webgl).
naga shader.wgsl out.frag --entry-point fs_main --glsl-version es310

# Write WGSL with explicit types everywhere.
naga shader.wgsl out.wgsl --wgsl-explicit-types
```

Each language-version flag (`--glsl-version`, `--hlsl-shader-model`, `--metal-version`,
`--spirv-version`) accepts only the versions naga supports; `--help` lists them.

**Config file** — provides access to the full set of options via a JSON document.
Flags and config are **mutually exclusive**: passing any flag other than `--color` alongside
`--config` is a hard error (clap enforces it at parse time). In config mode only the positional input/output file
paths are accepted on the command line — everything else goes in the JSON.

**Every flag has a corresponding config key**, so config is a complete single source — including
things that are otherwise flags. For example, reading stdin / writing stdout via `-` works in
config mode by setting the format in the JSON (the `--input-kind` / `--output-kind` flags
themselves are exclusive with `--config`):

```sh
cat shader.wgsl | naga - - --config-json '{"input_kind":"wgsl","output_kind":"spv"}' > out.spv
```

Likewise `format` (`"json"` for structured diagnostics + reflection), `before_compaction`, and
`bulk_validate` are all config keys.

```sh
# Use a config file.
naga shader.wgsl out.spv --config options.json

# Or pass the JSON inline.
naga shader.wgsl out.spv --config-json '{"spv_out":{"lang_version":[1,3]}}'
```

A partial config (only the specified keys are applied; everything else uses defaults):

```json
{
  "compact": true,
  "spirv_val": true,
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



---

## Structured output (`--format json`)

By default diagnostics are printed as human-readable text. Pass `--format json` to receive a
machine-readable JSON document containing diagnostics **and** reflection data. This is useful for
editor integrations, CI pipelines, and tooling that needs to inspect entry points or resources.

```sh
naga triangle.wgsl --format json
```

Abridged real output (`types` and `vs_main` trimmed; run it to see the full document):

```json
{
  "version": 1,
  "success": true,
  "diagnostics": [],
  "reflection": {
    "types": [
      { "name": null, "size": 16, "align": 16,
        "inner": { "vector": { "size": "Quad", "scalar": { "kind": "Float", "width": 4 } } } }
    ],
    "resources": [
      {
        "name": "tint",
        "binding": { "group": 0, "binding": 0 },
        "space": "Uniform",
        "ty": 0,
        "kind": { "buffer": { "minBindingSize": 16 } },
        "bindingArraySize": null
      }
    ],
    "overrides": [],
    "entryPoints": [
      {
        "name": "fs_main",
        "stage": "Fragment",
        "inputs": [],
        "outputs": [
          { "location": { "name": null, "location": 0,
              "ty": { "vector": { "size": "Quad", "scalar": { "kind": "Float", "width": 4 } } },
              "interpolation": "Perspective", "sampling": "Center",
              "perPrimitive": false, "blendSrc": null } }
        ],
        "resources": [0],
        "samplingPairs": [],
        "workgroupSize": [0, 0, 0],
        "workgroupSizeFromOverrides": [false, false, false],
        "workgroupStorageSize": 0,
        "dualSourceBlending": false,
        "taskPayloadSize": null,
        "meshInfo": null,
        "immediateUsage": { "Valid": { "slots": 0, "size": 0 } },
        "rayTracingPayload": null
      }
    ]
  }
}
```

**Diagnostics shape:** each entry in `diagnostics` carries `severity` (`"error"` or
`"warning"`), `message`, an optional `location` (`line`, `column`, `byteOffset`, `length` — no
file name, since the input is a single file/stream), zero or more `labels` (each with `message`
and an optional `location`), and a `notes` array (may be empty for validation errors). SPIR-V
parse errors have no source location.

To write the shader to stdout and the JSON document to a file, use `--json-out` (it implies
`--format json`):

```sh
naga shader.wgsl - --output-kind spv --json-out report.json > out.spv
```

**Reflection shape:** reflection comes from `naga::reflect` and describes the module before
pipeline overrides are applied. Indices (`ty`, `resources`, `samplingPairs`, `rayTracingPayload`)
point into the top-level `types` and `resources` arrays.

- `types`: every type in the module with its `size` and `align` in bytes (runtime-sized arrays
  count one element), and its shape (struct members carry `offset`).
- `resources`: every `@group`/`@binding` global, with its address `space`, declared type, `kind`
  (`buffer` with `minBindingSize`, `texture`, `sampler`, or `accelerationStructure`), and the
  element count of binding arrays.
- `overrides`: name, `@id`, type, and whether it has a default (overrides without one must be
  given a value when creating a pipeline).
- `entryPoints`: per entry point, the flattened `inputs` and `outputs` (vertex attributes,
  inter-stage variables, fragment outputs including `blendSrc`, built-ins, and mesh vertex and
  primitive outputs), the `resources` it statically uses, texture/sampler pairs, workgroup size
  (and which dimensions come from overrides), workgroup memory size, dual-source blending, task
  payload size, mesh output limits, immediate data usage, and ray tracing payload type.

The document carries a `version` number, but the format is not yet stable. Print its JSON Schema
with:

```sh
naga --print-output-schema
```

---

## External tool hooks

naga can hand off output files to external tools. The tools are looked up on `PATH` unless
`--spirv-val-path`, `--spirv-opt-path`, or `--dxc-path` names the executable. These are also
config keys (`spirv_val` / `spirv_opt` / `dxc` and `spirv_val_path` / `spirv_opt_path` /
`dxc_path`), so set them in `--config` rather than as flags when using a config file.

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
`out.fs_main.dxil`. With `--entry-point`, only that entry point is written and compiled.

The hooks also work with stdout output. SPIR-V is round-tripped through a temporary file.
`--dxc` requires `--entry-point` in that case and writes the DXIL to stdout:

```sh
naga shader.wgsl - --output-kind spv --spirv-val > out.spv
naga shader.wgsl - --output-kind hlsl --dxc --entry-point fs_main > fs_main.dxil
```

---

## Notes / gotchas

Run `naga --help` for the full list of behavioural notes (coordinate-space handling,
task/mesh limit fan-out, schema omissions, `Native` zero-init being SPIR-V-only, JSON
diagnostic details). One config-authoring detail worth repeating here: `--task-limits` and
`--validate-mesh-output` are conveniences that fan out to every applicable backend; in a
config, set the flat per-backend keys directly (there is **no** `common` sub-object in the
JSON). `task_dispatch_limits` is a struct with two required `u32` fields:

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
