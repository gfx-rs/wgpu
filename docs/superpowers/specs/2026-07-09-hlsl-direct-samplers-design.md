# HLSL backend: direct sampler binding (no sampler heap)

## Problem

Naga's HLSL backend always binds samplers through a D3D12 "sampler heap"
indirection. It declares two global heaps (`nagaSamplerHeap[2048]` and
`nagaComparisonSamplerHeap[2048]`), a per-bind-group `StructuredBuffer<uint>`
index buffer, and accesses every sampler as
`static const SamplerState s = nagaSamplerHeap[indexBuffer[reg]];`.

There is no way to emit an ordinary HLSL sampler bound directly to a register.
Consumers that manage their own sampler descriptors (i.e. bind samplers the
"normal" way) cannot use the backend. We want an option to disable the heap
indirection.

## Scope

- **In scope:** the naga HLSL backend (`naga/src/back/hlsl`) plus a snapshot
  test exercising the new mode.
- **Out of scope:** wiring the option through `wgpu-hal`'s dx12 backend or
  `wgpu-core`. The dx12 HAL owns descriptor-heap allocation and root-signature
  construction; making it emit direct samplers at runtime is a separate, much
  larger change. This spec only gives naga the capability.

## Design

### New public API

In `naga/src/back/hlsl/mod.rs`, add an enum describing how samplers are bound:

```rust
/// How the HLSL backend binds sampler global variables.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum SamplerBinding {
    /// Bind the entire D3D12 sampler heap (as both a standard and a comparison
    /// sampler heap) plus a per-bind-group sampler index buffer, and access
    /// each sampler through that indirection.
    ///
    /// This is required by the `wgpu` dx12 backend. See the module
    /// documentation's "Sampler Handling" section for details.
    #[default]
    Heap,
    /// Declare each sampler as an ordinary HLSL `SamplerState` /
    /// `SamplerComparisonState` bound directly to the register given by its
    /// [`Options::binding_map`] entry, with no heap indirection.
    Direct,
}
```

Add a field to `Options`:

```rust
/// How sampler global variables are bound. Defaults to [`SamplerBinding::Heap`].
pub sampler_binding: SamplerBinding,
```

`Options::default()` sets it to `SamplerBinding::Heap`. Because the field uses
serde `default` (the `Options` struct already carries `#[serde(default)]`), it
is optional in deserialized configs and defaults to `Heap`, so existing
callers and all existing snapshots are unchanged.

### Behavior in `Direct` mode

Only two sites in `writer.rs` branch on the mode; the heap/index-buffer writers
in `help.rs` are simply never reached because nothing calls them.

**1. `Writer::write_global_sampler` (`writer.rs`, ~line 1224).**

In `Heap` mode: unchanged.

In `Direct` mode: do **not** call `write_wrapped_sampler_buffer`. Resolve the
sampler's `BindTarget` from `binding_map` (via `resolve_resource_binding`) and
treat its `register` / `space` as a real HLSL register (symmetric with how
textures and buffers are already emitted). Emit:

- `TypeInner::Sampler { comparison }`:
  ```
  SamplerState <name> : register(s<reg>[, space<space>]);
  ```
  using `SamplerComparisonState` when `comparison` is true.
- `TypeInner::BindingArray { .. }` of a sampler:
  ```
  SamplerState <name>[<size>] : register(s<reg>[, space<space>]);
  ```
  Array size comes from the existing `write_array_size` / `binding_array_size`
  logic. `space<space>` is omitted when the space is 0, matching the existing
  register-writing convention elsewhere in the writer.

The `space` register-suffix is only written when non-zero, consistent with the
other `register(...)` emitters in `write_global`.

**2. `Writer::sampler_binding_array_info_from_expression` (`writer.rs`, ~line
4733).**

In `Heap` mode: unchanged (returns `Some(info)`, which drives the
`heap[indexBuffer[base + i]]` access form).

In `Direct` mode: return `None`. With `None`, `Access` / `AccessIndex`
expression codegen falls through to plain HLSL array indexing (`<name>[i]`),
which is correct for a directly-declared sampler array.

### What is intentionally *not* changed

- Heap / index-buffer writers (`write_sampler_heaps`,
  `write_wrapped_sampler_buffer` in `help.rs`) — unreachable in `Direct` mode,
  left as-is.
- `sampler_heap_target` and `sampler_buffer_binding_map` options — unused in
  `Direct` mode, left as-is (no deprecation).
- The `SamplerState` / `SamplerComparisonState` type emission in `write_type`
  already produces the right HLSL type name; reused unchanged.

## Testing

### Snapshot test

Add `naga/tests/in/wgsl/hlsl-sampler-binding-direct.wgsl` containing, at minimum:

- a `texture_2d<f32>` + a `sampler`, used in a `textureSample` — exercises the
  direct scalar sampler path;
- a `sampler_comparison`, used in a `textureSampleCompare` — exercises
  `SamplerComparisonState`;
- a `binding_array<sampler>` indexed by a value — exercises the direct sampler
  binding-array path (requires the
  `TEXTURE_AND_SAMPLER_BINDING_ARRAY` capability).

Add `naga/tests/in/wgsl/hlsl-sampler-binding-direct.toml`:

- `targets = "HLSL"` (HLSL only — this feature is HLSL-specific);
- required `capabilities` for the binding array;
- `[hlsl]` section with `sampler_binding = "Direct"`, `fake_missing_bindings`,
  and a `binding_map` giving every texture and sampler an explicit
  `{ register, space }` (and `binding_array_size` for the array).

Running the snapshot suite writes `naga/tests/out/hlsl/hlsl-sampler-binding-direct.hlsl`
and `.ron`. Manually verify the generated HLSL:

- contains `SamplerState ... : register(s...)` and
  `SamplerComparisonState ... : register(s...)` declarations;
- contains **no** `nagaSamplerHeap`, `nagaComparisonSamplerHeap`, or
  `StructuredBuffer<uint>` sampler-index-buffer declarations;
- indexes the sampler array directly (`name[i]`), not through a heap.

### Regression

The full existing snapshot suite (`cargo test -p naga --test naga`) must still
pass with **no** changes to any existing `out/hlsl/*` file — proving the
`Heap` default is byte-for-byte unchanged.

## Risks / notes

- FXC/DXC are not run in the snapshot tests, so correctness of the emitted HLSL
  is verified by inspection of the snapshot, not by compilation. This matches
  how the rest of the HLSL snapshot suite works.
- `binding_array<sampler>` requires the `TEXTURE_AND_SAMPLER_BINDING_ARRAY`
  capability; the test toml must enable it or validation fails.
