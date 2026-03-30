# Cooperative Matrix Extensions

🧪Experimental🧪

`wgpu` supports an experimental cooperative matrix feature when `Features::EXPERIMENTAL_COOPERATIVE_MATRIX` is enabled.
This exposes hardware-accelerated matrix multiply-accumulate (MMA) operations (for example, NVIDIA tensor cores,
Metal SIMD-group matrices, and Vulkan `VK_KHR_cooperative_matrix`).

**Note**: The features documented here may have bugs and are subject to breaking changes. The API and shader
semantics are expected to evolve. Please refer to the GitHub issue tracker for the latest status and discussions.

---

## Overview

Cooperative matrices allow a **workgroup** (or equivalent execution group) to collectively:

- load small matrix tiles from memory,
- perform matrix multiply-accumulate operations on those tiles, and
- store the results back to memory.

Conceptually, this is specialized hardware that evaluates:

> `C = A * B + C`

for relatively small tiles, but at very high throughput compared to composing the same operation from
scalar/vector instructions.

Cooperative matrix operations are most useful in workloads such as:

- machine learning and inference,
- dense linear algebra and scientific computing,
- image processing, filtering, and transforms.

The cooperative nature means that all lanes in the cooperating execution group must participate in
the operations; individual invocations cannot diverge.

Typical example:

- `A` is an M×K matrix.
- `B` is a K×N matrix.
- `C` is an M×N matrix, acting as the accumulator and result.

---

## Querying hardware support (host side)

Before using cooperative matrices in shaders, you must query what configurations your hardware and backend support.

On the `Adapter`, `wgpu` exposes:

- `Adapter::cooperative_matrix_properties() -> Vec<CooperativeMatrixProperties>`

Each `CooperativeMatrixProperties` describes a single supported configuration. Fields are:

- `m_size`: height of matrices A and C (type: `naga::CooperativeSize`)
- `n_size`: width of matrices B and C (type: `naga::CooperativeSize`)
- `k_size`: shared inner dimension of A and B (type: `naga::CooperativeSize`)
- `ab_type`: scalar element type for A and B (type: `naga::Scalar`)
- `cr_type`: scalar element type for C and the result (type: `naga::Scalar`)
- `saturating_accumulation`: `bool` indicating whether overflow clamping on accumulation
  is supported for this configuration

Example usage:

```/dev/null/cooperative-matrix-host.rs#L1-40
let coop_props = adapter.cooperative_matrix_properties();
for prop in &coop_props {
    println!(
        "{:?}x{:?}x{:?} - AB: {:?}, CR: {:?}, saturating: {}",
        prop.m_size, prop.n_size, prop.k_size,
        prop.ab_type, prop.cr_type,
        prop.saturating_accumulation,
    );
}
```

You **must**:

1. Enable `Features::EXPERIMENTAL_COOPERATIVE_MATRIX` on the `Device`.
2. Query `adapter.cooperative_matrix_properties()` and ensure that the configuration(s) you intend
   to use in WGSL are actually available on the running adapter/backend.
3. Treat the sizes and types as a contract between your shaders and the underlying hardware implementation.
   Using unsupported configurations is an error.

---

## Feature and backend requirements

### `wgpu` feature

- Using cooperative matrices requires enabling:
  - `Features::EXPERIMENTAL_COOPERATIVE_MATRIX`

This feature may be restricted to certain backends and hardware.

### Hardware / backend notes

These are general guidelines, not a complete compatibility matrix:

- **Metal**:
  - Requires Apple7+ (A14) or Mac2+ (M1) GPU with MSL 2.3+.
  - Strong support for 8×8 `f32`, 8×8 `f16`, and mixed-precision modes (e.g. `f16` A/B and `f32` accumulator C).
  - Implementation is based on SIMD-group matrix operations.

- **Vulkan**:
  - Requires the `VK_KHR_cooperative_matrix` extension.
  - Many NVIDIA and AMD GPUs support `f16` at 16×16 tile sizes and similar.
  - 8×8 `f32` support is hardware-dependent.
  - Exact configurations are enumerated by `Adapter::cooperative_matrix_properties()`.

- **Other backends**:
  - May not support cooperative matrices at all. In that case the feature will not be exposed, and
    `adapter.cooperative_matrix_properties()` will return an empty list.

> Always treat the properties returned at runtime as the source of truth.

---

## `wgpu` API surface

This section summarizes the host-side API elements related to cooperative matrices.
(For exact signatures and details, refer to the Rust documentation.)

### Adapter

- `Adapter::cooperative_matrix_properties() -> Vec<CooperativeMatrixProperties>`

Returns all cooperative matrix configurations supported by the adapter/backend.

### Structures

- `CooperativeMatrixProperties`
  - `m_size: naga::CooperativeSize`
  - `n_size: naga::CooperativeSize`
  - `k_size: naga::CooperativeSize`
  - `ab_type: naga::Scalar`
  - `cr_type: naga::Scalar`
  - `saturating_accumulation: bool`

The `naga` types (`CooperativeSize`, `Scalar`) are part of the shader translation layer and
determine the legal WGSL/cooperative matrix combinations.

There are currently no dedicated `wgpu` buffer or texture types for cooperative matrices; they are
expressed in WGSL as special value types accessed via pointers into ordinary `var<storage>` /
`var<workgroup>` / `var<private>` / etc.

---

## WGSL extension specification

Cooperative matrices are enabled and accessed via WGSL extensions. The exact extension spelling
may change; the details below describe the intended semantics.

### Enabling cooperative matrices in WGSL

Any WGSL program using cooperative matrices must declare an extension at the top of the shader, for example:

```/dev/null/example.wgsl#L1-3
enable wgpu_cooperative_matrix;
```

The shader is invalid if any cooperative matrix types or builtins are used without enabling this extension.

### Cooperative matrix types

A cooperative matrix is a value type parameterized by:

- tile size (M×N),
- scalar element type `T`, and
- role `R` indicating how the matrix participates in the multiply-accumulate:
  - `A`: left operand
  - `B`: right operand
  - `C`: accumulator / result

Conceptually:

```/dev/null/example.wgsl#L1-8
// A: MxK, B: KxN, C: MxN
type coop_matMxN<T, A>;
type coop_matMxN<T, B>;
type coop_matMxN<T, C>;
```

Concrete examples (sizes and types must match a supported configuration from
`Adapter::cooperative_matrix_properties`):

```/dev/null/example.wgsl#L10-20
// 8x8 single-precision tiles
alias CoopMatA = coop_mat8x8<f32, A>;
alias CoopMatB = coop_mat8x8<f32, B>;
alias CoopMatC = coop_mat8x8<f32, C>;

// 16x16 half-precision inputs, 16x16 f32 accumulator (mixed precision)
alias CoopMat16x16A = coop_mat16x16<f16, A>;
alias CoopMat16x16B = coop_mat16x16<f16, B>;
alias CoopMat16x16C = coop_mat16x16<f32, C>;
```

The actual set of legal `(M, N, T, R)` combinations is defined by the cooperative matrix
properties returned at runtime; shaders must not use arbitrary combinations.

### Roles and semantics

- `A` role:
  - Treated as the left operand in the multiplication. Has shape M×K.
  - Participates as `A` in `A * B + C`.

- `B` role:
  - Treated as the right operand in the multiplication. Has shape K×N.
  - Participates as `B` in `A * B + C`.

- `C` role:
  - Treated as accumulator and result. Has shape M×N.
  - Participates as `C` in `A * B + C`.

These roles are part of the type; they are not interchangeable.

### Cooperative matrix operations

WGSL provides built-in functions for operating on cooperative matrices. The exact spelling may
change; the semantics are:

#### `coopLoad`

Collectively load a tile from memory into a cooperative matrix.

```/dev/null/example.wgsl#L1-6
fn coopLoad<T, R>(
    ptr: ptr<STORAGE_CLASS, T>, // base pointer to scalar or vector elements
    stride: u32                  // stride (in elements) between rows/columns
) -> coop_matMxN<T, R>;
```

- Loads an M×N tile (or M×K / K×N, depending on role and operation) from memory pointed to by `ptr`.
- `stride` describes the layout in memory; it is usually the number of elements between adjacent rows.
- All invocations in the cooperative group must call `coopLoad` in a converged fashion.
- Memory address range must be valid and properly aligned for the scalar type.

> Implementation note: Each lane contributes to filling the tile based on an implementation-defined mapping from
> invocation/lane ID to sub-fragment of the matrix.

#### `coopStore`

Collectively store a cooperative matrix tile back to memory.

```/dev/null/example.wgsl#L8-13
fn coopStore<T, R>(
    value: coop_matMxN<T, R>,
    ptr: ptr<STORAGE_CLASS, T>,
    stride: u32
);
```

- Stores `value` into the memory region addressed by `ptr` with given `stride`.
- All invocations in the cooperative group must participate.
- The store must not alias overlapping tiles in undefined ways.

#### `coopMultiplyAdd`

Perform a matrix multiply-accumulate operation on cooperative matrices:

```/dev/null/example.wgsl#L15-23
fn coopMultiplyAdd<Tab, Tcr, MA, KA, KB, NB>(
    a: coop_matMAxKA<Tab, A>, // A: MAxKA tile
    b: coop_matKBxNB<Tab, B>, // B: KBxNB tile (KB == KA)
    c: coop_matMAxNB<Tcr, C>  // C: MAxNB accumulator/result
) -> coop_matMAxNB<Tcr, C>;
```

Semantics:

- Computes `C' = A * B + C`.
- Returns the resulting accumulator tile `C'`.
- Implies:
  - `KA == KB` (inner dimension must match).
  - Types `(Tab, Tcr)` must be one of the supported AB/CR combinations given by
    `CooperativeMatrixProperties`.
  - Sizes `(MA, NB, KA)` must match a supported `(m_size, n_size, k_size)` triple.

For example, with a supported configuration:

```/dev/null/example.wgsl#L25-39
enable wgpu_cooperative_matrix;

alias MatA = coop_mat8x8<f32, A>;
alias MatB = coop_mat8x8<f32, B>;
alias MatC = coop_mat8x8<f32, C>;

fn matmul_tile(
    ptr_a: ptr<storage, f32>,
    ptr_b: ptr<storage, f32>,
    ptr_c: ptr<storage, f32>,
    stride: u32,
) {
    let a: MatA = coopLoad<_, A>(ptr_a, stride);
    let b: MatB = coopLoad<_, B>(ptr_b, stride);
    let c: MatC = coopLoad<_, C>(ptr_c, stride);

    let result: MatC = coopMultiplyAdd(a, b, c);
    coopStore(result, ptr_c, stride);
}
```

If `saturating_accumulation` is true for the chosen configuration, then overflow during accumulation
is clamped (e.g. saturating arithmetic). If false, overflow behavior for the accumulator follows the
underlying scalar type semantics (e.g. IEEE-754 for floats).

### Workgroup cooperation and execution model

Cooperative matrix operations are **collective**:

- All invocations in the relevant execution group must execute each cooperative operation in uniform control flow:
  - Using `coopLoad`, `coopStore`, or `coopMultiplyAdd` in divergent control flow (e.g. some lanes taking
    a branch, others not) is undefined behavior.
  - The exact execution group may be a workgroup, a SIMD-group / subgroup, or another backend-specific
    granularity; shaders must treat it abstractly.

- The workgroup (or cooperating group) size is constrained by both:
  - the cooperative matrix configuration, and
  - backend-specific implementation details.

For portable code:

- Choose a workgroup size that is known to be supported efficiently on your target backends, for example:
  - `@workgroup_size(8, 8, 1)` to operate on an 8×8 tile, or
  - a multiple of the tile size where each subgroup handles a tile.

- Avoid control-flow divergence around cooperative operations.

Example:

```/dev/null/example.wgsl#L1-42
enable wgpu_cooperative_matrix;

struct Matrices {
    // Row-major tiles for A, B, C.
    data: array<f32>,
};

@group(0) @binding(0)
var<storage, read>  buf_a: Matrices;
@group(0) @binding(1)
var<storage, read>  buf_b: Matrices;
@group(0) @binding(2)
var<storage, read_write> buf_c: Matrices;

alias MatA = coop_mat8x8<f32, A>;
alias MatB = coop_mat8x8<f32, B>;
alias MatC = coop_mat8x8<f32, C>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Compute tile offset; this is one of many possible mappings.
    let tile_index = wg_id.x; // 1D tiling in this simple example
    let tile_offset = tile_index * 64u; // 8x8 tile has 64 elements

    // Base pointers for tiles of A, B, C.
    let base_a = &buf_a.data[tile_offset];
    let base_b = &buf_b.data[tile_offset];
    let base_c = &buf_c.data[tile_offset];

    let a: MatA = coopLoad<f32, A>(base_a, 8u);
    let b: MatB = coopLoad<f32, B>(base_b, 8u);
    let c: MatC = coopLoad<f32, C>(base_c, 8u);

    let result: MatC = coopMultiplyAdd(a, b, c);
    coopStore(result, base_c, 8u);
}
```

---

## Validation rules and undefined behavior

Implementations must validate the following where possible:

- The `wgpu_cooperative_matrix` WGSL extension is enabled if any cooperative matrix types
  or builtins are used.
- Tile sizes `(M, N, K)` and scalar types `(ab_type, cr_type)` match at least one
  `CooperativeMatrixProperties` entry for the current adapter/backend.
- Workgroup size, shader stage, and other pipeline configuration constraints required
  by the backend are satisfied.

The following are examples of **undefined behavior** (non-exhaustive):

- Using cooperative matrix operations without enabling the WGSL extension.
- Using a cooperative matrix type `(M, N, T, R)` not supported by
  `Adapter::cooperative_matrix_properties()`.
- Mismatching sizes or roles in `coopMultiplyAdd` (e.g. incompatible M/N/K, or incorrect roles).
- Executing `coopLoad`, `coopStore`, or `coopMultiplyAdd` in divergent control flow within the
  cooperating execution group.
- Providing invalid, misaligned, or out-of-bounds pointers to `coopLoad` / `coopStore`.
- Overlapping `coopStore` targets in a way that creates data races or aliasing that the memory
  model does not allow.

---

## Example: 64×64 matrix multiply using 8×8 tiles

The example in `examples/features/src/cooperative_matrix` demonstrates using cooperative matrices to
compute:

- `C = A * B + C` where:
  - `A` is 64×64,
  - `B` is 64×64,
  - `C` is 64×64.

A high-level tiling strategy:

1. Partition A, B, and C into 8×8 tiles.
2. Launch one workgroup per output tile of C (i.e. 8×8 tiles for a 64×64 matrix = 8×8 = 64 tiles).
3. Within each workgroup:
   - Loop over K-dimension tiles.
   - For each `k` tile:
     - Load an 8×8 tile of A (`MatA`).
     - Load an 8×8 tile of B (`MatB`).
     - Maintain an 8×8 accumulator tile (`MatC`) and repeatedly apply `coopMultiplyAdd`.
4. After the K loop, store the final accumulator tile back to C.

Key points from the example:

- Workgroup size is chosen so that all cooperative operations are well-defined and efficient for 8×8 tiles.
- Host-side code:
  - Enables `Features::EXPERIMENTAL_COOPERATIVE_MATRIX`.
  - Queries `cooperative_matrix_properties` and verifies that 8×8 `f32` or chosen configuration is supported.
  - Dispatches the compute pipeline with appropriate grid dimensions.

---

## Notes and best practices

- Always query `adapter.cooperative_matrix_properties()` and check that the configuration your shaders use exists.
  Do not hard-code assumptions about available tile sizes or element types.
- Treat the cooperative execution group as an abstract concept; avoid making assumptions about how
  tiles are mapped to lanes beyond what is guaranteed by the spec.
- Avoid divergent control flow around cooperative operations.
- Consider providing a fallback non-cooperative implementation for devices that do not support the feature.
- This is an experimental extension; API and semantics may change across versions of `wgpu` and `naga`.
