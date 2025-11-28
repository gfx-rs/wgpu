# Cooperative Matrix Multiplication

This example demonstrates how to use cooperative matrix operations (also known as tensor cores on NVIDIA GPUs) to perform efficient matrix multiplication on the GPU.

## Overview

Cooperative matrices allow a workgroup to collectively load, store, and perform matrix multiply-accumulate operations on small tiles of data. This enables hardware-accelerated matrix math that can be significantly faster than traditional element-wise approaches.

The example computes `C = A * B + C` where:
- A is a 64×64 matrix
- B is a 64×64 matrix
- C is a 64×64 matrix (accumulator/result)

## Querying Supported Configurations

Before using cooperative matrices, you should query what configurations your hardware supports:

```rust
let coop_props = adapter.cooperative_matrix_properties();
for prop in &coop_props {
    println!(
        "{:?}x{:?}x{:?} - AB: {:?}, CR: {:?}",
        prop.m_size, prop.n_size, prop.k_size,
        prop.ab_type, prop.cr_type
    );
}
```

Each `CooperativeMatrixProperties` describes a supported configuration with:
- `m_size`, `n_size`, `k_size`: Matrix dimensions as `naga::CooperativeSize` (M×K × K×N → M×N)
- `ab_type`: Element type for input matrices A and B (as `naga::Scalar`)
- `cr_type`: Element type for accumulator matrix C and the result
- `saturating_accumulation`: Whether overflow clamping is supported

## Key Concepts

### Cooperative Matrix Types

In WGSL, cooperative matrices are declared with a specific size, element type, and role:

```wgsl
coop_mat8x8<f32, A>  // Matrix A (left operand)
coop_mat8x8<f32, B>  // Matrix B (right operand)
coop_mat8x8<f32, C>  // Matrix C (accumulator)
```

The role (A, B, or C) determines how the matrix is used in multiply-accumulate operations.

### Operations

- `coopLoad<T>(pointer, stride)` - Cooperatively load a tile from memory
- `coopStore(matrix, pointer, stride)` - Cooperatively store a tile to memory
- `coopMultiplyAdd(a, b, c)` - Compute `a * b + c`

### Workgroup Cooperation

All threads in a workgroup must participate in cooperative matrix operations together. The workgroup size should match the cooperative matrix dimensions (8×8 in this example).

## Requirements

- GPU with cooperative matrix support:
  - Metal: Apple7+ (A14 chip) or Mac2+ (M1 chip) with MSL 2.3+
    - Supports 8x8 f32, 8x8 f16, and mixed precision (f16 inputs, f32 accumulator)
  - Vulkan: Requires VK_KHR_cooperative_matrix extension
    - Most NVIDIA/AMD GPUs support f16 at 16x16 sizes
    - 8x8 f32 support varies by hardware
- `Features::EXPERIMENTAL_COOPERATIVE_MATRIX` must be enabled
- Use `adapter.cooperative_matrix_properties()` to check available configurations

## Running

```bash
cargo run --bin wgpu-examples -- cooperative_matrix
```

## Notes

- This is an experimental feature and may not work on all hardware
- Always query `adapter.cooperative_matrix_properties()` to check what's supported
- The 8x8 f32 matrix format is well supported on Metal (simdgroup matrix operations)
- Vulkan support depends on hardware - most GPUs (NVIDIA, AMD) support f16 inputs at 16x16 sizes
- The shader uses the standard `create_shader_module` with full validation
- Results are verified against a CPU reference implementation
