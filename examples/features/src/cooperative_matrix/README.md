# Cooperative Matrix Multiplication

This example demonstrates how to use cooperative matrix operations (also known as tensor cores on NVIDIA GPUs) to perform efficient matrix multiplication on the GPU.

## Overview

Cooperative matrices allow a workgroup to collectively load, store, and perform matrix multiply-accumulate operations on small tiles of data. This enables hardware-accelerated matrix math that can be significantly faster than traditional element-wise approaches.

The example computes `C = A * B + C` where:
- A is a 64×64 matrix
- B is a 64×64 matrix
- C is a 64×64 matrix (accumulator/result)

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
  - Vulkan: Requires VK_KHR_cooperative_matrix with 8x8 f32 support (rare - most GPUs support f16 at 16x16 sizes)
- Experimental features must be enabled

## Running

```bash
cargo run --bin wgpu-examples -- cooperative_matrix
```

## Notes

- This is an experimental feature and may not work on all hardware
- The 8x8 f32 matrix format is well supported on Metal (simdgroup matrix operations)
- Vulkan support depends on hardware - most GPUs (NVIDIA, AMD) support f16 inputs at 16x16 sizes,
  so 8x8 f32 support via VK_KHR_cooperative_matrix may be limited
- The shader uses the standard `create_shader_module` with full validation
- Results are verified against a CPU reference implementation
