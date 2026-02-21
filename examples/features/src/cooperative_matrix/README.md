# Cooperative Matrix Multiplication

This example demonstrates how to use cooperative matrix operations (also known as tensor cores on NVIDIA GPUs) to perform efficient matrix multiplication on the GPU.

For the full description of the cooperative matrix feature (supported configurations, WGSL types and operations, validation rules, and backend support), see the central API spec:

- `docs/api-specs/cooperative_matrix.md`

## Example specifics

This example computes `C = A * B + C` where:
- A is a 64×64 matrix
- B is a 64×64 matrix
- C is a 64×64 matrix (accumulator/result)

The example:
- Tiles the 64×64 matrices into cooperative matrix tiles (e.g. 8×8) and performs a tiled matmul
- Uses a compute shader and compares GPU results against a CPU reference implementation

## Requirements

- A GPU and backend that expose `Features::EXPERIMENTAL_COOPERATIVE_MATRIX`
- A configuration returned from `adapter.cooperative_matrix_properties()` that matches the tile size and element types used by this example
- See `docs/api-specs/cooperative_matrix.md` for details on hardware / backend support

## Running

```bash
cargo run --bin wgpu-examples -- cooperative_matrix
```

## Notes

- This is an experimental feature and may not work on all hardware
- The shader uses the standard `create_shader_module` with full validation
- Results are verified against a CPU reference implementation
