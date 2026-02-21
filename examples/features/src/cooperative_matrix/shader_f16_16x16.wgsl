// Cooperative Matrix Multiplication Example (16x16 f16 variant)
//
// This shader demonstrates using cooperative matrix operations to perform
// matrix multiplication: C = A * B + C
//
// This variant uses 16x16 tiles with f16 precision, which is supported
// on AMD GPUs via VK_KHR_cooperative_matrix.
//
// The matrices are stored in row-major order in storage buffers.
// Each workgroup cooperatively loads tiles of A and B, multiplies them,
// and accumulates the result into C.

enable f16;
enable wgpu_cooperative_matrix;

// Matrix dimensions (16x16 tiles)
const TILE_SIZE: u32 = 16u;

@group(0) @binding(0)
var<storage, read> matrix_a: array<f16>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f16>;

@group(0) @binding(2)
var<storage, read_write> matrix_c: array<f16>;

// Dimensions passed as uniforms: M, N, K for C[M,N] = A[M,K] * B[K,N]
@group(0) @binding(3)
var<uniform> dimensions: vec4<u32>; // x=M, y=N, z=K, w=stride

// Workgroup size must be a multiple of subgroup size (64 on AMD)
// Using 64x1x1 ensures compatibility while still processing 16x16 tiles
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let M = dimensions.x;
    let N = dimensions.y;
    let K = dimensions.z;
    let stride = dimensions.w;

    // Each workgroup handles one 16x16 tile of the output matrix C
    let tile_row = workgroup_id.x * TILE_SIZE;
    let tile_col = workgroup_id.y * TILE_SIZE;

    // Load the C tile (accumulator)
    let c_offset = tile_row * stride + tile_col;
    var c_tile = coopLoad<coop_mat16x16<f16, C>>(&matrix_c[c_offset], stride);

    // Iterate over K dimension in tiles
    for (var k: u32 = 0u; k < K; k += TILE_SIZE) {
        // Load A tile: rows [tile_row, tile_row+16), cols [k, k+16)
        let a_offset = tile_row * K + k;
        let a_tile = coopLoad<coop_mat16x16<f16, A>>(&matrix_a[a_offset], K);

        // Load B tile: rows [k, k+16), cols [tile_col, tile_col+16)
        let b_offset = k * stride + tile_col;
        let b_tile = coopLoad<coop_mat16x16<f16, B>>(&matrix_b[b_offset], stride);

        // Multiply and accumulate: C += A * B
        c_tile = coopMultiplyAdd(a_tile, b_tile, c_tile);
    }

    // Store the result back to C
    coopStore(c_tile, &matrix_c[c_offset], stride);
}
