// Must have "wgpu-core/src/timestamp_normalization/common.wgsl"
// preprocessed before this file's contents.

struct U64MulU32Input {
    left: Uint64,
    right: u32,
    _padding: u32,
}

@group(0) @binding(0)
var<storage> input: array<U64MulU32Input>;

@group(0) @binding(1)
var<storage, read_write> output: array<Uint96>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;

    let input = input[index];

    output[index] = u64_mul_u32(input.left, input.right);
}
