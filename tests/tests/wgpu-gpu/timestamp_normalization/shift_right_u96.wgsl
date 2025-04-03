// Must have "wgpu-core/src/timestamp_normalization/common.wgsl"
// preprocessed before this file's contents.

struct ShiftRight96 {
    value: Uint96,
    shift: u32,
}

@group(0) @binding(0)
var<storage> input: array<ShiftRight96>;

@group(0) @binding(1)
var<storage, read_write> output: array<Uint96>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;

    let input = input[index];

    output[index] = shift_right_96(input.value, input.shift);
}
