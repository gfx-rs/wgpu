@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(in_vertex_index) - 1);
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}


@group(0)
@binding(0)
var<storage, read_write> buffer: array<u32>; // Used as both input and output for convenience.

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute
@workgroup_size(1)
fn main_cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var value = buffer[0];

    for (var i = 0u; i < 128u; i += 1u) {
        value = pcg_hash(value);
    }

    buffer[0] = value;
}
