enable f16;

@group(0) @binding(0)
var<storage, read_write> values: array<vec4<f16>>; // this is used as both values and output for convenience

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    values[global_id.x] = fma(values[0], values[0], values[0]);
}
