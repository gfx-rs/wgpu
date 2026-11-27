enable f16;

@group(0) @binding(0)
var<storage, read_write> output: array<f16, 4>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let x = output[0];
    output[0] = modf(x).fract;
    output[1] = modf(vec2(x)).whole.x;
    output[2] = modf(vec3(x)).fract.y;
    output[3] = modf(vec4(x)).whole.z;
    return;
}
