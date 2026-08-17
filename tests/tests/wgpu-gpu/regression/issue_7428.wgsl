@group(0) @binding(0)
var tex: texture_2d_array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<vec4f>;

@compute @workgroup_size(1)
fn main() {
    output[0] = textureLoad(tex, vec2i(0, 0), 0, 0);
}
