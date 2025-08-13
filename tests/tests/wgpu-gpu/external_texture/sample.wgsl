@group(0) @binding(0)
var tex: texture_external;
@group(0) @binding(1)
var samp: sampler;

@group(0) @binding(2)
var<storage, read> coords: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output[id.x] = textureSampleBaseClampToEdge(tex, samp, coords[id.x]);
}
