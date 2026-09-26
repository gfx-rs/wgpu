// Constant evaluation of vectors composed from zero-valued vectors.
// See https://github.com/gfx-rs/wgpu/issues/10463.

const composed = vec4u(vec2u(), 7u, 9u);

@group(0) @binding(0)
var<storage, read_write> o: array<u32, 8>;

@group(0) @binding(1)
var<storage, read_write> v: vec4u;

@compute @workgroup_size(1)
fn main() {
    o[0] = vec4u(vec2u(), 7u, 9u).y;
    o[1] = vec4u(vec2u(), 7u, 9u).z;
    o[2] = vec4u(vec2u(), 7u, 9u)[1];
    o[3] = bitcast<u32>(vec4f(1.0, vec2f(), 2.0).z);
    o[4] = bitcast<u32>(vec4f(vec3f(), 1.0).w);
    o[5] = vec4u(7u, vec2u(), 9u).y;
    o[6] = vec4u(vec2u(), 7u, 9u).wzyx.x;
    o[7] = composed.z;
    v = composed;
}
