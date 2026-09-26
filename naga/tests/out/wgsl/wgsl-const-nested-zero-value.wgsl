const composed: vec4<u32> = vec4<u32>(vec2<u32>(0u, 0u), 7u, 9u);

@group(0) @binding(0)
var<storage, read_write> o: array<u32, 8>;
@group(0) @binding(1)
var<storage, read_write> v: vec4<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    o[0] = 0u;
    o[1] = 7u;
    o[2] = 0u;
    o[3] = bitcast<u32>(0f);
    o[4] = bitcast<u32>(1f);
    o[5] = 0u;
    o[6] = 9u;
    o[7] = 7u;
    v = composed;
    return;
}
