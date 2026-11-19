struct S {
    v: vec3<f32>,
    w: f32,
}

@group(0) @binding(0) var<storage, read_write> out: vec4<u32>;

@compute @workgroup_size(1)
fn main() {
    let s = S(vec3(1.0, 2.0, 3.0), 4.0);
    out = vec4(bitcast<vec3<u32>>(s.v), 0u);
}
