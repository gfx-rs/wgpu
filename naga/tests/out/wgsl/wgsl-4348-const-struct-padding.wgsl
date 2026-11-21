struct PackedVec3_ {
    a: f32,
    c: vec3<f32>,
    d: f32,
}

struct Matrix {
    a: i32,
    c: vec4<i32>,
    d: mat3x2<f32>,
}

struct TrailingPad {
    m: mat2x2<f32>,
    a: f32,
}

const PACKED_VEC3_: PackedVec3_ = PackedVec3_(1f, vec3(2f), 3f);
const MATRIX: Matrix = Matrix(1i, vec4(3i), mat3x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f), vec2<f32>(5f, 6f)));
const TRAILING_PAD: TrailingPad = TrailingPad(mat2x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f)), 5f);

@group(0) @binding(0)
var<storage, read_write> out: array<f32>;

fn packed_vec3_(s: PackedVec3_) -> f32 {
    return ((s.a + s.c.x) + s.d);
}

fn matrix(s_1: Matrix) -> f32 {
    return ((f32(s_1.a) + f32(s_1.c.x)) + s_1.d[0].x);
}

fn trailing_pad(s_2: TrailingPad) -> f32 {
    return (s_2.m[0].x + s_2.a);
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let _e3 = packed_vec3_(PACKED_VEC3_);
    out[0] = _e3;
    let _e7 = matrix(MATRIX);
    out[1] = _e7;
    let _e11 = trailing_pad(TRAILING_PAD);
    out[2] = _e11;
    return;
}
