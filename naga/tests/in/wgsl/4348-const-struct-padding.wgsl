// Padding in a `constant` initializer that also calls a constructor must be
// written out byte by byte for Metal to compile it.

// Padding before a `packed_float3` initialized from a `float3`.
struct PackedVec3 {
    a: f32,
    c: vec3<f32>,
    d: f32,
}

// Padding before a matrix.
struct Matrix {
    a: i32,
    c: vec4i,
    d: mat3x2f,
}

// Trailing padding after a matrix.
struct TrailingPad {
    m: mat2x2f,
    a: f32,
}

const PACKED_VEC3 = PackedVec3(1.0, vec3<f32>(2.0), 3.0);
const MATRIX = Matrix(1, vec4i(3), mat3x2f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
const TRAILING_PAD = TrailingPad(mat2x2f(1.0, 2.0, 3.0, 4.0), 5.0);

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

fn packed_vec3(s: PackedVec3) -> f32 {
    return s.a + s.c.x + s.d;
}

fn matrix(s: Matrix) -> f32 {
    return f32(s.a) + f32(s.c.x) + s.d[0].x;
}

fn trailing_pad(s: TrailingPad) -> f32 {
    return s.m[0].x + s.a;
}

@compute @workgroup_size(1)
fn main() {
    out[0] = packed_vec3(PACKED_VEC3);
    out[1] = matrix(MATRIX);
    out[2] = trailing_pad(TRAILING_PAD);
}
