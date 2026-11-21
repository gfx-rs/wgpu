struct S {
    a: f32,
    m: mat3x2<f32>,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32, 4>;

@compute @workgroup_size(1)
fn main() {
    var s: S;
    s.m = mat3x2<f32>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    out[0] = s.m[0].x;
    out[1] = s.m[2].y;
}
