struct S {
    a: f32,
    m: mat3x2<f32>,
}

fn store_column(p: ptr<function, mat3x2<f32>>) {
    (*p)[1] = vec2(5.0, 6.0);
}

fn read_matrix(p: ptr<function, mat3x2<f32>>) -> f32 {
    return (*p)[2].y;
}

@group(0) @binding(0) var<storage, read_write> out: array<f32, 4>;

@compute @workgroup_size(1)
fn main() {
    var s: S;
    s.m = mat3x2<f32>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    store_column(&s.m);
    out[0] = s.m[1].y;
    out[1] = read_matrix(&s.m);
}
