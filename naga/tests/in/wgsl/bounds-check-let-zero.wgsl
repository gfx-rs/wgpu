// Indices that are `let`-bound and out of bounds, under the `ReadZeroSkipWrite` policy.
//
// WGSL `let` bindings are not const-expressions, so these must not be rejected
// at shader creation. Back ends must still produce valid output for them.

struct Globals {
    a: array<f32, 4>,
    v: vec4<f32>,
    m: mat3x4<f32>,
    d: array<f32>,
}

@group(0) @binding(0) var<storage, read_write> globals: Globals;

fn index_array_value() -> f32 {
    let i = 9;
    let a = array<f32, 2>(1.0, 2.0);
    return a[i];
}

fn index_vector_value() -> f32 {
    let i = 9;
    let v = vec2<f32>(1.0, 2.0);
    return v[i];
}

fn index_matrix_value() -> vec2<f32> {
    let i = 9u;
    let m = mat2x2<f32>(1.0, 2.0, 3.0, 4.0);
    return m[i];
}

fn index_negative_value() -> f32 {
    let i = -1;
    let v = vec2<f32>(1.0, 2.0);
    return v[i];
}

fn index_local() -> f32 {
    let i = 9;
    var a = array<f32, 2>(1.0, 2.0);
    a[i] = 3.0;
    return a[i];
}

fn index_storage() -> f32 {
    let i = 9;
    return globals.a[i] + globals.v[i] + globals.m[i][i];
}

fn set_storage(value: f32) {
    let i = 9;
    globals.a[i] = value;
    globals.v[i] = value;
    globals.m[i][i] = value;
}

fn index_runtime_sized_negative() -> f32 {
    let i = -1;
    globals.d[i] = 1.0;
    return globals.d[i];
}

@compute @workgroup_size(1)
fn main() {
    let sum = index_array_value() + index_vector_value() + index_matrix_value().x
        + index_negative_value() + index_local() + index_storage()
        + index_runtime_sized_negative();
    set_storage(sum);
}
