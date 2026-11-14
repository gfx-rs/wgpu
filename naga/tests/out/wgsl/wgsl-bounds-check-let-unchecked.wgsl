struct Globals {
    a: array<f32, 4>,
    v: vec4<f32>,
    m: mat3x4<f32>,
    d: array<f32>,
}

@group(0) @binding(0)
var<storage, read_write> globals: Globals;

fn index_array_value() -> f32 {
    let a_1 = array<f32, 2>(1f, 2f);
    return a_1[1u];
}

fn index_vector_value() -> f32 {
    let v = vec2<f32>(1f, 2f);
    return v[1u];
}

fn index_matrix_value() -> vec2<f32> {
    let m = mat2x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f));
    return m[1u];
}

fn index_negative_value() -> f32 {
    let v_1 = vec2<f32>(1f, 2f);
    return v_1[1u];
}

fn index_local() -> f32 {
    var a: array<f32, 2> = array<f32, 2>(1f, 2f);

    a[1u] = 3f;
    let _e8 = a[1u];
    return _e8;
}

fn index_storage() -> f32 {
    let _e4 = globals.a[3u];
    let _e8 = globals.v[3u];
    let _e14 = globals.m[2u][3u];
    return ((_e4 + _e8) + _e14);
}

fn set_storage(value: f32) {
    globals.a[3u] = value;
    globals.v[3u] = value;
    globals.m[2u][3u] = value;
    return;
}

fn index_runtime_sized_negative() -> f32 {
    globals.d[0u] = 1f;
    let _e8 = globals.d[0u];
    return _e8;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let _e0 = index_array_value();
    let _e1 = index_vector_value();
    let _e3 = index_matrix_value();
    let _e6 = index_negative_value();
    let _e8 = index_local();
    let _e10 = index_storage();
    let _e12 = index_runtime_sized_negative();
    let sum = ((((((_e0 + _e1) + _e3.x) + _e6) + _e8) + _e10) + _e12);
    set_storage(sum);
    return;
}
