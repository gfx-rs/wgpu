struct AlignedWrapper {
    value: i32,
}

struct Bar {
    _matrix: mat4x3<f32>,
    matrix_array: array<mat2x2<f32>,2>,
    atom: atomic<i32>,
    arr: array<vec2<u32>,2>,
    data: array<AlignedWrapper>,
}

struct Baz {
    m: mat3x2<f32>,
}

@group(0) @binding(0) 
var<storage, read_write> bar: Bar;
@group(0) @binding(1) 
var<uniform> baz: Baz;
@group(0) @binding(2) 
var<storage, read_write> qux: vec2<i32>;
var<workgroup> val: u32;

fn test_matrix_within_struct_accesses() {
    var idx: i32 = 1;
    var t: Baz;

    let _e5 = idx;
    idx = (_e5 - 1);
    _ = baz.m;
    _ = baz.m[0];
    let _e15 = idx;
    _ = baz.m[_e15];
    _ = baz.m[0][1];
    let _e27 = idx;
    _ = baz.m[0][_e27];
    let _e31 = idx;
    _ = baz.m[_e31][1];
    let _e37 = idx;
    let _e39 = idx;
    _ = baz.m[_e37][_e39];
    t = Baz(mat3x2<f32>(vec2<f32>(1.0), vec2<f32>(2.0), vec2<f32>(3.0)));
    let _e51 = idx;
    idx = (_e51 + 1);
    t.m = mat3x2<f32>(vec2<f32>(6.0), vec2<f32>(5.0), vec2<f32>(4.0));
    t.m[0] = vec2<f32>(9.0);
    let _e68 = idx;
    t.m[_e68] = vec2<f32>(90.0);
    t.m[0][1] = 10.0;
    let _e81 = idx;
    t.m[0][_e81] = 20.0;
    let _e85 = idx;
    t.m[_e85][1] = 30.0;
    let _e91 = idx;
    let _e93 = idx;
    t.m[_e91][_e93] = 40.0;
    return;
}

fn read_from_private(foo_1: ptr<function, f32>) -> f32 {
    let _e4 = (*foo_1);
    return _e4;
}

fn test_arr_as_arg(a: array<array<f32,10>,5>) -> f32 {
    return a[4][9];
}

fn assign_through_ptr_fn(p: ptr<workgroup, u32>) {
    (*p) = 42u;
    return;
}

@vertex 
fn foo_vert(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var foo: f32 = 0.0;
    var c: array<i32,5>;

    let baz_1 = foo;
    foo = 1.0;
    test_matrix_within_struct_accesses();
    let _matrix = bar._matrix;
    let arr = bar.arr;
    let b = bar._matrix[3][0];
    let a_1 = bar.data[(arrayLength((&bar.data)) - 2u)].value;
    let c_1 = qux;
    let data_pointer = (&bar.data[0].value);
    let _e30 = read_from_private((&foo));
    c = array<i32,5>(a_1, i32(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    let value = c[vi];
    let _e44 = test_arr_as_arg(array<array<f32,10>,5>(array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
    return vec4<f32>((_matrix * vec4<f32>(vec4<i32>(value))), 2.0);
}

@fragment 
fn foo_frag() -> @location(0) vec4<f32> {
    bar._matrix[1][2] = 1.0;
    bar._matrix = mat4x3<f32>(vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(2.0), vec3<f32>(3.0));
    bar.arr = array<vec2<u32>,2>(vec2<u32>(0u), vec2<u32>(1u));
    bar.data[1].value = 1;
    qux = vec2<i32>(0, 0);
    return vec4<f32>(0.0);
}

@compute @workgroup_size(1, 1, 1) 
fn atomics() {
    var tmp: i32;

    let value_1 = atomicLoad((&bar.atom));
    let _e8 = atomicAdd((&bar.atom), 5);
    tmp = _e8;
    let _e11 = atomicSub((&bar.atom), 5);
    tmp = _e11;
    let _e14 = atomicAnd((&bar.atom), 5);
    tmp = _e14;
    let _e17 = atomicOr((&bar.atom), 5);
    tmp = _e17;
    let _e20 = atomicXor((&bar.atom), 5);
    tmp = _e20;
    let _e23 = atomicMin((&bar.atom), 5);
    tmp = _e23;
    let _e26 = atomicMax((&bar.atom), 5);
    tmp = _e26;
    let _e29 = atomicExchange((&bar.atom), 5);
    tmp = _e29;
    atomicStore((&bar.atom), value_1);
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn assign_through_ptr() {
    assign_through_ptr_fn((&val));
    return;
}
