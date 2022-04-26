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

fn test_matrix_within_struct_accesses() {
    var idx: i32 = 9;
    var t: Baz;

    let _e4 = idx;
    idx = (_e4 - 1);
    let unnamed = baz.m;
    let unnamed_1 = baz.m[0];
    let _e14 = idx;
    let unnamed_2 = baz.m[_e14];
    let unnamed_3 = baz.m[0][1];
    let _e26 = idx;
    let unnamed_4 = baz.m[0][_e26];
    let _e30 = idx;
    let unnamed_5 = baz.m[_e30][1];
    let _e36 = idx;
    let _e38 = idx;
    let unnamed_6 = baz.m[_e36][_e38];
    t = Baz(mat3x2<f32>(vec2<f32>(1.0), vec2<f32>(2.0), vec2<f32>(3.0)));
    let _e50 = idx;
    idx = (_e50 + 1);
    t.m = mat3x2<f32>(vec2<f32>(6.0), vec2<f32>(5.0), vec2<f32>(4.0));
    t.m[0] = vec2<f32>(9.0);
    let _e67 = idx;
    t.m[_e67] = vec2<f32>(90.0);
    t.m[0][1] = 10.0;
    let _e80 = idx;
    t.m[0][_e80] = 20.0;
    let _e84 = idx;
    t.m[_e84][1] = 30.0;
    let _e90 = idx;
    let _e92 = idx;
    t.m[_e90][_e92] = 40.0;
    return;
}

fn read_from_private(foo_1: ptr<function, f32>) -> f32 {
    let _e3 = (*foo_1);
    return _e3;
}

fn test_arr_as_arg(a: array<array<f32,10>,5>) -> f32 {
    return a[4][9];
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
    let data_pointer = (&bar.data[0].value);
    let _e28 = read_from_private((&foo));
    c = array<i32,5>(a_1, i32(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    let value = c[vi];
    let _e42 = test_arr_as_arg(array<array<f32,10>,5>(array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
    return vec4<f32>((_matrix * vec4<f32>(vec4<i32>(value))), 2.0);
}

@fragment 
fn foo_frag() -> @location(0) vec4<f32> {
    bar._matrix[1][2] = 1.0;
    bar._matrix = mat4x3<f32>(vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(2.0), vec3<f32>(3.0));
    bar.arr = array<vec2<u32>,2>(vec2<u32>(0u), vec2<u32>(1u));
    bar.data[1].value = 1;
    return vec4<f32>(0.0);
}

@compute @workgroup_size(1, 1, 1) 
fn atomics() {
    var tmp: i32;

    let value_1 = atomicLoad((&bar.atom));
    let _e7 = atomicAdd((&bar.atom), 5);
    tmp = _e7;
    let _e10 = atomicSub((&bar.atom), 5);
    tmp = _e10;
    let _e13 = atomicAnd((&bar.atom), 5);
    tmp = _e13;
    let _e16 = atomicOr((&bar.atom), 5);
    tmp = _e16;
    let _e19 = atomicXor((&bar.atom), 5);
    tmp = _e19;
    let _e22 = atomicMin((&bar.atom), 5);
    tmp = _e22;
    let _e25 = atomicMax((&bar.atom), 5);
    tmp = _e25;
    let _e28 = atomicExchange((&bar.atom), 5);
    tmp = _e28;
    atomicStore((&bar.atom), value_1);
    return;
}
