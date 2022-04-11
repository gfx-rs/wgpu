struct AlignedWrapper {
    value: i32,
}

struct Bar {
    matrix: mat4x3<f32>,
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
    var unnamed: mat3x2<f32>;
    var unnamed_1: vec2<f32>;
    var unnamed_2: vec2<f32>;
    var unnamed_3: f32;
    var unnamed_4: f32;
    var unnamed_5: f32;
    var unnamed_6: f32;
    var t: Baz;

    let _e4 = idx;
    idx = (_e4 - 1);
    let _e8 = baz.m;
    unnamed = _e8;
    let _e13 = baz.m[0];
    unnamed_1 = _e13;
    let _e16 = idx;
    let _e18 = baz.m[_e16];
    unnamed_2 = _e18;
    let _e25 = baz.m[0][1];
    unnamed_3 = _e25;
    let _e30 = idx;
    let _e32 = baz.m[0][_e30];
    unnamed_4 = _e32;
    let _e35 = idx;
    let _e39 = baz.m[_e35][1];
    unnamed_5 = _e39;
    let _e42 = idx;
    let _e44 = idx;
    let _e46 = baz.m[_e42][_e44];
    unnamed_6 = _e46;
    t = Baz(mat3x2<f32>(vec2<f32>(1.0), vec2<f32>(2.0), vec2<f32>(3.0)));
    let _e57 = idx;
    idx = (_e57 + 1);
    t.m = mat3x2<f32>(vec2<f32>(6.0), vec2<f32>(5.0), vec2<f32>(4.0));
    t.m[0] = vec2<f32>(9.0);
    let _e74 = idx;
    t.m[_e74] = vec2<f32>(90.0);
    t.m[0][1] = 10.0;
    let _e87 = idx;
    t.m[0][_e87] = 20.0;
    let _e91 = idx;
    t.m[_e91][1] = 30.0;
    let _e97 = idx;
    let _e99 = idx;
    t.m[_e97][_e99] = 40.0;
    return;
}

fn read_from_private(foo_1: ptr<function, f32>) -> f32 {
    let _e3 = (*foo_1);
    return _e3;
}

@stage(vertex) 
fn foo_vert(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var foo: f32 = 0.0;
    var c: array<i32,5>;

    let baz_1 = foo;
    foo = 1.0;
    test_matrix_within_struct_accesses();
    let matrix = bar.matrix;
    let arr = bar.arr;
    let b = bar.matrix[3][0];
    let a = bar.data[(arrayLength((&bar.data)) - 2u)].value;
    let data_pointer = (&bar.data[0].value);
    let _e28 = read_from_private((&foo));
    c = array<i32,5>(a, i32(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    let value = c[vi];
    return vec4<f32>((matrix * vec4<f32>(vec4<i32>(value))), 2.0);
}

@stage(fragment) 
fn foo_frag() -> @location(0) vec4<f32> {
    bar.matrix[1][2] = 1.0;
    bar.matrix = mat4x3<f32>(vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(2.0), vec3<f32>(3.0));
    bar.arr = array<vec2<u32>,2>(vec2<u32>(0u), vec2<u32>(1u));
    bar.data[1].value = 1;
    return vec4<f32>(0.0);
}

@stage(compute) @workgroup_size(1, 1, 1) 
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
