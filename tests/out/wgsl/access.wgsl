struct GlobalConst {
    a: u32,
    b: vec3<u32>,
    c: i32,
}

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

struct MatCx2InArray {
    am: array<mat4x2<f32>,2>,
}

var<private> global_const: GlobalConst = GlobalConst(0u, vec3<u32>(0u, 0u, 0u), 0);
@group(0) @binding(0) 
var<storage, read_write> bar: Bar;
@group(0) @binding(1) 
var<uniform> baz: Baz;
@group(0) @binding(2) 
var<storage, read_write> qux: vec2<i32>;
@group(0) @binding(3) 
var<uniform> nested_mat_cx2_: MatCx2InArray;
var<workgroup> val: u32;

fn test_matrix_within_struct_accesses() {
    var idx: i32 = 1;
    var t: Baz;

    let _e6 = idx;
    idx = (_e6 - 1);
    _ = baz.m;
    _ = baz.m[0];
    let _e16 = idx;
    _ = baz.m[_e16];
    _ = baz.m[0][1];
    let _e28 = idx;
    _ = baz.m[0][_e28];
    let _e32 = idx;
    _ = baz.m[_e32][1];
    let _e38 = idx;
    let _e40 = idx;
    _ = baz.m[_e38][_e40];
    t = Baz(mat3x2<f32>(vec2<f32>(1.0), vec2<f32>(2.0), vec2<f32>(3.0)));
    let _e52 = idx;
    idx = (_e52 + 1);
    t.m = mat3x2<f32>(vec2<f32>(6.0), vec2<f32>(5.0), vec2<f32>(4.0));
    t.m[0] = vec2<f32>(9.0);
    let _e69 = idx;
    t.m[_e69] = vec2<f32>(90.0);
    t.m[0][1] = 10.0;
    let _e82 = idx;
    t.m[0][_e82] = 20.0;
    let _e86 = idx;
    t.m[_e86][1] = 30.0;
    let _e92 = idx;
    let _e94 = idx;
    t.m[_e92][_e94] = 40.0;
    return;
}

fn test_matrix_within_array_within_struct_accesses() {
    var idx_1: i32 = 1;
    var t_1: MatCx2InArray;

    let _e7 = idx_1;
    idx_1 = (_e7 - 1);
    _ = nested_mat_cx2_.am;
    _ = nested_mat_cx2_.am[0];
    _ = nested_mat_cx2_.am[0][0];
    let _e25 = idx_1;
    _ = nested_mat_cx2_.am[0][_e25];
    _ = nested_mat_cx2_.am[0][0][1];
    let _e41 = idx_1;
    _ = nested_mat_cx2_.am[0][0][_e41];
    let _e47 = idx_1;
    _ = nested_mat_cx2_.am[0][_e47][1];
    let _e55 = idx_1;
    let _e57 = idx_1;
    _ = nested_mat_cx2_.am[0][_e55][_e57];
    t_1 = MatCx2InArray(array<mat4x2<f32>,2>(mat4x2<f32>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0)), mat4x2<f32>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0))));
    let _e63 = idx_1;
    idx_1 = (_e63 + 1);
    t_1.am = array<mat4x2<f32>,2>(mat4x2<f32>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0)), mat4x2<f32>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0)));
    t_1.am[0] = mat4x2<f32>(vec2<f32>(8.0), vec2<f32>(7.0), vec2<f32>(6.0), vec2<f32>(5.0));
    t_1.am[0][0] = vec2<f32>(9.0);
    let _e90 = idx_1;
    t_1.am[0][_e90] = vec2<f32>(90.0);
    t_1.am[0][0][1] = 10.0;
    let _e107 = idx_1;
    t_1.am[0][0][_e107] = 20.0;
    let _e113 = idx_1;
    t_1.am[0][_e113][1] = 30.0;
    let _e121 = idx_1;
    let _e123 = idx_1;
    t_1.am[0][_e121][_e123] = 40.0;
    return;
}

fn read_from_private(foo_1: ptr<function, f32>) -> f32 {
    let _e6 = (*foo_1);
    return _e6;
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
    test_matrix_within_array_within_struct_accesses();
    let _matrix = bar._matrix;
    let arr = bar.arr;
    let b = bar._matrix[3][0];
    let a_1 = bar.data[(arrayLength((&bar.data)) - 2u)].value;
    let c_1 = qux;
    let data_pointer = (&bar.data[0].value);
    let _e32 = read_from_private((&foo));
    c = array<i32,5>(a_1, i32(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    let value = c[vi];
    let _e46 = test_arr_as_arg(array<array<f32,10>,5>(array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), array<f32,10>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
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
    let _e10 = atomicAdd((&bar.atom), 5);
    tmp = _e10;
    let _e13 = atomicSub((&bar.atom), 5);
    tmp = _e13;
    let _e16 = atomicAnd((&bar.atom), 5);
    tmp = _e16;
    let _e19 = atomicOr((&bar.atom), 5);
    tmp = _e19;
    let _e22 = atomicXor((&bar.atom), 5);
    tmp = _e22;
    let _e25 = atomicMin((&bar.atom), 5);
    tmp = _e25;
    let _e28 = atomicMax((&bar.atom), 5);
    tmp = _e28;
    let _e31 = atomicExchange((&bar.atom), 5);
    tmp = _e31;
    atomicStore((&bar.atom), value_1);
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn assign_through_ptr() {
    assign_through_ptr_fn((&val));
    return;
}
