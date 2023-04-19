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
    matrix_array: array<mat2x2<f32>, 2>,
    atom: atomic<i32>,
    atom_arr: array<atomic<i32>, 10>,
    arr: array<vec2<u32>, 2>,
    data: array<AlignedWrapper>,
}

struct Baz {
    m: mat3x2<f32>,
}

struct MatCx2InArray {
    am: array<mat4x2<f32>, 2>,
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
    var t: Baz = Baz(mat3x2<f32>(vec2(1.0), vec2(2.0), vec2(3.0)));

    let _e3 = idx;
    idx = (_e3 - 1);
    let l0_ = baz.m;
    let l1_ = baz.m[0];
    let _e14 = idx;
    let l2_ = baz.m[_e14];
    let l3_ = baz.m[0][1];
    let _e25 = idx;
    let l4_ = baz.m[0][_e25];
    let _e30 = idx;
    let l5_ = baz.m[_e30][1];
    let _e36 = idx;
    let _e38 = idx;
    let l6_ = baz.m[_e36][_e38];
    let _e51 = idx;
    idx = (_e51 + 1);
    t.m = mat3x2<f32>(vec2(6.0), vec2(5.0), vec2(4.0));
    t.m[0] = vec2(9.0);
    let _e66 = idx;
    t.m[_e66] = vec2(90.0);
    t.m[0][1] = 10.0;
    let _e76 = idx;
    t.m[0][_e76] = 20.0;
    let _e80 = idx;
    t.m[_e80][1] = 30.0;
    let _e85 = idx;
    let _e87 = idx;
    t.m[_e85][_e87] = 40.0;
    return;
}

fn test_matrix_within_array_within_struct_accesses() {
    var idx_1: i32 = 1;
    var t_1: MatCx2InArray = MatCx2InArray(array<mat4x2<f32>, 2>());

    let _e3 = idx_1;
    idx_1 = (_e3 - 1);
    let l0_1 = nested_mat_cx2_.am;
    let l1_1 = nested_mat_cx2_.am[0];
    let l2_1 = nested_mat_cx2_.am[0][0];
    let _e20 = idx_1;
    let l3_1 = nested_mat_cx2_.am[0][_e20];
    let l4_1 = nested_mat_cx2_.am[0][0][1];
    let _e33 = idx_1;
    let l5_1 = nested_mat_cx2_.am[0][0][_e33];
    let _e39 = idx_1;
    let l6_1 = nested_mat_cx2_.am[0][_e39][1];
    let _e46 = idx_1;
    let _e48 = idx_1;
    let l7_ = nested_mat_cx2_.am[0][_e46][_e48];
    let _e55 = idx_1;
    idx_1 = (_e55 + 1);
    t_1.am = array<mat4x2<f32>, 2>();
    t_1.am[0] = mat4x2<f32>(vec2(8.0), vec2(7.0), vec2(6.0), vec2(5.0));
    t_1.am[0][0] = vec2(9.0);
    let _e77 = idx_1;
    t_1.am[0][_e77] = vec2(90.0);
    t_1.am[0][0][1] = 10.0;
    let _e89 = idx_1;
    t_1.am[0][0][_e89] = 20.0;
    let _e94 = idx_1;
    t_1.am[0][_e94][1] = 30.0;
    let _e100 = idx_1;
    let _e102 = idx_1;
    t_1.am[0][_e100][_e102] = 40.0;
    return;
}

fn read_from_private(foo_1: ptr<function, f32>) -> f32 {
    let _e1 = (*foo_1);
    return _e1;
}

fn test_arr_as_arg(a: array<array<f32, 10>, 5>) -> f32 {
    return a[4][9];
}

fn assign_through_ptr_fn(p: ptr<workgroup, u32>) {
    (*p) = 42u;
    return;
}

fn assign_array_through_ptr_fn(foo_2: ptr<function, array<vec4<f32>, 2>>) {
    (*foo_2) = array<vec4<f32>, 2>(vec4(1.0), vec4(2.0));
    return;
}

@vertex 
fn foo_vert(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var foo: f32 = 0.0;
    var c2_: array<i32, 5>;

    let baz_1 = foo;
    foo = 1.0;
    test_matrix_within_struct_accesses();
    test_matrix_within_array_within_struct_accesses();
    let _matrix = bar._matrix;
    let arr_1 = bar.arr;
    let b = bar._matrix[3][0];
    let a_1 = bar.data[(arrayLength((&bar.data)) - 2u)].value;
    let c = qux;
    let data_pointer = (&bar.data[0].value);
    let _e33 = read_from_private((&foo));
    c2_ = array<i32, 5>(a_1, i32(b), 3, 4, 5);
    c2_[(vi + 1u)] = 42;
    let value = c2_[vi];
    let _e47 = test_arr_as_arg(array<array<f32, 10>, 5>());
    return vec4<f32>((_matrix * vec4<f32>(vec4(value))), 2.0);
}

@fragment 
fn foo_frag() -> @location(0) vec4<f32> {
    bar._matrix[1][2] = 1.0;
    bar._matrix = mat4x3<f32>(vec3(0.0), vec3(1.0), vec3(2.0), vec3(3.0));
    bar.arr = array<vec2<u32>, 2>(vec2(0u), vec2(1u));
    bar.data[1].value = 1;
    qux = vec2<i32>();
    return vec4(0.0);
}

@compute @workgroup_size(1, 1, 1) 
fn assign_through_ptr() {
    var arr: array<vec4<f32>, 2> = array<vec4<f32>, 2>(vec4(6.0), vec4(7.0));

    assign_through_ptr_fn((&val));
    assign_array_through_ptr_fn((&arr));
    return;
}
