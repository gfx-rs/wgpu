#version 310 es

precision highp float;
precision highp int;

struct GlobalConst {
    uint a;
    uvec3 b;
    int c;
};
struct AlignedWrapper {
    int value;
};
struct Baz {
    mat3x2 m;
};
struct MatCx2InArray {
    mat4x2 am[2];
};
struct AssignToMember {
    uint x;
};
struct S {
    int m;
};
struct Inner {
    int delicious;
};
struct Outer {
    Inner om_nom_nom;
    uint thing;
};
GlobalConst msl_padding_global_const = GlobalConst(0u, uvec3(0u, 0u, 0u), 0);

layout(std430) buffer Bar_block_0Vertex {
    mat4x3 _matrix;
    mat2x2 matrix_array[2];
    int atom;
    int atom_arr[10];
    uvec2 arr[2];
    AlignedWrapper data[];
} _group_0_binding_0_vs;

layout(std140) uniform Baz_block_1Vertex { Baz _group_0_binding_1_vs; };

layout(std430) buffer type_13_block_2Vertex { ivec2 _group_0_binding_2_vs; };

layout(std140) uniform MatCx2InArray_block_3Vertex { MatCx2InArray _group_0_binding_3_vs; };


void test_matrix_within_struct_accesses() {
    int idx = 1;
    Baz t = Baz(mat3x2(vec2(1.0), vec2(2.0), vec2(3.0)));
    int _e3 = idx;
    idx = (_e3 - 1);
    mat3x2 l0_ = _group_0_binding_1_vs.m;
    vec2 l1_ = _group_0_binding_1_vs.m[0];
    int _e14 = idx;
    vec2 l2_ = _group_0_binding_1_vs.m[_e14];
    float l3_ = _group_0_binding_1_vs.m[0][1];
    int _e25 = idx;
    float l4_ = _group_0_binding_1_vs.m[0][_e25];
    int _e30 = idx;
    float l5_ = _group_0_binding_1_vs.m[_e30][1];
    int _e36 = idx;
    int _e38 = idx;
    float l6_ = _group_0_binding_1_vs.m[_e36][_e38];
    int _e51 = idx;
    idx = (_e51 + 1);
    t.m = mat3x2(vec2(6.0), vec2(5.0), vec2(4.0));
    t.m[0] = vec2(9.0);
    int _e66 = idx;
    t.m[_e66] = vec2(90.0);
    t.m[0][1] = 10.0;
    int _e76 = idx;
    t.m[0][_e76] = 20.0;
    int _e80 = idx;
    t.m[_e80][1] = 30.0;
    int _e85 = idx;
    int _e87 = idx;
    t.m[_e85][_e87] = 40.0;
    return;
}

void test_matrix_within_array_within_struct_accesses() {
    int idx_1 = 1;
    MatCx2InArray t_1 = MatCx2InArray(mat4x2[2](mat4x2(0.0), mat4x2(0.0)));
    int _e3 = idx_1;
    idx_1 = (_e3 - 1);
    mat4x2 l0_1[2] = _group_0_binding_3_vs.am;
    mat4x2 l1_1 = _group_0_binding_3_vs.am[0];
    vec2 l2_1 = _group_0_binding_3_vs.am[0][0];
    int _e20 = idx_1;
    vec2 l3_1 = _group_0_binding_3_vs.am[0][_e20];
    float l4_1 = _group_0_binding_3_vs.am[0][0][1];
    int _e33 = idx_1;
    float l5_1 = _group_0_binding_3_vs.am[0][0][_e33];
    int _e39 = idx_1;
    float l6_1 = _group_0_binding_3_vs.am[0][_e39][1];
    int _e46 = idx_1;
    int _e48 = idx_1;
    float l7_ = _group_0_binding_3_vs.am[0][_e46][_e48];
    int _e55 = idx_1;
    idx_1 = (_e55 + 1);
    t_1.am = mat4x2[2](mat4x2(0.0), mat4x2(0.0));
    t_1.am[0] = mat4x2(vec2(8.0), vec2(7.0), vec2(6.0), vec2(5.0));
    t_1.am[0][0] = vec2(9.0);
    int _e77 = idx_1;
    t_1.am[0][_e77] = vec2(90.0);
    t_1.am[0][0][1] = 10.0;
    int _e89 = idx_1;
    t_1.am[0][0][_e89] = 20.0;
    int _e94 = idx_1;
    t_1.am[0][_e94][1] = 30.0;
    int _e100 = idx_1;
    int _e102 = idx_1;
    t_1.am[0][_e100][_e102] = 40.0;
    return;
}

float read_from_private(inout float foo_1) {
    float _e1 = foo_1;
    return _e1;
}

float test_arr_as_arg(float a[5][10]) {
    return a[4][9];
}

void assign_through_ptr_fn(inout uint p) {
    p = 42u;
    return;
}

void assign_array_through_ptr_fn(inout vec4 foo_2[2]) {
    foo_2 = vec4[2](vec4(1.0), vec4(2.0));
    return;
}

void assign_through_ptr() {
    uint val = 33u;
    vec4 arr[2] = vec4[2](vec4(6.0), vec4(7.0));
    assign_through_ptr_fn(val);
    assign_array_through_ptr_fn(arr);
    return;
}

uint fetch_arg_ptr_member(inout AssignToMember p_1) {
    uint _e2 = p_1.x;
    return _e2;
}

void assign_to_arg_ptr_member(inout AssignToMember p_2) {
    p_2.x = 10u;
    return;
}

uint fetch_arg_ptr_array_element(inout uint p_3[4]) {
    uint _e2 = p_3[1];
    return _e2;
}

void assign_to_arg_ptr_array_element(inout uint p_4[4]) {
    p_4[1] = 10u;
    return;
}

void assign_to_ptr_components() {
    AssignToMember s1_ = AssignToMember(0u);
    uint a1_[4] = uint[4](0u, 0u, 0u, 0u);
    assign_to_arg_ptr_member(s1_);
    uint _e1 = fetch_arg_ptr_member(s1_);
    assign_to_arg_ptr_array_element(a1_);
    uint _e3 = fetch_arg_ptr_array_element(a1_);
    return;
}

bool index_ptr(bool value) {
    bool a_1[1] = bool[1](false);
    a_1 = bool[1](value);
    bool _e4 = a_1[0];
    return _e4;
}

int member_ptr() {
    S s = S(42);
    int _e4 = s.m;
    return _e4;
}

int let_members_of_members() {
    Inner inner_1 = Outer(Inner(0), 0u).om_nom_nom;
    int delishus_1 = inner_1.delicious;
    if ((Outer(Inner(0), 0u).thing != uint(delishus_1))) {
    }
    return Outer(Inner(0), 0u).om_nom_nom.delicious;
}

int var_members_of_members() {
    Outer thing = Outer(Inner(0), 0u);
    Inner inner = Inner(0);
    int delishus = 0;
    Inner _e3 = thing.om_nom_nom;
    inner = _e3;
    int _e6 = inner.delicious;
    delishus = _e6;
    uint _e9 = thing.thing;
    int _e10 = delishus;
    if ((_e9 != uint(_e10))) {
    }
    int _e15 = thing.om_nom_nom.delicious;
    return _e15;
}

void main() {
    uint vi = uint(gl_VertexID);
    float foo = 0.0;
    int c2_[5] = int[5](0, 0, 0, 0, 0);
    float baz_1 = foo;
    foo = 1.0;
    GlobalConst phony = msl_padding_global_const;
    test_matrix_within_struct_accesses();
    test_matrix_within_array_within_struct_accesses();
    mat4x3 _matrix = _group_0_binding_0_vs._matrix;
    uvec2 arr_1[2] = _group_0_binding_0_vs.arr;
    float b = _group_0_binding_0_vs._matrix[3u][0];
    int a_2 = _group_0_binding_0_vs.data[(uint(_group_0_binding_0_vs.data.length()) - 2u)].value;
    ivec2 c = _group_0_binding_2_vs;
    float _e35 = read_from_private(foo);
    c2_ = int[5](a_2, int(b), 3, 4, 5);
    c2_[(vi + 1u)] = 42;
    int value_1 = c2_[vi];
    float _e49 = test_arr_as_arg(float[5][10](float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
    gl_Position = vec4((_matrix * vec4(ivec4(value_1))), 2.0);
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

