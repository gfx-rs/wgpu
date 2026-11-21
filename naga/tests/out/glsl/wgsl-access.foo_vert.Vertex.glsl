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
struct Baz_std140_ {
    vec2 m_0_;
    vec2 m_1_;
    vec2 m_2_;
};
struct mat4x2f32_std140_ {
    vec2 c_0_;
    vec2 c_1_;
    vec2 c_2_;
    vec2 c_3_;
};
struct MatCx2InArray_std140_ {
    mat4x2f32_std140_ am[2];
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

layout(std140) uniform Baz_std140_block_1Vertex { Baz_std140_ _group_0_binding_1_vs; };

layout(std430) buffer type_13_block_2Vertex { ivec2 _group_0_binding_2_vs; };

layout(std140) uniform MatCx2InArray_std140_block_3Vertex { MatCx2InArray_std140_ _group_0_binding_3_vs; };


void test_matrix_within_struct_accesses() {
    int idx = 1;
    Baz t = Baz(mat3x2(vec2(1.0), vec2(2.0), vec2(3.0)));
    int _e2 = idx;
    idx = (_e2 - 1);
    vec2 _e6 = _group_0_binding_1_vs.m_0_;
    vec2 _e8 = _group_0_binding_1_vs.m_1_;
    vec2 _e10 = _group_0_binding_1_vs.m_2_;
    mat3x2 l0_ = mat3x2(_e6, _e8, _e10);
    vec2 l1_ = _group_0_binding_1_vs.m_0_;
    int _e15 = idx;
    vec2 _e18 = _group_0_binding_1_vs.m_0_;
    vec2 _e20 = _group_0_binding_1_vs.m_1_;
    vec2 _e22 = _group_0_binding_1_vs.m_2_;
    vec2 l2_ = mat3x2(_e18, _e20, _e22)[_e15];
    float l3_ = _group_0_binding_1_vs.m_0_.y;
    int _e31 = idx;
    float l4_ = _group_0_binding_1_vs.m_0_[_e31];
    int _e34 = idx;
    vec2 _e37 = _group_0_binding_1_vs.m_0_;
    vec2 _e39 = _group_0_binding_1_vs.m_1_;
    vec2 _e41 = _group_0_binding_1_vs.m_2_;
    float l5_ = mat3x2(_e37, _e39, _e41)[_e34].y;
    int _e45 = idx;
    int _e46 = idx;
    vec2 _e49 = _group_0_binding_1_vs.m_0_;
    vec2 _e51 = _group_0_binding_1_vs.m_1_;
    vec2 _e53 = _group_0_binding_1_vs.m_2_;
    float l6_ = mat3x2(_e49, _e51, _e53)[_e45][_e46];
    int _e66 = idx;
    idx = (_e66 + 1);
    t.m = mat3x2(vec2(6.0), vec2(5.0), vec2(4.0));
    t.m[0] = vec2(9.0);
    int _e82 = idx;
    t.m[_e82] = vec2(90.0);
    t.m[0][1] = 10.0;
    int _e91 = idx;
    t.m[0][_e91] = 20.0;
    int _e94 = idx;
    t.m[_e94][1] = 30.0;
    int _e98 = idx;
    int _e100 = idx;
    t.m[_e98][_e100] = 40.0;
    return;
}

mat4x2[2] convert_std140_0_(mat4x2f32_std140_ v[2]) {
    return mat4x2[2](mat4x2(v[0].c_0_, v[0].c_1_, v[0].c_2_, v[0].c_3_), mat4x2(v[1].c_0_, v[1].c_1_, v[1].c_2_, v[1].c_3_));
}

void test_matrix_within_array_within_struct_accesses() {
    int idx_1 = 1;
    MatCx2InArray t_1 = MatCx2InArray(mat4x2[2](mat4x2(0.0), mat4x2(0.0)));
    int _e2 = idx_1;
    idx_1 = (_e2 - 1);
    mat4x2f32_std140_ _e6[2] = _group_0_binding_3_vs.am;
    mat4x2 _e132[2] = convert_std140_0_(_e6);
    vec2 _e11 = _group_0_binding_3_vs.am[0].c_0_;
    vec2 _e13 = _group_0_binding_3_vs.am[0].c_1_;
    vec2 _e15 = _group_0_binding_3_vs.am[0].c_2_;
    vec2 _e17 = _group_0_binding_3_vs.am[0].c_3_;
    mat4x2 l1_1 = mat4x2(_e11, _e13, _e15, _e17);
    vec2 l2_1 = _group_0_binding_3_vs.am[0].c_0_;
    int _e27 = idx_1;
    vec2 _e29 = _group_0_binding_3_vs.am[0].c_0_;
    vec2 _e31 = _group_0_binding_3_vs.am[0].c_1_;
    vec2 _e33 = _group_0_binding_3_vs.am[0].c_2_;
    vec2 _e35 = _group_0_binding_3_vs.am[0].c_3_;
    vec2 l3_1 = mat4x2(_e29, _e31, _e33, _e35)[_e27];
    float l4_1 = _group_0_binding_3_vs.am[0].c_0_.y;
    int _e48 = idx_1;
    float l5_1 = _group_0_binding_3_vs.am[0].c_0_[_e48];
    int _e54 = idx_1;
    vec2 _e56 = _group_0_binding_3_vs.am[0].c_0_;
    vec2 _e58 = _group_0_binding_3_vs.am[0].c_1_;
    vec2 _e60 = _group_0_binding_3_vs.am[0].c_2_;
    vec2 _e62 = _group_0_binding_3_vs.am[0].c_3_;
    float l6_1 = mat4x2(_e56, _e58, _e60, _e62)[_e54].y;
    int _e69 = idx_1;
    int _e70 = idx_1;
    vec2 _e72 = _group_0_binding_3_vs.am[0].c_0_;
    vec2 _e74 = _group_0_binding_3_vs.am[0].c_1_;
    vec2 _e76 = _group_0_binding_3_vs.am[0].c_2_;
    vec2 _e78 = _group_0_binding_3_vs.am[0].c_3_;
    float l7_ = mat4x2(_e72, _e74, _e76, _e78)[_e69][_e70];
    int _e85 = idx_1;
    idx_1 = (_e85 + 1);
    t_1.am = mat4x2[2](mat4x2(0.0), mat4x2(0.0));
    t_1.am[0] = mat4x2(vec2(8.0), vec2(7.0), vec2(6.0), vec2(5.0));
    t_1.am[0][0] = vec2(9.0);
    int _e107 = idx_1;
    t_1.am[0][_e107] = vec2(90.0);
    t_1.am[0][0][1] = 10.0;
    int _e118 = idx_1;
    t_1.am[0][0][_e118] = 20.0;
    int _e122 = idx_1;
    t_1.am[0][_e122][1] = 30.0;
    int _e127 = idx_1;
    int _e129 = idx_1;
    t_1.am[0][_e127][_e129] = 40.0;
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

