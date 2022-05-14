#version 310 es

precision highp float;
precision highp int;

struct AlignedWrapper {
    int value;
};
struct Baz {
    mat3x2 m;
};
layout(std430) buffer Bar_block_0Vertex {
    mat4x3 _matrix;
    mat2x2 matrix_array[2];
    int atom;
    uvec2 arr[2];
    AlignedWrapper data[];
} _group_0_binding_0_vs;

uniform Baz_block_1Vertex { Baz _group_0_binding_1_vs; };

layout(std430) buffer type_9_block_2Vertex { ivec2 _group_0_binding_2_vs; };


void test_matrix_within_struct_accesses() {
    int idx = 1;
    Baz t = Baz(mat3x2(0.0));
    int _e5 = idx;
    idx = (_e5 - 1);
    mat3x2 unnamed = _group_0_binding_1_vs.m;
    vec2 unnamed_1 = _group_0_binding_1_vs.m[0];
    int _e15 = idx;
    vec2 unnamed_2 = _group_0_binding_1_vs.m[_e15];
    float unnamed_3 = _group_0_binding_1_vs.m[0][1];
    int _e27 = idx;
    float unnamed_4 = _group_0_binding_1_vs.m[0][_e27];
    int _e31 = idx;
    float unnamed_5 = _group_0_binding_1_vs.m[_e31][1];
    int _e37 = idx;
    int _e39 = idx;
    float unnamed_6 = _group_0_binding_1_vs.m[_e37][_e39];
    t = Baz(mat3x2(vec2(1.0), vec2(2.0), vec2(3.0)));
    int _e51 = idx;
    idx = (_e51 + 1);
    t.m = mat3x2(vec2(6.0), vec2(5.0), vec2(4.0));
    t.m[0] = vec2(9.0);
    int _e68 = idx;
    t.m[_e68] = vec2(90.0);
    t.m[0][1] = 10.0;
    int _e81 = idx;
    t.m[0][_e81] = 20.0;
    int _e85 = idx;
    t.m[_e85][1] = 30.0;
    int _e91 = idx;
    int _e93 = idx;
    t.m[_e91][_e93] = 40.0;
    return;
}

float read_from_private(inout float foo_1) {
    float _e4 = foo_1;
    return _e4;
}

float test_arr_as_arg(float a[5][10]) {
    return a[4][9];
}

void assign_through_ptr_fn(inout uint p) {
    p = 42u;
    return;
}

void main() {
    uint vi = uint(gl_VertexID);
    float foo = 0.0;
    int c[5] = int[5](0, 0, 0, 0, 0);
    float baz_1 = foo;
    foo = 1.0;
    test_matrix_within_struct_accesses();
    mat4x3 _matrix = _group_0_binding_0_vs._matrix;
    uvec2 arr[2] = _group_0_binding_0_vs.arr;
    float b = _group_0_binding_0_vs._matrix[3][0];
    int a_1 = _group_0_binding_0_vs.data[(uint(_group_0_binding_0_vs.data.length()) - 2u)].value;
    ivec2 c_1 = _group_0_binding_2_vs;
    float _e30 = read_from_private(foo);
    c = int[5](a_1, int(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    int value = c[vi];
    float _e44 = test_arr_as_arg(float[5][10](float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
    gl_Position = vec4((_matrix * vec4(ivec4(value))), 2.0);
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

