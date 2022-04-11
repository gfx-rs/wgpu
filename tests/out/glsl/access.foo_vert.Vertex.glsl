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
    mat4x3 matrix;
    mat2x2 matrix_array[2];
    int atom;
    uvec2 arr[2];
    AlignedWrapper data[];
} _group_0_binding_0_vs;

uniform Baz_block_1Vertex { Baz _group_0_binding_1_vs; };


void test_matrix_within_struct_accesses() {
    int idx = 9;
    mat3x2 unnamed = mat3x2(0.0);
    vec2 unnamed_1 = vec2(0.0);
    vec2 unnamed_2 = vec2(0.0);
    float unnamed_3 = 0.0;
    float unnamed_4 = 0.0;
    float unnamed_5 = 0.0;
    float unnamed_6 = 0.0;
    Baz t = Baz(mat3x2(0.0));
    int _e4 = idx;
    idx = (_e4 - 1);
    mat3x2 _e8 = _group_0_binding_1_vs.m;
    unnamed = _e8;
    vec2 _e13 = _group_0_binding_1_vs.m[0];
    unnamed_1 = _e13;
    int _e16 = idx;
    vec2 _e18 = _group_0_binding_1_vs.m[_e16];
    unnamed_2 = _e18;
    float _e25 = _group_0_binding_1_vs.m[0][1];
    unnamed_3 = _e25;
    int _e30 = idx;
    float _e32 = _group_0_binding_1_vs.m[0][_e30];
    unnamed_4 = _e32;
    int _e35 = idx;
    float _e39 = _group_0_binding_1_vs.m[_e35][1];
    unnamed_5 = _e39;
    int _e42 = idx;
    int _e44 = idx;
    float _e46 = _group_0_binding_1_vs.m[_e42][_e44];
    unnamed_6 = _e46;
    t = Baz(mat3x2(vec2(1.0), vec2(2.0), vec2(3.0)));
    int _e57 = idx;
    idx = (_e57 + 1);
    t.m = mat3x2(vec2(6.0), vec2(5.0), vec2(4.0));
    t.m[0] = vec2(9.0);
    int _e74 = idx;
    t.m[_e74] = vec2(90.0);
    t.m[0][1] = 10.0;
    int _e87 = idx;
    t.m[0][_e87] = 20.0;
    int _e91 = idx;
    t.m[_e91][1] = 30.0;
    int _e97 = idx;
    int _e99 = idx;
    t.m[_e97][_e99] = 40.0;
    return;
}

float read_from_private(inout float foo_1) {
    float _e3 = foo_1;
    return _e3;
}

void main() {
    uint vi = uint(gl_VertexID);
    float foo = 0.0;
    int c[5] = int[5](0, 0, 0, 0, 0);
    float baz_1 = foo;
    foo = 1.0;
    test_matrix_within_struct_accesses();
    mat4x3 matrix = _group_0_binding_0_vs.matrix;
    uvec2 arr[2] = _group_0_binding_0_vs.arr;
    float b = _group_0_binding_0_vs.matrix[3][0];
    int a = _group_0_binding_0_vs.data[(uint(_group_0_binding_0_vs.data.length()) - 2u)].value;
    float _e28 = read_from_private(foo);
    c = int[5](a, int(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    int value = c[vi];
    gl_Position = vec4((matrix * vec4(ivec4(value))), 2.0);
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

