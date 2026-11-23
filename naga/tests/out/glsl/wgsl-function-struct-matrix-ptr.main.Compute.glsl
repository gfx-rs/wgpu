#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct S {
    float a;
    mat3x2 m;
};
layout(std430) buffer type_4_block_0Compute { float _group_0_binding_0_cs[4]; };


void store_column(inout mat3x2 p) {
    p[1] = vec2(5.0, 6.0);
    return;
}

float read_matrix(inout mat3x2 p_1) {
    float _e3 = p_1[2][1];
    return _e3;
}

void main() {
    S s = S(0.0, mat3x2(0.0));
    s.m = mat3x2(vec2(1.0, 2.0), vec2(3.0, 4.0), vec2(5.0, 6.0));
    store_column(s.m);
    float _e18 = s.m[1][1];
    _group_0_binding_0_cs[0] = _e18;
    float _e22 = read_matrix(s.m);
    _group_0_binding_0_cs[1] = _e22;
    return;
}

