#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct S {
    float a;
    mat3x2 m;
};
layout(std430) buffer type_2_block_0Compute { float _group_0_binding_0_cs[4]; };


void main() {
    S s = S(0.0, mat3x2(0.0));
    s.m = mat3x2(vec2(1.0, 2.0), vec2(3.0, 4.0), vec2(5.0, 6.0));
    float _e17 = s.m[0][0];
    _group_0_binding_0_cs[0] = _e17;
    float _e23 = s.m[2][1];
    _group_0_binding_0_cs[1] = _e23;
    return;
}

