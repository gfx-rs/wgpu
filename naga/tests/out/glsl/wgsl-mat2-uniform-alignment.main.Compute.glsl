#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct S {
    int a;
    mat2x2 b;
};
struct S_std140_ {
    int a;
    vec2 b_0_;
    vec2 b_1_;
};
layout(std140) uniform S_std140_block_0Compute { S_std140_ _group_0_binding_0_cs; };


void main() {
    S_std140_ _e1 = _group_0_binding_0_cs;
    S v = S(_e1.a, mat2x2(_e1.b_0_, _e1.b_1_));
    return;
}

