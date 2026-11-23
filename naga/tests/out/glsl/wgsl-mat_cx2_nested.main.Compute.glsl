#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Inner {
    mat2x2 m1_;
    mat4x2 m2_;
};
struct Outer {
    vec4 a;
    Inner i;
};
struct Inner_std140_ {
    vec2 m1_0_;
    vec2 m1_1_;
    vec2 m2_0_;
    vec2 m2_1_;
    vec2 m2_2_;
    vec2 m2_3_;
};
struct Outer_std140_ {
    vec4 a;
    Inner_std140_ i;
};
layout(std140) uniform Inner_std140_block_0Compute { Inner_std140_ _group_0_binding_0_cs; };

layout(std140) uniform Outer_std140_block_1Compute { Outer_std140_ _group_0_binding_1_cs; };

layout(std430) buffer type_4_block_2Compute { float _group_0_binding_2_cs[16]; };


void main() {
    vec2 _e2 = _group_0_binding_0_cs.m1_0_;
    vec2 _e4 = _group_0_binding_0_cs.m1_1_;
    mat2x2 k = mat2x2(_e2, _e4);
    vec2 _e9 = _group_0_binding_1_cs.i.m2_0_;
    vec2 _e11 = _group_0_binding_1_cs.i.m2_1_;
    vec2 _e13 = _group_0_binding_1_cs.i.m2_2_;
    vec2 _e15 = _group_0_binding_1_cs.i.m2_3_;
    mat4x2 m = mat4x2(_e9, _e11, _e13, _e15);
    _group_0_binding_2_cs[0] = ((((m[0].x + m[1].x) + m[2].x) + m[3].x) + k[0].x);
    return;
}

