#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(r32f) readonly uniform highp image2D _group_0_binding_0_cs;

layout(rg32f) readonly uniform highp image2D _group_0_binding_1_cs;

layout(rgba32f) readonly uniform highp image2D _group_0_binding_2_cs;


void main() {
    vec4 phony = imageLoad(_group_0_binding_0_cs, ivec2(uvec2(0u)));
    vec4 phony_1 = imageLoad(_group_0_binding_1_cs, ivec2(uvec2(0u)));
    vec4 phony_2 = imageLoad(_group_0_binding_2_cs, ivec2(uvec2(0u)));
}

