#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(r32f) writeonly uniform highp image2D _group_1_binding_0_cs;

layout(rg32f) writeonly uniform highp image2D _group_1_binding_1_cs;

layout(rgba32f) writeonly uniform highp image2D _group_1_binding_2_cs;


void main() {
    imageStore(_group_1_binding_0_cs, ivec2(uvec2(0u)), vec4(0.0));
    imageStore(_group_1_binding_1_cs, ivec2(uvec2(0u)), vec4(0.0));
    imageStore(_group_1_binding_2_cs, ivec2(uvec2(0u)), vec4(0.0));
    return;
}

