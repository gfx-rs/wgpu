#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) buffer type_block_0Compute { ivec4 _group_0_binding_0_cs; };

layout(std430) buffer type_1_block_1Compute { int _group_0_binding_1_cs; };

layout(std430) buffer type_1_block_2Compute { int _group_0_binding_2_cs; };


void main() {
    ivec2 a = ivec2(1, 2);
    ivec2 b = ivec2(3, 4);
    _group_0_binding_0_cs = ivec4(4, 3, 2, 1);
    _group_0_binding_1_cs = 2;
    _group_0_binding_2_cs = 6;
    return;
}

