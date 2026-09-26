#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const uvec4 composed = uvec4(uvec2(0u, 0u), 7u, 9u);

layout(std430) buffer type_3_block_0Compute { uint _group_0_binding_0_cs[8]; };

layout(std430) buffer type_block_1Compute { uvec4 _group_0_binding_1_cs; };


void main() {
    _group_0_binding_0_cs[0] = 0u;
    _group_0_binding_0_cs[1] = 7u;
    _group_0_binding_0_cs[2] = 0u;
    _group_0_binding_0_cs[3] = floatBitsToUint(0.0);
    _group_0_binding_0_cs[4] = floatBitsToUint(1.0);
    _group_0_binding_0_cs[5] = 0u;
    _group_0_binding_0_cs[6] = 9u;
    _group_0_binding_0_cs[7] = 7u;
    _group_0_binding_1_cs = composed;
    return;
}

