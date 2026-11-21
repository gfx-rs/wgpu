#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Padded {
    uvec3 xyz;
    uint tail;
};
const uint W = 64u;
const uint N = 513u;

layout(std430) buffer type_6_block_0Compute { uint _group_0_binding_0_cs[]; };


void main() {
    uvec3 group_2 = gl_WorkGroupID;
    uint base = (group_2.x * 4u);
    _group_0_binding_0_cs[base] = group_2.x;
    _group_0_binding_0_cs[(base + 1u)] = 0u;
    _group_0_binding_0_cs[(base + 2u)] = 0u;
    _group_0_binding_0_cs[(base + 3u)] = 0u;
    return;
}

