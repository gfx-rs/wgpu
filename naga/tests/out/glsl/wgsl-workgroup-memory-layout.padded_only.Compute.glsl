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

shared Padded padded;

layout(std430) buffer type_6_block_0Compute { uint _group_0_binding_0_cs[]; };


void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        padded = Padded(uvec3(0u), 0u);
    }
    memoryBarrierShared();
    barrier();
    uvec3 group_1 = gl_WorkGroupID;
    uint base = (group_1.x * 4u);
    uint _e9 = padded.xyz.x;
    uint _e13 = padded.xyz.y;
    uint _e18 = padded.xyz.z;
    uint _e22 = padded.tail;
    _group_0_binding_0_cs[base] = (((_e9 + _e13) + _e18) + _e22);
    padded = Padded(uvec3(13u, 14u, 15u), 16u);
    uint _e38 = padded.xyz.x;
    uint _e42 = padded.xyz.y;
    uint _e47 = padded.xyz.z;
    uint _e51 = padded.tail;
    _group_0_binding_0_cs[(base + 1u)] = (((_e38 + _e42) + _e47) + _e51);
    _group_0_binding_0_cs[(base + 2u)] = 0u;
    _group_0_binding_0_cs[(base + 3u)] = 0u;
    return;
}

