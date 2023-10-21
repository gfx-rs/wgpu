#version 430 core
#extension GL_ARB_compute_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main() {
    uint num_subgroups = gl_NumSubgroups;
    uint subgroup_id = gl_SubgroupID;
    uint subgroup_size = gl_SubgroupSize;
    uint subgroup_invocation_id = gl_SubgroupInvocationID;
    subgroupMemoryBarrier();
    barrier();
    uvec4 _e8 = subgroupBallot(((subgroup_invocation_id & 1u) == 1u));
    uvec4 _e9 = subgroupBallot(true);
    bool _e12 = subgroupAll((subgroup_invocation_id != 0u));
    bool _e15 = subgroupAny((subgroup_invocation_id == 0u));
    uint _e16 = subgroupAdd(subgroup_invocation_id);
    uint _e17 = subgroupMul(subgroup_invocation_id);
    uint _e18 = subgroupMin(subgroup_invocation_id);
    uint _e19 = subgroupMax(subgroup_invocation_id);
    uint _e20 = subgroupAnd(subgroup_invocation_id);
    uint _e21 = subgroupOr(subgroup_invocation_id);
    uint _e22 = subgroupXor(subgroup_invocation_id);
    uint _e23 = subgroupExclusiveAdd(subgroup_invocation_id);
    uint _e24 = subgroupExclusiveMul(subgroup_invocation_id);
    uint _e25 = subgroupInclusiveAdd(subgroup_invocation_id);
    uint _e26 = subgroupInclusiveMul(subgroup_invocation_id);
    uint _e27 = subgroupBroadcastFirst(subgroup_invocation_id);
    uint _e29 = subgroupBroadcast(subgroup_invocation_id, 4u);
    uint _e33 = subgroupShuffle(subgroup_invocation_id, ((subgroup_size - 1u) - subgroup_invocation_id));
    uint _e35 = subgroupShuffleDown(subgroup_invocation_id, 1u);
    uint _e37 = subgroupShuffleUp(subgroup_invocation_id, 1u);
    uint _e40 = subgroupShuffleXor(subgroup_invocation_id, (subgroup_size - 1u));
    return;
}

