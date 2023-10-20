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
    bool _e11 = subgroupAll((subgroup_invocation_id != 0u));
    bool _e14 = subgroupAny((subgroup_invocation_id == 0u));
    uint _e15 = subgroupAdd(subgroup_invocation_id);
    uint _e16 = subgroupMul(subgroup_invocation_id);
    uint _e17 = subgroupMin(subgroup_invocation_id);
    uint _e18 = subgroupMax(subgroup_invocation_id);
    uint _e19 = subgroupAnd(subgroup_invocation_id);
    uint _e20 = subgroupOr(subgroup_invocation_id);
    uint _e21 = subgroupXor(subgroup_invocation_id);
    uint _e22 = subgroupExclusiveAdd(subgroup_invocation_id);
    uint _e23 = subgroupExclusiveMul(subgroup_invocation_id);
    uint _e24 = subgroupInclusiveAdd(subgroup_invocation_id);
    uint _e25 = subgroupInclusiveMul(subgroup_invocation_id);
    uint _e26 = subgroupBroadcastFirst(subgroup_invocation_id);
    uint _e28 = subgroupBroadcast(subgroup_invocation_id, 4u);
    uint _e32 = subgroupShuffle(subgroup_invocation_id, ((subgroup_size - 1u) - subgroup_invocation_id));
    uint _e34 = subgroupShuffleDown(subgroup_invocation_id, 1u);
    uint _e36 = subgroupShuffleUp(subgroup_invocation_id, 1u);
    uint _e39 = subgroupShuffleXor(subgroup_invocation_id, (subgroup_size - 1u));
    return;
}

