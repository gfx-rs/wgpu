#version 430 core
#extension GL_ARB_compute_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Structure {
    uint num_subgroups;
    uint subgroup_size;
};

void main() {
    Structure sizes = Structure(gl_NumSubgroups, gl_SubgroupSize);
    uint subgroup_id = gl_SubgroupID;
    uint subgroup_invocation_id = gl_SubgroupInvocationID;
    subgroupMemoryBarrier();
    barrier();
    uvec4 _e7 = subgroupBallot(((subgroup_invocation_id & 1u) == 1u));
    uvec4 _e8 = subgroupBallot(true);
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
    uint _e33 = subgroupShuffle(subgroup_invocation_id, ((sizes.subgroup_size - 1u) - subgroup_invocation_id));
    uint _e35 = subgroupShuffleDown(subgroup_invocation_id, 1u);
    uint _e37 = subgroupShuffleUp(subgroup_invocation_id, 1u);
    uint _e41 = subgroupShuffleXor(subgroup_invocation_id, (sizes.subgroup_size - 1u));
    return;
}

