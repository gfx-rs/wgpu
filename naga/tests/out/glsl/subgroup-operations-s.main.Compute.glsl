#version 430 core
#extension GL_ARB_compute_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uint num_subgroups_1 = 0u;

uint subgroup_id_1 = 0u;

uint subgroup_size_1 = 0u;

uint subgroup_invocation_id_1 = 0u;


void main_1() {
    uint _e5 = subgroup_size_1;
    uint _e6 = subgroup_invocation_id_1;
    uvec4 _e9 = subgroupBallot(((_e6 & 1u) == 1u));
    uvec4 _e10 = subgroupBallot(true);
    bool _e12 = subgroupAll((_e6 != 0u));
    bool _e14 = subgroupAny((_e6 == 0u));
    uint _e15 = subgroupAdd(_e6);
    uint _e16 = subgroupMul(_e6);
    uint _e17 = subgroupMin(_e6);
    uint _e18 = subgroupMax(_e6);
    uint _e19 = subgroupAnd(_e6);
    uint _e20 = subgroupOr(_e6);
    uint _e21 = subgroupXor(_e6);
    uint _e22 = subgroupExclusiveAdd(_e6);
    uint _e23 = subgroupExclusiveMul(_e6);
    uint _e24 = subgroupInclusiveAdd(_e6);
    uint _e25 = subgroupInclusiveMul(_e6);
    uint _e26 = subgroupBroadcastFirst(_e6);
    uint _e27 = subgroupBroadcast(_e6, 4u);
    uint _e30 = subgroupShuffle(_e6, ((_e5 - 1u) - _e6));
    uint _e31 = subgroupShuffleDown(_e6, 1u);
    uint _e32 = subgroupShuffleUp(_e6, 1u);
    uint _e34 = subgroupShuffleXor(_e6, (_e5 - 1u));
    return;
}

void main() {
    uint num_subgroups = gl_NumSubgroups;
    uint subgroup_id = gl_SubgroupID;
    uint subgroup_size = gl_SubgroupSize;
    uint subgroup_invocation_id = gl_SubgroupInvocationID;
    num_subgroups_1 = num_subgroups;
    subgroup_id_1 = subgroup_id;
    subgroup_size_1 = subgroup_size;
    subgroup_invocation_id_1 = subgroup_invocation_id;
    main_1();
}

