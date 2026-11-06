#version 430 core
#extension GL_ARB_compute_shader : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_KHR_shader_subgroup_quad : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct type_1 {
    uint member;
};
uint global = 0u;

layout(std430) buffer type_1_block_0Compute { type_1 _group_0_binding_0_cs; };


void function() {
    uint _e3 = global;
    uvec4 _e5 = subgroupBallot((_e3 != 0u));
    uint _e6 = subgroupBallotFindLSB(_e5);
    uint _e7 = subgroupBallotFindMSB(_e5);
    _group_0_binding_0_cs.member = (_e6 + _e7);
    return;
}

void main() {
    uint param = gl_SubgroupInvocationID;
    global = param;
    function();
}

