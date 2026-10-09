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

struct Output {
    uint result;
};
layout(std430) buffer Output_block_0Compute { Output _group_0_binding_0_cs; };

uint gen_gl_SubgroupInvocationID_1 = 0u;

uint gen_gl_SubgroupSize_1 = 0u;


void main_1() {
    uvec4 mask = uvec4(0u);
    uint lsb = 0u;
    uint msb = 0u;
    uint _e4 = gen_gl_SubgroupInvocationID_1;
    uint _e5 = gen_gl_SubgroupSize_1;
    mask = uvec4(_e4, _e5, 3u, 4u);
    uvec4 _e10 = mask;
    uint _e11 = subgroupBallotFindLSB(_e10);
    lsb = _e11;
    uvec4 _e13 = mask;
    uint _e14 = subgroupBallotFindMSB(_e13);
    msb = _e14;
    uint _e16 = lsb;
    uint _e17 = msb;
    _group_0_binding_0_cs.result = (_e16 + _e17);
    return;
}

void main() {
    uint gen_gl_SubgroupInvocationID = gl_SubgroupInvocationID;
    uint gen_gl_SubgroupSize = gl_SubgroupSize;
    gen_gl_SubgroupInvocationID_1 = gen_gl_SubgroupInvocationID;
    gen_gl_SubgroupSize_1 = gen_gl_SubgroupSize;
    main_1();
    return;
}

