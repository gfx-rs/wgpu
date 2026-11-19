#version 460
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _4
{
    uint _m0;
};

layout(set = 0, binding = 0, std430) buffer _10_9
{
    _4 _m0;
} _9;

uint _6 = 0u;

void _13()
{
    uvec4 _23 = subgroupBallot(_6 != 0u);
    _9._m0._m0 = subgroupBallotFindLSB(_23) + subgroupBallotFindMSB(_23);
}

void main()
{
    _6 = gl_SubgroupInvocationID;
    _13();
}

