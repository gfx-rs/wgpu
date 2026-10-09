#version 460
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _4
{
    uint _m0;
};

layout(set = 0, binding = 0, std430) buffer _7_6
{
    _4 _m0;
} _6;

uint _9 = 0u;
uint _12 = 0u;

void _14()
{
    uvec4 _21 = uvec4(0u);
    uint _24 = 0u;
    uint _27 = 0u;
    _21 = uvec4(_9, _12, 3u, 4u);
    _24 = subgroupBallotFindLSB(_21);
    _27 = subgroupBallotFindMSB(_21);
    _6._m0._m0 = _24 + _27;
}

void main()
{
    _9 = gl_SubgroupInvocationID;
    _12 = gl_SubgroupSize;
    _14();
}

