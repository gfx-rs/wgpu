#version 460
#extension GL_KHR_shader_subgroup_ballot : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _4
{
    uint _m0;
};

layout(set = 0, binding = 0, std430) buffer _8_7
{
    _4 _m0;
} _7;

uvec3 _10 = uvec3(0u);

void _14()
{
    uvec4 _22 = uvec4(0u);
    uint _25 = 0u;
    uint _28 = 0u;
    _22 = uvec4(_10.x, 2u, 3u, 4u);
    _25 = subgroupBallotFindLSB(_22);
    _28 = subgroupBallotFindMSB(_22);
    _7._m0._m0 = _25 + _28;
}

void main()
{
    _10 = gl_GlobalInvocationID;
    _14();
}

