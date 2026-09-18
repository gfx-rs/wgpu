#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    vec4 _m0;
    mat4 _m1;
};

layout(set = 0, binding = 0, std430) buffer _13_12
{
    float _m0[];
} _12;

layout(push_constant, std430) uniform _10_9
{
    _6 _m0;
} _9;

void main()
{
    _12._m0[0u] = _9._m0._m0.x;
    _12._m0[1u] = _9._m0._m0[min(gl_LocalInvocationIndex, 3u)];
    _12._m0[2u] = _9._m0._m1[0u].x;
    _12._m0[3u] = _9._m0._m1[0u][min(gl_LocalInvocationIndex, 3u)];
}

