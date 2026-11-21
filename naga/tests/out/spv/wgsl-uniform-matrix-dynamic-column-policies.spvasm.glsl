#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    mat3x2 _m0;
};

struct _13
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
};

struct _14
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
};

layout(set = 0, binding = 0, std140) uniform _17_16
{
    _13 _m0;
} _16;

layout(set = 0, binding = 1, std140) uniform _20_19
{
    _14 _m0[2];
} _19;

layout(set = 0, binding = 2, std140) uniform _23_22
{
    vec4 _m0[2];
} _22;

layout(set = 0, binding = 3, std430) buffer _26_25
{
    vec2 _m0[4];
} _25;

void main()
{
    mat3x2 _55 = mat3x2(_16._m0._m0, _16._m0._m1, _16._m0._m2);
    vec2 _65;
    if (gl_LocalInvocationIndex < 3u)
    {
        _65 = _55[gl_LocalInvocationIndex];
    }
    else
    {
        _65 = vec2(0.0);
    }
    _25._m0[0u] = _65;
    _25._m0[1u] = _19._m0[min(gl_LocalInvocationIndex, 1u)]._m1;
    mat3x2 _82 = mat3x2(_19._m0[min(gl_LocalInvocationIndex, 1u)]._m0, _19._m0[min(gl_LocalInvocationIndex, 1u)]._m1, _19._m0[min(gl_LocalInvocationIndex, 1u)]._m2);
    vec2 _88;
    if (gl_LocalInvocationIndex < 3u)
    {
        _88 = _82[gl_LocalInvocationIndex];
    }
    else
    {
        _88 = vec2(0.0);
    }
    _25._m0[2u] = _88;
    _25._m0[3u] = _22._m0[min(gl_LocalInvocationIndex, 1u)].xy;
}

