#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    float _m0;
    mat3x2 _m1;
};

struct _7
{
    mat3x2 _m0;
};

struct _11
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
};

struct _12
{
    float _m0;
    vec2 _m1;
    vec2 _m2;
    vec2 _m3;
};

struct _13
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
};

layout(set = 0, binding = 0, std140) uniform _17_16
{
    _11 _m0;
} _16;

layout(set = 0, binding = 1, std140) uniform _20_19
{
    _12 _m0;
} _19;

layout(set = 0, binding = 2, std140) uniform _23_22
{
    _13 _m0[2];
} _22;

layout(set = 0, binding = 3, std430) buffer _26_25
{
    vec2 _m0[8];
} _25;

void main()
{
    mat3x2 _56 = mat3x2(_16._m0._m0, _16._m0._m1, _16._m0._m2);
    _25._m0[0u] = _56[min(gl_LocalInvocationIndex, 2u)];
    mat3x2 _70 = mat3x2(_19._m0._m1, _19._m0._m2, _19._m0._m3);
    _25._m0[1u] = _70[min(gl_LocalInvocationIndex, 2u)];
    mat3x2 _82 = mat3x2(_16._m0._m0, _16._m0._m1, _16._m0._m2);
    mat3x2 _95 = mat3x2(_19._m0._m1, _19._m0._m2, _19._m0._m3);
    _25._m0[2u] = vec2(_82[min(gl_LocalInvocationIndex, 2u)][min(gl_LocalInvocationIndex, 1u)], _95[min(gl_LocalInvocationIndex, 2u)][min(gl_LocalInvocationIndex, 1u)]);
    mat3x2 _113 = mat3x2(_22._m0[min(gl_LocalInvocationIndex, 1u)]._m0, _22._m0[min(gl_LocalInvocationIndex, 1u)]._m1, _22._m0[min(gl_LocalInvocationIndex, 1u)]._m2);
    _25._m0[3u] = _113[min(gl_LocalInvocationIndex, 2u)];
    mat3x2 _128 = mat3x2(_22._m0[min(gl_LocalInvocationIndex, 1u)]._m0, _22._m0[min(gl_LocalInvocationIndex, 1u)]._m1, _22._m0[min(gl_LocalInvocationIndex, 1u)]._m2);
    _25._m0[4u] = vec2(_128[min(gl_LocalInvocationIndex, 2u)][min(gl_LocalInvocationIndex, 1u)], 0.0);
}

