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
    vec2 _66;
    if (gl_LocalInvocationIndex < 3u)
    {
        _66 = _56[gl_LocalInvocationIndex];
    }
    else
    {
        _66 = vec2(0.0);
    }
    _25._m0[0u] = _66;
    mat3x2 _75 = mat3x2(_19._m0._m1, _19._m0._m2, _19._m0._m3);
    vec2 _81;
    if (gl_LocalInvocationIndex < 3u)
    {
        _81 = _75[gl_LocalInvocationIndex];
    }
    else
    {
        _81 = vec2(0.0);
    }
    _25._m0[1u] = _81;
    mat3x2 _90 = mat3x2(_16._m0._m0, _16._m0._m1, _16._m0._m2);
    float _100;
    if ((gl_LocalInvocationIndex < 2u) && (gl_LocalInvocationIndex < 3u))
    {
        _100 = _90[gl_LocalInvocationIndex][gl_LocalInvocationIndex];
    }
    else
    {
        _100 = 0.0;
    }
    mat3x2 _108 = mat3x2(_19._m0._m1, _19._m0._m2, _19._m0._m3);
    float _116;
    if ((gl_LocalInvocationIndex < 2u) && (gl_LocalInvocationIndex < 3u))
    {
        _116 = _108[gl_LocalInvocationIndex][gl_LocalInvocationIndex];
    }
    else
    {
        _116 = 0.0;
    }
    _25._m0[2u] = vec2(_100, _116);
    vec2 _125;
    if (gl_LocalInvocationIndex < 2u)
    {
        _125 = _22._m0[gl_LocalInvocationIndex]._m0;
    }
    else
    {
        _125 = vec2(0.0);
    }
    vec2 _131;
    if (gl_LocalInvocationIndex < 2u)
    {
        _131 = _22._m0[gl_LocalInvocationIndex]._m1;
    }
    else
    {
        _131 = vec2(0.0);
    }
    vec2 _137;
    if (gl_LocalInvocationIndex < 2u)
    {
        _137 = _22._m0[gl_LocalInvocationIndex]._m2;
    }
    else
    {
        _137 = vec2(0.0);
    }
    mat3x2 _139 = mat3x2(_125, _131, _137);
    vec2 _145;
    if (gl_LocalInvocationIndex < 3u)
    {
        _145 = _139[gl_LocalInvocationIndex];
    }
    else
    {
        _145 = vec2(0.0);
    }
    _25._m0[3u] = _145;
    vec2 _152;
    if (gl_LocalInvocationIndex < 2u)
    {
        _152 = _22._m0[gl_LocalInvocationIndex]._m0;
    }
    else
    {
        _152 = vec2(0.0);
    }
    vec2 _158;
    if (gl_LocalInvocationIndex < 2u)
    {
        _158 = _22._m0[gl_LocalInvocationIndex]._m1;
    }
    else
    {
        _158 = vec2(0.0);
    }
    vec2 _164;
    if (gl_LocalInvocationIndex < 2u)
    {
        _164 = _22._m0[gl_LocalInvocationIndex]._m2;
    }
    else
    {
        _164 = vec2(0.0);
    }
    mat3x2 _166 = mat3x2(_152, _158, _164);
    float _174;
    if ((gl_LocalInvocationIndex < 2u) && (gl_LocalInvocationIndex < 3u))
    {
        _174 = _166[gl_LocalInvocationIndex][gl_LocalInvocationIndex];
    }
    else
    {
        _174 = 0.0;
    }
    _25._m0[4u] = vec2(_174, 0.0);
}

