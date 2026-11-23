#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _9
{
    vec3 _m0;
    float _m1;
};

struct _23
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
};

struct _24
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
    vec2 _m3;
};

layout(set = 0, binding = 1, std430) buffer _33_32
{
    _9 _m0;
} _32;

layout(set = 0, binding = 2, std430) readonly buffer _36_35
{
    vec2 _m0[];
} _35;

layout(set = 0, binding = 3, std140) uniform _39_38
{
    vec4 _m0[20];
} _38;

layout(set = 0, binding = 4, std140) uniform _42_41
{
    vec3 _m0;
} _41;

layout(set = 0, binding = 5, std140) uniform _45_44
{
    _23 _m0;
} _44;

layout(set = 0, binding = 6, std140) uniform _48_47
{
    mat2x4 _m0[2][2];
} _47;

layout(set = 0, binding = 7, std140) uniform _51_50
{
    _24 _m0[2][2];
} _50;

shared float _28[10];
shared uint _30;

void _55(vec3 _54)
{
}

void _59()
{
    int _70 = 1;
    _32._m0._m0 = vec3(1.0);
    _32._m0._m0.x = 1.0;
    _32._m0._m0.x = 2.0;
    _32._m0._m0[_70] = 3.0;
    _55(_32._m0._m0);
}

void main()
{
    float _115 = 1.0;
    bool _117 = true;
    uint _119 = 0u;
    _119 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (_119 >= 10u)
        {
            break;
        }
        _28[_119] = 0.0;
        _119++;
        continue;
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _30 = 0u;
    }
    barrier();
    _59();
    _28[7u] = (mat4x2(_50._m0[0u][0u]._m0, _50._m0[0u][0u]._m1, _50._m0[0u][0u]._m2, _50._m0[0u][0u]._m3) * _47._m0[0u][0u][0u]).x;
    _28[6u] = (mat3x2(_44._m0._m0, _44._m0._m1, _44._m0._m2) * _41._m0).x;
    _28[5u] = _35._m0[1u].y;
    _28[4u] = _38._m0[0u].w;
    _28[3u] = _32._m0._m1;
    _28[2u] = _32._m0._m0.x;
    _32._m0._m1 = 4.0;
    _28[1u] = float(uint(_35._m0.length()));
    atomicExchange(_30, 2u);
}

