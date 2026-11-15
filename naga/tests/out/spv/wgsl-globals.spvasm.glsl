#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _9
{
    vec3 _m0;
    float _m1;
};

struct _26
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
};

struct _27
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
    vec2 _m3;
};

layout(set = 0, binding = 1, std430) buffer _36_35
{
    _9 _m0;
} _35;

layout(set = 0, binding = 2, std430) readonly buffer _39_38
{
    vec2 _m0[];
} _38;

layout(set = 0, binding = 3, std140) uniform _42_41
{
    vec4 _m0[20];
} _41;

layout(set = 0, binding = 4, std140) uniform _45_44
{
    vec3 _m0;
} _44;

layout(set = 0, binding = 5, std140) uniform _48_47
{
    _26 _m0;
} _47;

layout(set = 0, binding = 6, std140) uniform _51_50
{
    mat2x4 _m0[2][2];
} _50;

layout(set = 0, binding = 7, std140) uniform _54_53
{
    _27 _m0[2][2];
} _53;

shared float _31[10];
shared uint _33;

void _58(vec3 _57)
{
}

void _62()
{
    int _73 = 1;
    _35._m0._m0 = vec3(1.0);
    _35._m0._m0.x = 1.0;
    _35._m0._m0.x = 2.0;
    _35._m0._m0[_73] = 3.0;
    _58(_35._m0._m0);
}

mat4x2 _99(_27 _101)
{
    return mat4x2(_101._m0, _101._m1, _101._m2, _101._m3);
}

mat3x2 _108(_26 _110)
{
    return mat3x2(_110._m0, _110._m1, _110._m2);
}

void main()
{
    float _134 = 1.0;
    bool _136 = true;
    if (gl_LocalInvocationIndex == 0u)
    {
        _31 = float[](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        _33 = 0u;
    }
    barrier();
    _62();
    _31[7u] = (_99(_53._m0[0u][0u]) * _50._m0[0u][0u][0u]).x;
    _31[6u] = (_108(_47._m0) * _44._m0).x;
    _31[5u] = _38._m0[1u].y;
    _31[4u] = _41._m0[0u].w;
    _31[3u] = _35._m0._m1;
    _31[2u] = _35._m0._m0.x;
    _35._m0._m1 = 4.0;
    _31[1u] = float(uint(_38._m0.length()));
    atomicExchange(_33, 2u);
}

