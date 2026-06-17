#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _9
{
    vec3 _m0;
    float _m1;
};

struct _25
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
};

struct _26
{
    vec2 _m0;
    vec2 _m1;
    vec2 _m2;
    vec2 _m3;
};

layout(set = 0, binding = 1, std430) buffer _35_34
{
    _9 _m0;
} _34;

layout(set = 0, binding = 2, std430) readonly buffer _38_37
{
    vec2 _m0[];
} _37;

layout(set = 0, binding = 3, std140) uniform _41_40
{
    vec4 _m0[20];
} _40;

layout(set = 0, binding = 4, std140) uniform _44_43
{
    vec3 _m0;
} _43;

layout(set = 0, binding = 5, std140) uniform _47_46
{
    _25 _m0;
} _46;

layout(set = 0, binding = 6, std140) uniform _50_49
{
    mat2x4 _m0[2][2];
} _49;

layout(set = 0, binding = 7, std140) uniform _53_52
{
    _26 _m0[2][2];
} _52;

shared float _30[10];
shared uint _32;

void _57(vec3 _56)
{
}

void _61()
{
    int _72 = 1;
    _34._m0._m0 = vec3(1.0);
    _34._m0._m0.x = 1.0;
    _34._m0._m0.x = 2.0;
    _34._m0._m0[_72] = 3.0;
    _57(_34._m0._m0);
}

mat4x2 _96(_26 _98)
{
    return mat4x2(_98._m0, _98._m1, _98._m2, _98._m3);
}

mat3x2 _105(_25 _107)
{
    return mat3x2(_107._m0, _107._m1, _107._m2);
}

void main()
{
    float _131 = 1.0;
    bool _133 = true;
    if (gl_LocalInvocationIndex == 0u)
    {
        _30 = float[](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        _32 = 0u;
    }
    barrier();
    _61();
    _30[7u] = (_96(_52._m0[0u][0u]) * _49._m0[0u][0u][0u]).x;
    _30[6u] = (_105(_46._m0) * _43._m0).x;
    _30[5u] = _37._m0[1u].y;
    _30[4u] = _40._m0[0u].w;
    _30[3u] = _34._m0._m1;
    _30[2u] = _34._m0._m0.x;
    _34._m0._m1 = 4.0;
    _30[1u] = float(uint(_37._m0.length()));
    atomicExchange(_32, 2u);
}

