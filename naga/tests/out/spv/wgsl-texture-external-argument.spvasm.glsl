#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _12
{
    float _m0;
    float _m1;
    float _m2;
    float _m3;
};

struct _13
{
    mat4 _m0;
    mat3 _m1;
    _12 _m2;
    _12 _m3;
    mat3x2 _m4;
    mat3x2 _m5;
    uvec2 _m6;
    uint _m7;
};

struct _14
{
    mat4 _m0;
    mat3 _m1;
    _12 _m2;
    _12 _m3;
    vec2 _m4;
    vec2 _m5;
    vec2 _m6;
    vec2 _m7;
    vec2 _m8;
    vec2 _m9;
    uvec2 _m10;
    uint _m11;
};

layout(set = 0, binding = 3, std140) uniform _21_20
{
    _14 _m0;
} _20;

layout(set = 0, binding = 0) uniform texture2D _15;
layout(set = 0, binding = 1) uniform texture2D _18;
layout(set = 0, binding = 2) uniform texture2D _19;

_13 _46(_14 _48)
{
    return _13(_48._m0, _48._m1, _48._m2, _48._m3, mat3x2(_48._m4, _48._m5, _48._m6), mat3x2(_48._m7, _48._m8, _48._m9), _48._m10, _48._m11);
}

uint _31(texture2D _24, texture2D _26, texture2D _28, _13 _30)
{
    return 0u;
}

uint _43(texture2D _36, texture2D _38, texture2D _40, _13 _42)
{
    return _31(_36, _38, _40, _42);
}

void main()
{
}

