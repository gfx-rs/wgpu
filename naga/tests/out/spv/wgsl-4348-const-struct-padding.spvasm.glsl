#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    float _m0;
    vec3 _m1;
    float _m2;
};

struct _10
{
    int _m0;
    ivec4 _m1;
    mat3x2 _m2;
};

struct _12
{
    mat2 _m0;
    float _m1;
};

layout(set = 0, binding = 0, std430) buffer _33_32
{
    float _m0[];
} _32;

float _37(_5 _36)
{
    return (_36._m0 + _36._m1.x) + _36._m2;
}

float _48(_10 _47)
{
    return (float(_47._m0) + float(_47._m1.x)) + _47._m2[0].x;
}

float _63(_12 _62)
{
    return _62._m0[0].x + _62._m1;
}

void main()
{
    _32._m0[0u] = _37(_5(1.0, vec3(2.0), 3.0));
    _32._m0[1u] = _48(_10(1, ivec4(3), mat3x2(vec2(1.0, 2.0), vec2(3.0, 4.0), vec2(5.0, 6.0))));
    _32._m0[2u] = _63(_12(mat2(vec2(1.0, 2.0), vec2(3.0, 4.0)), 5.0));
}

