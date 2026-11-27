#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const float _24[2] = float[](1.0, 2.0);

layout(set = 0, binding = 0, std430) buffer _10_15
{
    float _m0[4];
    vec4 _m1;
    mat3x4 _m2;
    float _m3[];
} _15;

float _18()
{
    float _37;
    if (9u < 2u)
    {
        _37 = _24[9u];
    }
    else
    {
        _37 = 0.0;
    }
    return _37;
}

float _39()
{
    float _46;
    if (9u < 2u)
    {
        _46 = vec2(1.0, 2.0).x;
    }
    else
    {
        _46 = 0.0;
    }
    return _46;
}

vec2 _48()
{
    mat2 _56 = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    vec2 _64;
    if (9u < 2u)
    {
        _64 = _56[9u];
    }
    else
    {
        _64 = vec2(0.0);
    }
    return _64;
}

float _66()
{
    float _74;
    if (4294967295u < 2u)
    {
        _74 = vec2(1.0, 2.0).x;
    }
    else
    {
        _74 = 0.0;
    }
    return _74;
}

float _76()
{
    float _77[2] = _24;
    if (9u < 2u)
    {
        _77[9u] = 3.0;
    }
    float _88;
    if (9u < 2u)
    {
        _88 = _77[9u];
    }
    else
    {
        _88 = 0.0;
    }
    return _88;
}

float _90()
{
    float _100;
    if (9u < 4u)
    {
        _100 = _15._m0[9u];
    }
    else
    {
        _100 = 0.0;
    }
    float _108;
    if (9u < 4u)
    {
        _108 = _15._m1.x;
    }
    else
    {
        _108 = 0.0;
    }
    float _119;
    if ((9u < 4u) && (9u < 3u))
    {
        _119 = _15._m2[9u].x;
    }
    else
    {
        _119 = 0.0;
    }
    return (_100 + _108) + _119;
}

float _141()
{
    if (4294967295u < uint(_15._m3.length()))
    {
        _15._m3[4294967295u] = 1.0;
    }
    float _155;
    if (4294967295u < uint(_15._m3.length()))
    {
        _155 = _15._m3[4294967295u];
    }
    else
    {
        _155 = 0.0;
    }
    return _155;
}

void _123(float _122)
{
    if (9u < 4u)
    {
        _15._m0[9u] = _122;
    }
    if (9u < 4u)
    {
        _15._m1.x = _122;
    }
    if ((9u < 4u) && (9u < 3u))
    {
        _15._m2[9u].x = _122;
    }
}

void main()
{
    float _170 = _90();
    float _172 = _141();
    _123((((((_18() + _39()) + _48().x) + _66()) + _76()) + _170) + _172);
}

