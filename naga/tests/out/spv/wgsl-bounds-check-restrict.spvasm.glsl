#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _10_12
{
    float _m0[10];
    vec4 _m1;
    mat3x4 _m2;
    float _m3[];
} _12;

float _16(int _15)
{
    return _12._m0[min(uint(_15), 9u)];
}

float _28(int _27)
{
    return _12._m3[min(uint(_27), (uint(_12._m3.length()) - 1u))];
}

float _40(int _39)
{
    return _12._m1[min(uint(_39), 3u)];
}

float _49(vec4 _47, int _48)
{
    return _47[min(uint(_48), 3u)];
}

vec4 _56(int _55)
{
    return _12._m2[min(uint(_55), 2u)];
}

float _67(int _65, int _66)
{
    return _12._m2[min(uint(_65), 2u)][min(uint(_66), 3u)];
}

float _76(int _75)
{
    return _12._m0[min(uint(int(clamp(sin(float(_75) / 100.0) * 100.0, -2147483648.0, 2147483520.0))), 9u)];
}

float _91()
{
    return (_12._m0[9u] + _12._m1.w) + _12._m2[2u].w;
}

void _105(int _103, float _104)
{
    _12._m0[min(uint(_103), 9u)] = _104;
}

void _113(int _111, float _112)
{
    _12._m3[min(uint(_111), (uint(_12._m3.length()) - 1u))] = _112;
}

void _122(int _120, float _121)
{
    _12._m1[min(uint(_120), 3u)] = _121;
}

void _129(int _127, vec4 _128)
{
    _12._m2[min(uint(_127), 2u)] = _128;
}

void _138(int _135, int _136, float _137)
{
    _12._m2[min(uint(_135), 2u)][min(uint(_136), 3u)] = _137;
}

void _147(int _145, float _146)
{
    _12._m0[min(uint(int(clamp(sin(float(_145) / 100.0) * 100.0, -2147483648.0, 2147483520.0))), 9u)] = _146;
}

void _159(float _158)
{
    _12._m0[9u] = _158;
    _12._m1.w = _158;
    _12._m2[2u].w = _158;
}

float _166()
{
    return _12._m3[min(1000u, (uint(_12._m3.length()) - 1u))];
}

void _176(float _175)
{
    _12._m3[min(1000u, (uint(_12._m3.length()) - 1u))] = _175;
}

void main()
{
    _105(1, 2.0);
    _113(1, 2.0);
    _122(1, 2.0);
    _129(1, vec4(2.0, 3.0, 4.0, 5.0));
    _138(1, 2, 1.0);
    _147(1, 1.0);
    _159(1.0);
    _176(1.0);
}

