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
    float _29;
    if (uint(_15) < 10u)
    {
        _29 = _12._m0[_15];
    }
    else
    {
        _29 = 0.0;
    }
    return _29;
}

float _32(int _31)
{
    float _42;
    if (uint(_31) < uint(_12._m3.length()))
    {
        _42 = _12._m3[_31];
    }
    else
    {
        _42 = 0.0;
    }
    return _42;
}

float _45(int _44)
{
    float _55;
    if (uint(_44) < 4u)
    {
        _55 = _12._m1[_44];
    }
    else
    {
        _55 = 0.0;
    }
    return _55;
}

float _59(vec4 _57, int _58)
{
    float _66;
    if (uint(_58) < 4u)
    {
        _66 = _57[_58];
    }
    else
    {
        _66 = 0.0;
    }
    return _66;
}

vec4 _69(int _68)
{
    vec4 _80;
    if (uint(_68) < 3u)
    {
        _80 = _12._m2[_68];
    }
    else
    {
        _80 = vec4(0.0);
    }
    return _80;
}

float _84(int _82, int _83)
{
    float _94;
    if ((uint(_83) < 4u) && (uint(_82) < 3u))
    {
        _94 = _12._m2[_82][_83];
    }
    else
    {
        _94 = 0.0;
    }
    return _94;
}

float _97(int _96)
{
    int _107 = int(clamp(sin(float(_96) / 100.0) * 100.0, -2147483648.0, 2147483520.0));
    float _113;
    if (uint(_107) < 10u)
    {
        _113 = _12._m0[_107];
    }
    else
    {
        _113 = 0.0;
    }
    return _113;
}

float _115()
{
    return (_12._m0[9u] + _12._m1.w) + _12._m2[2u].w;
}

void _130(int _128, float _129)
{
    if (uint(_128) < 10u)
    {
        _12._m0[_128] = _129;
    }
}

void _140(int _138, float _139)
{
    if (uint(_138) < uint(_12._m3.length()))
    {
        _12._m3[_138] = _139;
    }
}

void _150(int _148, float _149)
{
    if (uint(_148) < 4u)
    {
        _12._m1[_148] = _149;
    }
}

void _159(int _157, vec4 _158)
{
    if (uint(_157) < 3u)
    {
        _12._m2[_157] = _158;
    }
}

void _170(int _167, int _168, float _169)
{
    if ((uint(_168) < 4u) && (uint(_167) < 3u))
    {
        _12._m2[_167][_168] = _169;
    }
}

void _182(int _180, float _181)
{
    int _189 = int(clamp(sin(float(_180) / 100.0) * 100.0, -2147483648.0, 2147483520.0));
    if (uint(_189) < 10u)
    {
        _12._m0[_189] = _181;
    }
}

void _196(float _195)
{
    _12._m0[9u] = _195;
    _12._m1.w = _195;
    _12._m2[2u].w = _195;
}

float _203()
{
    float _212;
    if (1000u < uint(_12._m3.length()))
    {
        _212 = _12._m3[1000u];
    }
    else
    {
        _212 = 0.0;
    }
    return _212;
}

void _215(float _214)
{
    if (1000u < uint(_12._m3.length()))
    {
        _12._m3[1000u] = _214;
    }
}

void main()
{
    _130(1, 2.0);
    _140(1, 2.0);
    _150(1, 2.0);
    _159(1, vec4(2.0, 3.0, 4.0, 5.0));
    _170(1, 2, 1.0);
    _182(1, 1.0);
    _196(1.0);
    _215(1.0);
}

