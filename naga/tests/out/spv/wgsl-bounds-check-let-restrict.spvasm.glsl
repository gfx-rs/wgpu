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
    return _24[1u];
}

float _33()
{
    return 2.0;
}

vec2 _38()
{
    mat2 _47 = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    return _47[1u];
}

float _52()
{
    return 2.0;
}

float _57()
{
    float _58[2] = _24;
    _58[1u] = 3.0;
    return _58[1u];
}

float _64()
{
    return (_15._m0[3u] + _15._m1.w) + _15._m2[2u].w;
}

float _89()
{
    _15._m3[min(4294967295u, (uint(_15._m3.length()) - 1u))] = 1.0;
    return _15._m3[min(4294967295u, (uint(_15._m3.length()) - 1u))];
}

void _82(float _81)
{
    _15._m0[3u] = _81;
    _15._m1.w = _81;
    _15._m2[2u].w = _81;
}

void main()
{
    float _116 = _64();
    float _118 = _89();
    _82((((((_18() + _33()) + _38().x) + _52()) + _57()) + _116) + _118);
}

