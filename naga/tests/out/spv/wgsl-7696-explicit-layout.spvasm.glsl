#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    vec2 _m0;
    float _m1;
};

struct _11
{
    mat2 _m0;
    vec4 _m1[2];
};

struct _19
{
    vec2 _m0;
    vec2 _m1;
    vec4 _m2[2];
};

const vec2 _64[6] = vec2[](vec2(-1.0), vec2(1.0), vec2(0.0), vec2(2.0), vec2(3.0), vec2(4.0));

layout(set = 0, binding = 0, std430) buffer _21_20
{
    _11 _m0;
} _20;

layout(set = 0, binding = 1, std140) uniform _24_23
{
    _19 _m0;
} _23;

uint _26[3] = uint[](0u, 0u, 0u);
shared int _29[5];

uint _31(uint _33, uint _34)
{
    return _33 % ((_34 == 0u) ? 1u : _34);
}

void main()
{
    _5 _68 = _5(vec2(0.0), 0.0);
    if (gl_LocalInvocationIndex == 0u)
    {
        _29 = int[](0, 0, 0, 0, 0);
    }
    barrier();
    _68 = _5(_64[_31(gl_LocalInvocationIndex, 6u)], 1.0);
    _26[_31(gl_LocalInvocationIndex, 3u)] = gl_LocalInvocationIndex;
    _29[_31(gl_LocalInvocationIndex, 5u)] = int(gl_LocalInvocationIndex);
    barrier();
    _20._m0._m1[0u] = vec4(_68._m0, _68._m1, float(_26[0u]) + float(_29[1u])) + _23._m0._m2[0u];
}

