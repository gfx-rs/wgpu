#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _15
{
    vec2 _m0;
    vec2 _m1;
};

layout(set = 0, binding = 0, std140) uniform _20_19
{
    _15 _m0[64];
} _19;

layout(set = 0, binding = 1, std140) uniform _23_22
{
    _15 _m0[20][2];
} _22;

layout(set = 0, binding = 2, std430) buffer _26_25
{
    mat2 _m0[4];
} _25;

mat2[64] _30(_15 _29[64])
{
    _15 _34[64] = _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0)));
    mat2 _37[64] = mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)));
    uint _40 = 0u;
    uvec2 _54 = uvec2(4294967295u);
    _34 = _29;
    for (;;)
    {
        if (all(equal(uvec2(0u), _54)))
        {
            break;
        }
        _54 -= uvec2(uint(_54.y == 0u), 1u);
        if (!(_40 < 64u))
        {
            break;
        }
        _37[_40] = mat2(_34[_40]._m0, _34[_40]._m1);
        _40++;
        continue;
    }
    return _37;
}

mat2[2] _82(_15 _81[2])
{
    return mat2[](mat2(_81[0]._m0, _81[0]._m1), mat2(_81[1]._m0, _81[1]._m1));
}

mat2[20][2] _96(_15 _95[20][2])
{
    _15 _98[20][2] = _15[][](_15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))), _15[](_15(vec2(0.0), vec2(0.0)), _15(vec2(0.0), vec2(0.0))));
    mat2 _101[20][2] = mat2[][](mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))), mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0))));
    uint _104 = 0u;
    uvec2 _110 = uvec2(4294967295u);
    _98 = _95;
    for (;;)
    {
        if (all(equal(uvec2(0u), _110)))
        {
            break;
        }
        _110 -= uvec2(uint(_110.y == 0u), 1u);
        if (!(_104 < 20u))
        {
            break;
        }
        _101[_104] = _82(_98[_104]);
        _104++;
        continue;
    }
    return _101;
}

void main()
{
    mat2 _152[64] = _30(_19._m0);
    _25._m0[0u] = _152[gl_LocalInvocationIndex];
    mat2 _156[20][2] = _96(_22._m0);
    _25._m0[1u] = _156[gl_LocalInvocationIndex][1u];
}

