#version 460
#extension GL_EXT_ray_query : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _8
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT _9;

void _65(rayQueryEXT _66, accelerationStructureEXT _67, _8 _68, inout uint _69, out float _70)
{
    bool _95 = (_68._m0 & 256u) != 0u;
    bool _103 = (_68._m0 & 16u) != 0u;
    bool _106 = (_68._m0 & 32u) != 0u;
    bool _114 = (_68._m0 & 1u) != 0u;
    bool _117 = (_68._m0 & 2u) != 0u;
    bool _120 = (_68._m0 & 64u) != 0u;
    bool _123 = (_68._m0 & 128u) != 0u;
    _70 = _68._m3;
    if (((((((!((((((_117 && _114) || (_123 && _114)) || (_123 && _117)) || (_123 && _120)) || (_120 && _114)) || (_120 && _117))) && (_68._m2 <= _68._m3)) && (_68._m2 >= 0.0)) && (!(any(isnan(_68._m4)) || any(isinf(_68._m4))))) && (!(any(isnan(_68._m5)) || any(isinf(_68._m5))))) && (!(((_68._m0 & 512u) != 0u) && _95))) && (!(((_103 && _95) || (_106 && _95)) || (_106 && _103))))
    {
        rayQueryInitializeEXT(_66, _67, _68._m0, _68._m1, _68._m4, _68._m2, _68._m5, _68._m3);
        _69 = 1u;
    }
    else
    {
    }
}

bool _149(rayQueryEXT _150, inout uint _151)
{
    bool _153 = false;
    if ((!((_151 & 4u) != 0u)) && ((_151 & 1u) != 0u))
    {
        bool _164 = rayQueryProceedEXT(_150);
        _153 = _164;
        _151 |= (_164 ? 2u : 6u);
    }
    return _153;
}

void main()
{
    uint _24 = 0u;
    uint _28 = 0u;
    float _30 = 0.0;
    uvec2 _43 = uvec2(4294967295u);
    rayQueryEXT _26;
    for (;;)
    {
        if (all(equal(uvec2(0u), _43)))
        {
            break;
        }
        _43 -= uvec2(uint(_43.y == 0u), 1u);
        if (!(_24 < 4u))
        {
            break;
        }
        _28 = 0u;
        _65(_26, _9, _8(1u, 255u, 0.001000000047497451305389404296875, 1000.0, vec3(float(_24)), vec3(0.0, -1.0, 0.0)), _28, _30);
        bool _146 = _149(_26, _28);
        _24++;
        continue;
    }
}

