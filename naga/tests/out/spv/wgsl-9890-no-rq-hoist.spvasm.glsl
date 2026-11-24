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

layout(set = 0, binding = 0) uniform accelerationStructureEXT _10;

void _67(rayQueryEXT _68, accelerationStructureEXT _69, _8 _70, inout uint _71, out float _72)
{
    bool _97 = (_70._m0 & 256u) != 0u;
    bool _105 = (_70._m0 & 16u) != 0u;
    bool _108 = (_70._m0 & 32u) != 0u;
    bool _116 = (_70._m0 & 1u) != 0u;
    bool _119 = (_70._m0 & 2u) != 0u;
    bool _122 = (_70._m0 & 64u) != 0u;
    bool _125 = (_70._m0 & 128u) != 0u;
    _72 = _70._m3;
    if (((((((!((((((_119 && _116) || (_125 && _116)) || (_125 && _119)) || (_125 && _122)) || (_122 && _116)) || (_122 && _119))) && (_70._m2 <= _70._m3)) && (_70._m2 >= 0.0)) && (!(any(isnan(_70._m4)) || any(isinf(_70._m4))))) && (!(any(isnan(_70._m5)) || any(isinf(_70._m5))))) && (!(((_70._m0 & 512u) != 0u) && _97))) && (!(((_105 && _97) || (_108 && _97)) || (_108 && _105))))
    {
        rayQueryInitializeEXT(_68, _69, _70._m0, _70._m1, _70._m4, _70._m2, _70._m5, _70._m3);
        _71 = 1u;
    }
    else
    {
    }
}

bool _151(rayQueryEXT _152, inout uint _153)
{
    bool _155 = false;
    if ((!((_153 & 4u) != 0u)) && ((_153 & 1u) != 0u))
    {
        bool _166 = rayQueryProceedEXT(_152);
        _155 = _166;
        _153 |= (_166 ? 2u : 6u);
    }
    return _155;
}

void main()
{
    uint _28 = 0u;
    uvec2 _35 = uvec2(4294967295u);
    uint _32 = 0u;
    float _34 = 0.0;
    rayQueryEXT _30;
    for (;;)
    {
        uvec2 _44 = _35;
        _35 = _44 - uvec2(uint(_44.y == 0u), 1u);
        if (all(equal(_44, uvec2(0u))))
        {
            break;
        }
        if (!(_28 < 4u))
        {
            break;
        }
        _32 = 0u;
        _67(_30, _10, _8(1u, 255u, 0.001000000047497451305389404296875, 1000.0, vec3(float(_28)), vec3(0.0, -1.0, 0.0)), _32, _34);
        bool _148 = _151(_30, _32);
        _28++;
        continue;
    }
}

