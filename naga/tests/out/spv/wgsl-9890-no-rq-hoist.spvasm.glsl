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

void _66(rayQueryEXT _67, accelerationStructureEXT _68, _8 _69, inout uint _70, out float _71)
{
    bool _96 = (_69._m0 & 256u) != 0u;
    bool _104 = (_69._m0 & 16u) != 0u;
    bool _107 = (_69._m0 & 32u) != 0u;
    bool _115 = (_69._m0 & 1u) != 0u;
    bool _118 = (_69._m0 & 2u) != 0u;
    bool _121 = (_69._m0 & 64u) != 0u;
    bool _124 = (_69._m0 & 128u) != 0u;
    _71 = _69._m3;
    if (((((((!((((((_118 && _115) || (_124 && _115)) || (_124 && _118)) || (_124 && _121)) || (_121 && _115)) || (_121 && _118))) && (_69._m2 <= _69._m3)) && (_69._m2 >= 0.0)) && (!(any(isnan(_69._m4)) || any(isinf(_69._m4))))) && (!(any(isnan(_69._m5)) || any(isinf(_69._m5))))) && (!(((_69._m0 & 512u) != 0u) && _96))) && (!(((_104 && _96) || (_107 && _96)) || (_107 && _104))))
    {
        rayQueryInitializeEXT(_67, _68, _69._m0, _69._m1, _69._m4, _69._m2, _69._m5, _69._m3);
        _70 = 1u;
    }
    else
    {
    }
}

bool _150(rayQueryEXT _151, inout uint _152)
{
    bool _154 = false;
    if ((!((_152 & 4u) != 0u)) && ((_152 & 1u) != 0u))
    {
        bool _165 = rayQueryProceedEXT(_151);
        _154 = _165;
        _152 |= (_165 ? 2u : 6u);
    }
    return _154;
}

void main()
{
    uint _28 = 0u;
    uvec2 _35 = uvec2(0u);
    uint _32 = 0u;
    float _34 = 0.0;
    _35 = uvec2(4294967295u);
    rayQueryEXT _30;
    for (;;)
    {
        if (all(equal(_35, uvec2(0u))))
        {
            break;
        }
        _35 -= uvec2(uint(_35.y == 0u), 1u);
        if (!(_28 < 4u))
        {
            break;
        }
        _32 = 0u;
        _66(_30, _10, _8(1u, 255u, 0.001000000047497451305389404296875, 1000.0, vec3(float(_28)), vec3(0.0, -1.0, 0.0)), _32, _34);
        bool _147 = _150(_30, _32);
        _28++;
        continue;
    }
}

