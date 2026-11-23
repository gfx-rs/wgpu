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

layout(set = 0, binding = 0) uniform accelerationStructureEXT _11;

void _45(rayQueryEXT _46, accelerationStructureEXT _47, _8 _48, inout uint _49, out float _50)
{
    bool _76 = (_48._m0 & 256u) != 0u;
    bool _84 = (_48._m0 & 16u) != 0u;
    bool _87 = (_48._m0 & 32u) != 0u;
    bool _95 = (_48._m0 & 1u) != 0u;
    bool _98 = (_48._m0 & 2u) != 0u;
    bool _101 = (_48._m0 & 64u) != 0u;
    bool _104 = (_48._m0 & 128u) != 0u;
    _50 = _48._m3;
    if (((((((!((((((_98 && _95) || (_104 && _95)) || (_104 && _98)) || (_104 && _101)) || (_101 && _95)) || (_101 && _98))) && (_48._m2 <= _48._m3)) && (_48._m2 >= 0.0)) && (!(any(isnan(_48._m4)) || any(isinf(_48._m4))))) && (!(any(isnan(_48._m5)) || any(isinf(_48._m5))))) && (!(((_48._m0 & 512u) != 0u) && _76))) && (!(((_84 && _76) || (_87 && _76)) || (_87 && _84))))
    {
        rayQueryInitializeEXT(_46, _47, _48._m0, _48._m1, _48._m4, _48._m2, _48._m5, _48._m3);
        _49 = 1u;
    }
    else
    {
    }
}

bool _145(rayQueryEXT _146, inout uint _147)
{
    bool _149 = false;
    if ((!((_147 & 4u) != 0u)) && ((_147 & 1u) != 0u))
    {
        bool _160 = rayQueryProceedEXT(_146);
        _149 = _160;
        _147 |= (_160 ? 2u : 6u);
    }
    return _149;
}

void main()
{
    uvec2 _40 = uvec2(0u);
    uint _36 = 0u;
    float _38 = 0.0;
    rayQueryEXT _33;
    _45(_33, _11, _8(4u, 255u, 34.0, 38.0, vec3(46.0), vec3(58.0, 62.0, 74.0)), _36, _38);
    _40 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_40, uvec2(0u))))
        {
            break;
        }
        _40 -= uvec2(uint(_40.y == 0u), 1u);
        bool _142 = _145(_33, _36);
        if (!_142)
        {
            break;
        }
        continue;
    }
}

