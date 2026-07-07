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

void _37(rayQueryEXT _38, accelerationStructureEXT _39, _8 _40, inout uint _41, out float _42)
{
    bool _68 = (_40._m0 & 256u) != 0u;
    bool _76 = (_40._m0 & 16u) != 0u;
    bool _79 = (_40._m0 & 32u) != 0u;
    bool _88 = (_40._m0 & 1u) != 0u;
    bool _91 = (_40._m0 & 2u) != 0u;
    bool _94 = (_40._m0 & 64u) != 0u;
    bool _97 = (_40._m0 & 128u) != 0u;
    _42 = _40._m3;
    if (((((((!((((((_91 && _88) || (_97 && _88)) || (_97 && _91)) || (_97 && _94)) || (_94 && _88)) || (_94 && _91))) && (_40._m2 <= _40._m3)) && (_40._m2 >= 0.0)) && (!(any(isnan(_40._m4)) || any(isinf(_40._m4))))) && (!(any(isnan(_40._m5)) || any(isinf(_40._m5))))) && (!(((_40._m0 & 512u) != 0u) && _68))) && (!(((_76 && _68) || (_79 && _68)) || (_79 && _76))))
    {
        rayQueryInitializeEXT(_38, _39, _40._m0, _40._m1, _40._m4, _40._m2, _40._m5, _40._m3);
        _41 = 1u;
    }
    else
    {
    }
}

bool _144(rayQueryEXT _145, inout uint _146)
{
    bool _148 = false;
    if ((!((_146 & 4u) != 0u)) && ((_146 & 1u) != 0u))
    {
        bool _159 = rayQueryProceedEXT(_145);
        _148 = _159;
        _146 |= (_159 ? 2u : 6u);
    }
    return _148;
}

void main()
{
    uint _30 = 0u;
    float _33 = 0.0;
    uvec2 _130 = uvec2(4294967295u);
    rayQueryEXT _27;
    _37(_27, _10, _8(4u, 255u, 34.0, 38.0, vec3(46.0), vec3(58.0, 62.0, 74.0)), _30, _33);
    for (;;)
    {
        if (all(equal(uvec2(0u), _130)))
        {
            break;
        }
        _130 -= uvec2(uint(_130.y == 0u), 1u);
        bool _141 = _144(_27, _30);
        if (!_141)
        {
            break;
        }
        continue;
    }
}

