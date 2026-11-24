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

void _44(rayQueryEXT _45, accelerationStructureEXT _46, _8 _47, inout uint _48, out float _49)
{
    bool _75 = (_47._m0 & 256u) != 0u;
    bool _83 = (_47._m0 & 16u) != 0u;
    bool _86 = (_47._m0 & 32u) != 0u;
    bool _94 = (_47._m0 & 1u) != 0u;
    bool _97 = (_47._m0 & 2u) != 0u;
    bool _100 = (_47._m0 & 64u) != 0u;
    bool _103 = (_47._m0 & 128u) != 0u;
    _49 = _47._m3;
    if (((((((!((((((_97 && _94) || (_103 && _94)) || (_103 && _97)) || (_103 && _100)) || (_100 && _94)) || (_100 && _97))) && (_47._m2 <= _47._m3)) && (_47._m2 >= 0.0)) && (!(any(isnan(_47._m4)) || any(isinf(_47._m4))))) && (!(any(isnan(_47._m5)) || any(isinf(_47._m5))))) && (!(((_47._m0 & 512u) != 0u) && _75))) && (!(((_83 && _75) || (_86 && _75)) || (_86 && _83))))
    {
        rayQueryInitializeEXT(_45, _46, _47._m0, _47._m1, _47._m4, _47._m2, _47._m5, _47._m3);
        _48 = 1u;
    }
    else
    {
    }
}

bool _146(rayQueryEXT _147, inout uint _148)
{
    bool _150 = false;
    if ((!((_148 & 4u) != 0u)) && ((_148 & 1u) != 0u))
    {
        bool _161 = rayQueryProceedEXT(_147);
        _150 = _161;
        _148 |= (_161 ? 2u : 6u);
    }
    return _150;
}

void main()
{
    uvec2 _40 = uvec2(4294967295u);
    uint _36 = 0u;
    float _38 = 0.0;
    rayQueryEXT _33;
    _44(_33, _11, _8(4u, 255u, 34.0, 38.0, vec3(46.0), vec3(58.0, 62.0, 74.0)), _36, _38);
    for (;;)
    {
        uvec2 _132 = _40;
        _40 = _132 - uvec2(uint(_132.y == 0u), 1u);
        if (all(equal(_132, uvec2(0u))))
        {
            break;
        }
        bool _143 = _146(_33, _36);
        if (!_143)
        {
            break;
        }
        continue;
    }
}

