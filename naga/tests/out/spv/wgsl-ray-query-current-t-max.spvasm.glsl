#version 460
#extension GL_EXT_ray_query : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _10
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

layout(set = 0, binding = 1, std430) buffer _14_13
{
    float _m0[2];
} _13;

layout(set = 0, binding = 0) uniform accelerationStructureEXT _11;

void _39(rayQueryEXT _40, accelerationStructureEXT _41, _10 _42, inout uint _43, out float _44)
{
    bool _70 = (_42._m0 & 256u) != 0u;
    bool _78 = (_42._m0 & 16u) != 0u;
    bool _81 = (_42._m0 & 32u) != 0u;
    bool _90 = (_42._m0 & 1u) != 0u;
    bool _92 = (_42._m0 & 2u) != 0u;
    bool _95 = (_42._m0 & 64u) != 0u;
    bool _98 = (_42._m0 & 128u) != 0u;
    _44 = _42._m3;
    if (((((((!((((((_92 && _90) || (_98 && _90)) || (_98 && _92)) || (_98 && _95)) || (_95 && _90)) || (_95 && _92))) && (_42._m2 <= _42._m3)) && (_42._m2 >= 0.0)) && (!(any(isnan(_42._m4)) || any(isinf(_42._m4))))) && (!(any(isnan(_42._m5)) || any(isinf(_42._m5))))) && (!(((_42._m0 & 512u) != 0u) && _70))) && (!(((_78 && _70) || (_81 && _70)) || (_81 && _78))))
    {
        rayQueryInitializeEXT(_40, _41, _42._m0, _42._m1, _42._m4, _42._m2, _42._m5, _42._m3);
        _43 = 1u;
    }
    else
    {
    }
}

float _123(rayQueryEXT _124, uint _125, float _126)
{
    float _128 = 0.0;
    if ((_125 & 1u) != 0u)
    {
        uint _134 = rayQueryGetIntersectionTypeEXT(_124, bool(1u));
        if (_134 == 0u)
        {
            _128 = _126;
        }
        else
        {
            float _140 = rayQueryGetIntersectionTEXT(_124, bool(1u));
            _128 = _140;
        }
    }
    return _128;
}

bool _168(rayQueryEXT _169, inout uint _170)
{
    bool _172 = false;
    if ((!((_170 & 4u) != 0u)) && ((_170 & 1u) != 0u))
    {
        bool _184 = rayQueryProceedEXT(_169);
        _172 = _184;
        _170 |= (_184 ? 2u : 6u);
    }
    return _172;
}

void _198(rayQueryEXT _199, uint _200, float _201, float _202)
{
    if ((!((_200 & 4u) != 0u)) && ((_200 & 2u) != 0u))
    {
        uint _215 = rayQueryGetIntersectionTypeEXT(_199, bool(0u));
        float _217 = rayQueryGetRayTMinEXT(_199);
        uint _218 = rayQueryGetIntersectionTypeEXT(_199, bool(1u));
        float _205;
        if (_218 == 0u)
        {
            _205 = _202;
        }
        else
        {
            float _224 = rayQueryGetIntersectionTEXT(_199, bool(0u));
            _205 = _224;
        }
        if (((_201 >= _217) && (_201 <= _205)) && (_215 == 1u))
        {
            rayQueryGenerateIntersectionEXT(_199, _201);
        }
    }
}

float _20(vec3 _17, vec3 _18, float _19)
{
    uint _32 = 0u;
    float _34 = 0.0;
    uvec2 _154 = uvec2(4294967295u);
    rayQueryEXT _29;
    _39(_29, _11, _10(0u, 255u, 0.100000001490116119384765625, 100.0, _17, _18), _32, _34);
    _13._m0[0u] = _123(_29, _32, _34);
    for (;;)
    {
        if (all(equal(uvec2(0u), _154)))
        {
            break;
        }
        _154 -= uvec2(uint(_154.y == 0u), 1u);
        bool _165 = _168(_29, _32);
        if (!_165)
        {
            break;
        }
        if (_19 <= _123(_29, _32, _34))
        {
            _198(_29, _32, _19, _34);
        }
        continue;
    }
    return _123(_29, _32, _34);
}

void main()
{
    float _244 = _20(vec3(0.0), vec3(0.0, 1.0, 0.0), 2.0);
    _13._m0[1u] = _244;
}

