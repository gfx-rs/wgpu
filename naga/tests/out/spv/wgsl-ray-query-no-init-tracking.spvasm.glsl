////////////////////////////////
// Entry point: "main" (comp) //
////////////////////////////////
#version 460
#extension GL_EXT_ray_query : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _10
{
    uint _m0;
    float _m1;
    uint _m2;
    uint _m3;
    uint _m4;
    uint _m5;
    uint _m6;
    vec2 _m7;
    bool _m8;
    mat4x3 _m9;
    mat4x3 _m10;
};

struct _12
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

struct _13
{
    uint _m0;
    vec3 _m1;
};

layout(set = 0, binding = 1, std430) buffer _18_17
{
    _13 _m0;
} _17;

layout(set = 0, binding = 0) uniform accelerationStructureEXT _15;

void _42(rayQueryEXT _43, accelerationStructureEXT _44, _12 _45, inout uint _46, out float _47)
{
    bool _72 = (_45._m0 & 256u) != 0u;
    bool _80 = (_45._m0 & 16u) != 0u;
    bool _83 = (_45._m0 & 32u) != 0u;
    bool _92 = (_45._m0 & 1u) != 0u;
    bool _95 = (_45._m0 & 2u) != 0u;
    bool _98 = (_45._m0 & 64u) != 0u;
    bool _101 = (_45._m0 & 128u) != 0u;
    _47 = _45._m3;
    if (((((((!((((((_95 && _92) || (_101 && _92)) || (_101 && _95)) || (_101 && _98)) || (_98 && _92)) || (_98 && _95))) && (_45._m2 <= _45._m3)) && (_45._m2 >= 0.0)) && (!(any(isnan(_45._m4)) || any(isinf(_45._m4))))) && (!(any(isnan(_45._m5)) || any(isinf(_45._m5))))) && (!(((_45._m0 & 512u) != 0u) && _72))) && (!(((_80 && _72) || (_83 && _72)) || (_83 && _80))))
    {
        rayQueryInitializeEXT(_43, _44, _45._m0, _45._m1, _45._m4, _45._m2, _45._m5, _45._m3);
        _46 = 1u;
    }
    else
    {
    }
}

bool _148(rayQueryEXT _149, inout uint _150)
{
    bool _152 = false;
    if ((!((_150 & 4u) != 0u)) && ((_150 & 1u) != 0u))
    {
        bool _163 = rayQueryProceedEXT(_149);
        _152 = _163;
        _150 |= (_163 ? 2u : 6u);
    }
    return _152;
}

_10 _176(rayQueryEXT _177, uint _178)
{
    _10 _181 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if (((_178 & 4u) != 0u) && ((_178 & 2u) != 0u))
    {
        uint _190 = rayQueryGetIntersectionTypeEXT(_177, bool(1u));
        _181._m0 = _190;
        if (_190 != 0u)
        {
            uint _195 = rayQueryGetIntersectionInstanceCustomIndexEXT(_177, bool(1u));
            uint _196 = rayQueryGetIntersectionInstanceIdEXT(_177, bool(1u));
            uint _197 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_177, bool(1u));
            uint _198 = rayQueryGetIntersectionGeometryIndexEXT(_177, bool(1u));
            uint _199 = rayQueryGetIntersectionPrimitiveIndexEXT(_177, bool(1u));
            mat4x3 _200 = rayQueryGetIntersectionObjectToWorldEXT(_177, bool(1u));
            mat4x3 _201 = rayQueryGetIntersectionWorldToObjectEXT(_177, bool(1u));
            _181._m2 = _195;
            _181._m3 = _196;
            _181._m4 = _197;
            _181._m5 = _198;
            _181._m6 = _199;
            _181._m9 = _200;
            _181._m10 = _201;
            float _216 = rayQueryGetIntersectionTEXT(_177, bool(1u));
            _181._m1 = _216;
            if (_190 == 1u)
            {
                vec2 _218 = rayQueryGetIntersectionBarycentricsEXT(_177, bool(1u));
                bool _219 = rayQueryGetIntersectionFrontFaceEXT(_177, bool(1u));
                _181._m7 = _218;
                _181._m8 = _219;
            }
        }
    }
    return _181;
}

_10 _25(vec3 _21, vec3 _22, accelerationStructureEXT _23)
{
    uint _34 = 0u;
    float _37 = 0.0;
    uvec2 _134 = uvec2(4294967295u);
    rayQueryEXT _31;
    _42(_31, _23, _12(4u, 255u, 0.100000001490116119384765625, 100.0, _21, _22), _34, _37);
    for (;;)
    {
        if (all(equal(uvec2(0u), _134)))
        {
            break;
        }
        _134 -= uvec2(uint(_134.y == 0u), 1u);
        bool _145 = _148(_31, _34);
        if (!_145)
        {
            break;
        }
        continue;
    }
    return _176(_31, _34);
}

vec3 _229(vec3 _227, _10 _228)
{
    return normalize(_227 - (_228._m9 * vec4(normalize((_228._m10 * vec4(_227, 1.0)).xy) * 2.400000095367431640625, 0.0, 1.0)));
}

void main()
{
    _10 _254 = _25(vec3(0.0), vec3(0.0, 1.0, 0.0), _15);
    _17._m0._m0 = uint(_254._m0 == 0u);
    _17._m0._m1 = _229(vec3(0.0, 1.0, 0.0) * _254._m1, _254);
}


//////////////////////////////////////////
// Entry point: "main_candidate" (comp) //
//////////////////////////////////////////
#version 460
#extension GL_EXT_ray_query : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _10
{
    uint _m0;
    float _m1;
    uint _m2;
    uint _m3;
    uint _m4;
    uint _m5;
    uint _m6;
    vec2 _m7;
    bool _m8;
    mat4x3 _m9;
    mat4x3 _m10;
};

struct _12
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

struct _13
{
    uint _m0;
    vec3 _m1;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT _15;

void _42(rayQueryEXT _43, accelerationStructureEXT _44, _12 _45, inout uint _46, out float _47)
{
    bool _72 = (_45._m0 & 256u) != 0u;
    bool _80 = (_45._m0 & 16u) != 0u;
    bool _83 = (_45._m0 & 32u) != 0u;
    bool _92 = (_45._m0 & 1u) != 0u;
    bool _95 = (_45._m0 & 2u) != 0u;
    bool _98 = (_45._m0 & 64u) != 0u;
    bool _101 = (_45._m0 & 128u) != 0u;
    _47 = _45._m3;
    if (((((((!((((((_95 && _92) || (_101 && _92)) || (_101 && _95)) || (_101 && _98)) || (_98 && _92)) || (_98 && _95))) && (_45._m2 <= _45._m3)) && (_45._m2 >= 0.0)) && (!(any(isnan(_45._m4)) || any(isinf(_45._m4))))) && (!(any(isnan(_45._m5)) || any(isinf(_45._m5))))) && (!(((_45._m0 & 512u) != 0u) && _72))) && (!(((_80 && _72) || (_83 && _72)) || (_83 && _80))))
    {
        rayQueryInitializeEXT(_43, _44, _45._m0, _45._m1, _45._m4, _45._m2, _45._m5, _45._m3);
        _46 = 1u;
    }
    else
    {
    }
}

_10 _275(rayQueryEXT _276, uint _277)
{
    _10 _279 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if ((!((_277 & 4u) != 0u)) && ((_277 & 2u) != 0u))
    {
        uint _289 = rayQueryGetIntersectionTypeEXT(_276, bool(0u));
        uint _291 = (_289 == 0u) ? 1u : 3u;
        _279._m0 = _291;
        if (_291 != 0u)
        {
            uint _296 = rayQueryGetIntersectionInstanceCustomIndexEXT(_276, bool(0u));
            uint _297 = rayQueryGetIntersectionInstanceIdEXT(_276, bool(0u));
            uint _298 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_276, bool(0u));
            uint _299 = rayQueryGetIntersectionGeometryIndexEXT(_276, bool(0u));
            uint _300 = rayQueryGetIntersectionPrimitiveIndexEXT(_276, bool(0u));
            mat4x3 _301 = rayQueryGetIntersectionObjectToWorldEXT(_276, bool(0u));
            mat4x3 _302 = rayQueryGetIntersectionWorldToObjectEXT(_276, bool(0u));
            _279._m2 = _296;
            _279._m3 = _297;
            _279._m4 = _298;
            _279._m5 = _299;
            _279._m6 = _300;
            _279._m9 = _301;
            _279._m10 = _302;
            if (_291 == 1u)
            {
                float _313 = rayQueryGetIntersectionTEXT(_276, bool(0u));
                _279._m1 = _313;
                vec2 _315 = rayQueryGetIntersectionBarycentricsEXT(_276, bool(0u));
                bool _316 = rayQueryGetIntersectionFrontFaceEXT(_276, bool(0u));
                _279._m7 = _315;
                _279._m8 = _316;
            }
        }
    }
    return _279;
}

void _327(rayQueryEXT _328, uint _329, float _330, float _331)
{
    if ((!((_329 & 4u) != 0u)) && ((_329 & 2u) != 0u))
    {
        uint _344 = rayQueryGetIntersectionTypeEXT(_328, bool(0u));
        float _346 = rayQueryGetRayTMinEXT(_328);
        uint _347 = rayQueryGetIntersectionTypeEXT(_328, bool(1u));
        float _334;
        if (_347 == 0u)
        {
            _334 = _331;
        }
        else
        {
            float _353 = rayQueryGetIntersectionTEXT(_328, bool(0u));
            _334 = _353;
        }
        if (((_330 >= _346) && (_330 <= _334)) && (_344 == 1u))
        {
            rayQueryGenerateIntersectionEXT(_328, _330);
        }
    }
}

void _368(rayQueryEXT _369, uint _370)
{
    if ((!((_370 & 4u) != 0u)) && ((_370 & 2u) != 0u))
    {
        uint _381 = rayQueryGetIntersectionTypeEXT(_369, bool(0u));
        if (_381 == 0u)
        {
            rayQueryConfirmIntersectionEXT(_369);
        }
    }
}

void _387(rayQueryEXT _388, uint _389)
{
    if ((!((_389 & 4u) != 0u)) && ((_389 & 2u) != 0u))
    {
        rayQueryTerminateEXT(_388);
    }
}

void main()
{
    uint _271 = 0u;
    float _272 = 0.0;
    rayQueryEXT _270;
    _42(_270, _15, _12(4u, 255u, 0.100000001490116119384765625, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), _271, _272);
    _10 _320 = _275(_270, _271);
    if (_320._m0 == 3u)
    {
        _327(_270, _271, 10.0, _272);
        return;
    }
    else
    {
        if (_320._m0 == 1u)
        {
            _368(_270, _271);
            return;
        }
        else
        {
            _387(_270, _271);
            return;
        }
    }
}

