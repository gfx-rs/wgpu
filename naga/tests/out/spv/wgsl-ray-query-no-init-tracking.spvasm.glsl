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
    _10 _259 = _25(vec3(0.0), vec3(0.0, 1.0, 0.0), _15);
    _17._m0._m0 = uint(_259._m0 == 0u);
    _17._m0._m1 = _229(vec3(0.0, 1.0, 0.0) * _259._m1, _259);
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

_10 _280(rayQueryEXT _281, uint _282)
{
    _10 _284 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if ((!((_282 & 4u) != 0u)) && ((_282 & 2u) != 0u))
    {
        uint _294 = rayQueryGetIntersectionTypeEXT(_281, bool(0u));
        uint _296 = (_294 == 0u) ? 1u : 3u;
        _284._m0 = _296;
        if (_296 != 0u)
        {
            uint _301 = rayQueryGetIntersectionInstanceCustomIndexEXT(_281, bool(0u));
            uint _302 = rayQueryGetIntersectionInstanceIdEXT(_281, bool(0u));
            uint _303 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_281, bool(0u));
            uint _304 = rayQueryGetIntersectionGeometryIndexEXT(_281, bool(0u));
            uint _305 = rayQueryGetIntersectionPrimitiveIndexEXT(_281, bool(0u));
            mat4x3 _306 = rayQueryGetIntersectionObjectToWorldEXT(_281, bool(0u));
            mat4x3 _307 = rayQueryGetIntersectionWorldToObjectEXT(_281, bool(0u));
            _284._m2 = _301;
            _284._m3 = _302;
            _284._m4 = _303;
            _284._m5 = _304;
            _284._m6 = _305;
            _284._m9 = _306;
            _284._m10 = _307;
            if (_296 == 1u)
            {
                float _318 = rayQueryGetIntersectionTEXT(_281, bool(0u));
                _284._m1 = _318;
                vec2 _320 = rayQueryGetIntersectionBarycentricsEXT(_281, bool(0u));
                bool _321 = rayQueryGetIntersectionFrontFaceEXT(_281, bool(0u));
                _284._m7 = _320;
                _284._m8 = _321;
            }
        }
    }
    return _284;
}

void _332(rayQueryEXT _333, uint _334, float _335, float _336)
{
    if ((!((_334 & 4u) != 0u)) && ((_334 & 2u) != 0u))
    {
        uint _349 = rayQueryGetIntersectionTypeEXT(_333, bool(0u));
        float _351 = rayQueryGetRayTMinEXT(_333);
        uint _352 = rayQueryGetIntersectionTypeEXT(_333, bool(1u));
        float _339;
        if (_352 == 0u)
        {
            _339 = _336;
        }
        else
        {
            float _358 = rayQueryGetIntersectionTEXT(_333, bool(1u));
            _339 = _358;
        }
        if (((_335 >= _351) && (_335 <= _339)) && (_349 == 1u))
        {
            rayQueryGenerateIntersectionEXT(_333, _335);
        }
    }
}

void _373(rayQueryEXT _374, uint _375)
{
    if ((!((_375 & 4u) != 0u)) && ((_375 & 2u) != 0u))
    {
        uint _386 = rayQueryGetIntersectionTypeEXT(_374, bool(0u));
        if (_386 == 0u)
        {
            rayQueryConfirmIntersectionEXT(_374);
        }
    }
}

void _392(rayQueryEXT _393, uint _394)
{
    if ((!((_394 & 4u) != 0u)) && ((_394 & 2u) != 0u))
    {
        rayQueryTerminateEXT(_393);
    }
}

void main()
{
    uint _276 = 0u;
    float _277 = 0.0;
    rayQueryEXT _275;
    _42(_275, _15, _12(4u, 255u, 0.100000001490116119384765625, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), _276, _277);
    _10 _325 = _280(_275, _276);
    if (_325._m0 == 3u)
    {
        _332(_275, _276, 10.0, _277);
        return;
    }
    else
    {
        if (_325._m0 == 1u)
        {
            _373(_275, _276);
            return;
        }
        else
        {
            _392(_275, _276);
            return;
        }
    }
}

