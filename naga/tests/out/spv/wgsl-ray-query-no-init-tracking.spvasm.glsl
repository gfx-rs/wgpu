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

layout(set = 0, binding = 1, std430) buffer _19_18
{
    _13 _m0;
} _18;

layout(set = 0, binding = 0) uniform accelerationStructureEXT _16;

void _49(rayQueryEXT _50, accelerationStructureEXT _51, _12 _52, inout uint _53, out float _54)
{
    bool _79 = (_52._m0 & 256u) != 0u;
    bool _87 = (_52._m0 & 16u) != 0u;
    bool _90 = (_52._m0 & 32u) != 0u;
    bool _98 = (_52._m0 & 1u) != 0u;
    bool _101 = (_52._m0 & 2u) != 0u;
    bool _104 = (_52._m0 & 64u) != 0u;
    bool _107 = (_52._m0 & 128u) != 0u;
    _54 = _52._m3;
    if (((((((!((((((_101 && _98) || (_107 && _98)) || (_107 && _101)) || (_107 && _104)) || (_104 && _98)) || (_104 && _101))) && (_52._m2 <= _52._m3)) && (_52._m2 >= 0.0)) && (!(any(isnan(_52._m4)) || any(isinf(_52._m4))))) && (!(any(isnan(_52._m5)) || any(isinf(_52._m5))))) && (!(((_52._m0 & 512u) != 0u) && _79))) && (!(((_87 && _79) || (_90 && _79)) || (_90 && _87))))
    {
        rayQueryInitializeEXT(_50, _51, _52._m0, _52._m1, _52._m4, _52._m2, _52._m5, _52._m3);
        _53 = 1u;
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

_10 _178(rayQueryEXT _179, uint _180)
{
    _10 _183 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if (((_180 & 4u) != 0u) && ((_180 & 2u) != 0u))
    {
        uint _192 = rayQueryGetIntersectionTypeEXT(_179, bool(1u));
        _183._m0 = _192;
        if (_192 != 0u)
        {
            uint _197 = rayQueryGetIntersectionInstanceCustomIndexEXT(_179, bool(1u));
            uint _198 = rayQueryGetIntersectionInstanceIdEXT(_179, bool(1u));
            uint _199 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_179, bool(1u));
            uint _200 = rayQueryGetIntersectionGeometryIndexEXT(_179, bool(1u));
            uint _201 = rayQueryGetIntersectionPrimitiveIndexEXT(_179, bool(1u));
            mat4x3 _202 = rayQueryGetIntersectionObjectToWorldEXT(_179, bool(1u));
            mat4x3 _203 = rayQueryGetIntersectionWorldToObjectEXT(_179, bool(1u));
            _183._m2 = _197;
            _183._m3 = _198;
            _183._m4 = _199;
            _183._m5 = _200;
            _183._m6 = _201;
            _183._m9 = _202;
            _183._m10 = _203;
            float _218 = rayQueryGetIntersectionTEXT(_179, bool(1u));
            _183._m1 = _218;
            if (_192 == 1u)
            {
                vec2 _220 = rayQueryGetIntersectionBarycentricsEXT(_179, bool(1u));
                bool _221 = rayQueryGetIntersectionFrontFaceEXT(_179, bool(1u));
                _183._m7 = _220;
                _183._m8 = _221;
            }
        }
    }
    return _183;
}

_10 _26(vec3 _22, vec3 _23, accelerationStructureEXT _24)
{
    uvec2 _44 = uvec2(4294967295u);
    uint _40 = 0u;
    float _42 = 0.0;
    rayQueryEXT _37;
    _49(_37, _24, _12(4u, 255u, 0.100000001490116119384765625, 100.0, _22, _23), _40, _42);
    for (;;)
    {
        uvec2 _136 = _44;
        _44 = _136 - uvec2(uint(_136.y == 0u), 1u);
        if (all(equal(_136, uvec2(0u))))
        {
            break;
        }
        bool _147 = _150(_37, _40);
        if (!_147)
        {
            break;
        }
        continue;
    }
    return _178(_37, _40);
}

vec3 _231(vec3 _229, _10 _230)
{
    return normalize(_229 - (_230._m9 * vec4(normalize((_230._m10 * vec4(_229, 1.0)).xy) * 2.400000095367431640625, 0.0, 1.0)));
}

void main()
{
    _10 _256 = _26(vec3(0.0), vec3(0.0, 1.0, 0.0), _16);
    _18._m0._m0 = uint(_256._m0 == 0u);
    _18._m0._m1 = _231(vec3(0.0, 1.0, 0.0) * _256._m1, _256);
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

layout(set = 0, binding = 0) uniform accelerationStructureEXT _16;

void _49(rayQueryEXT _50, accelerationStructureEXT _51, _12 _52, inout uint _53, out float _54)
{
    bool _79 = (_52._m0 & 256u) != 0u;
    bool _87 = (_52._m0 & 16u) != 0u;
    bool _90 = (_52._m0 & 32u) != 0u;
    bool _98 = (_52._m0 & 1u) != 0u;
    bool _101 = (_52._m0 & 2u) != 0u;
    bool _104 = (_52._m0 & 64u) != 0u;
    bool _107 = (_52._m0 & 128u) != 0u;
    _54 = _52._m3;
    if (((((((!((((((_101 && _98) || (_107 && _98)) || (_107 && _101)) || (_107 && _104)) || (_104 && _98)) || (_104 && _101))) && (_52._m2 <= _52._m3)) && (_52._m2 >= 0.0)) && (!(any(isnan(_52._m4)) || any(isinf(_52._m4))))) && (!(any(isnan(_52._m5)) || any(isinf(_52._m5))))) && (!(((_52._m0 & 512u) != 0u) && _79))) && (!(((_87 && _79) || (_90 && _79)) || (_90 && _87))))
    {
        rayQueryInitializeEXT(_50, _51, _52._m0, _52._m1, _52._m4, _52._m2, _52._m5, _52._m3);
        _53 = 1u;
    }
    else
    {
    }
}

_10 _277(rayQueryEXT _278, uint _279)
{
    _10 _281 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if ((!((_279 & 4u) != 0u)) && ((_279 & 2u) != 0u))
    {
        uint _291 = rayQueryGetIntersectionTypeEXT(_278, bool(0u));
        uint _293 = (_291 == 0u) ? 1u : 3u;
        _281._m0 = _293;
        if (_293 != 0u)
        {
            uint _298 = rayQueryGetIntersectionInstanceCustomIndexEXT(_278, bool(0u));
            uint _299 = rayQueryGetIntersectionInstanceIdEXT(_278, bool(0u));
            uint _300 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_278, bool(0u));
            uint _301 = rayQueryGetIntersectionGeometryIndexEXT(_278, bool(0u));
            uint _302 = rayQueryGetIntersectionPrimitiveIndexEXT(_278, bool(0u));
            mat4x3 _303 = rayQueryGetIntersectionObjectToWorldEXT(_278, bool(0u));
            mat4x3 _304 = rayQueryGetIntersectionWorldToObjectEXT(_278, bool(0u));
            _281._m2 = _298;
            _281._m3 = _299;
            _281._m4 = _300;
            _281._m5 = _301;
            _281._m6 = _302;
            _281._m9 = _303;
            _281._m10 = _304;
            if (_293 == 1u)
            {
                float _315 = rayQueryGetIntersectionTEXT(_278, bool(0u));
                _281._m1 = _315;
                vec2 _317 = rayQueryGetIntersectionBarycentricsEXT(_278, bool(0u));
                bool _318 = rayQueryGetIntersectionFrontFaceEXT(_278, bool(0u));
                _281._m7 = _317;
                _281._m8 = _318;
            }
        }
    }
    return _281;
}

void _329(rayQueryEXT _330, uint _331, float _332, float _333)
{
    if ((!((_331 & 4u) != 0u)) && ((_331 & 2u) != 0u))
    {
        uint _346 = rayQueryGetIntersectionTypeEXT(_330, bool(0u));
        float _348 = rayQueryGetRayTMinEXT(_330);
        uint _349 = rayQueryGetIntersectionTypeEXT(_330, bool(1u));
        float _336;
        if (_349 == 0u)
        {
            _336 = _333;
        }
        else
        {
            float _355 = rayQueryGetIntersectionTEXT(_330, bool(1u));
            _336 = _355;
        }
        if (((_332 >= _348) && (_332 <= _336)) && (_346 == 1u))
        {
            rayQueryGenerateIntersectionEXT(_330, _332);
        }
    }
}

void _370(rayQueryEXT _371, uint _372)
{
    if ((!((_372 & 4u) != 0u)) && ((_372 & 2u) != 0u))
    {
        uint _383 = rayQueryGetIntersectionTypeEXT(_371, bool(0u));
        if (_383 == 0u)
        {
            rayQueryConfirmIntersectionEXT(_371);
        }
    }
}

void _389(rayQueryEXT _390, uint _391)
{
    if ((!((_391 & 4u) != 0u)) && ((_391 & 2u) != 0u))
    {
        rayQueryTerminateEXT(_390);
    }
}

void main()
{
    uint _273 = 0u;
    float _274 = 0.0;
    rayQueryEXT _272;
    _49(_272, _16, _12(4u, 255u, 0.100000001490116119384765625, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), _273, _274);
    _10 _322 = _277(_272, _273);
    if (_322._m0 == 3u)
    {
        _329(_272, _273, 10.0, _274);
        return;
    }
    else
    {
        if (_322._m0 == 1u)
        {
            _370(_272, _273);
            return;
        }
        else
        {
            _389(_272, _273);
            return;
        }
    }
}

