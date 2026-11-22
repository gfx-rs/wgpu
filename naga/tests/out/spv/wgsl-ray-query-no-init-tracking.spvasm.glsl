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

void _50(rayQueryEXT _51, accelerationStructureEXT _52, _12 _53, inout uint _54, out float _55)
{
    bool _80 = (_53._m0 & 256u) != 0u;
    bool _88 = (_53._m0 & 16u) != 0u;
    bool _91 = (_53._m0 & 32u) != 0u;
    bool _99 = (_53._m0 & 1u) != 0u;
    bool _102 = (_53._m0 & 2u) != 0u;
    bool _105 = (_53._m0 & 64u) != 0u;
    bool _108 = (_53._m0 & 128u) != 0u;
    _55 = _53._m3;
    if (((((((!((((((_102 && _99) || (_108 && _99)) || (_108 && _102)) || (_108 && _105)) || (_105 && _99)) || (_105 && _102))) && (_53._m2 <= _53._m3)) && (_53._m2 >= 0.0)) && (!(any(isnan(_53._m4)) || any(isinf(_53._m4))))) && (!(any(isnan(_53._m5)) || any(isinf(_53._m5))))) && (!(((_53._m0 & 512u) != 0u) && _80))) && (!(((_88 && _80) || (_91 && _80)) || (_91 && _88))))
    {
        rayQueryInitializeEXT(_51, _52, _53._m0, _53._m1, _53._m4, _53._m2, _53._m5, _53._m3);
        _54 = 1u;
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

_10 _177(rayQueryEXT _178, uint _179)
{
    _10 _182 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if (((_179 & 4u) != 0u) && ((_179 & 2u) != 0u))
    {
        uint _191 = rayQueryGetIntersectionTypeEXT(_178, bool(1u));
        _182._m0 = _191;
        if (_191 != 0u)
        {
            uint _196 = rayQueryGetIntersectionInstanceCustomIndexEXT(_178, bool(1u));
            uint _197 = rayQueryGetIntersectionInstanceIdEXT(_178, bool(1u));
            uint _198 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_178, bool(1u));
            uint _199 = rayQueryGetIntersectionGeometryIndexEXT(_178, bool(1u));
            uint _200 = rayQueryGetIntersectionPrimitiveIndexEXT(_178, bool(1u));
            mat4x3 _201 = rayQueryGetIntersectionObjectToWorldEXT(_178, bool(1u));
            mat4x3 _202 = rayQueryGetIntersectionWorldToObjectEXT(_178, bool(1u));
            _182._m2 = _196;
            _182._m3 = _197;
            _182._m4 = _198;
            _182._m5 = _199;
            _182._m6 = _200;
            _182._m9 = _201;
            _182._m10 = _202;
            float _217 = rayQueryGetIntersectionTEXT(_178, bool(1u));
            _182._m1 = _217;
            if (_191 == 1u)
            {
                vec2 _219 = rayQueryGetIntersectionBarycentricsEXT(_178, bool(1u));
                bool _220 = rayQueryGetIntersectionFrontFaceEXT(_178, bool(1u));
                _182._m7 = _219;
                _182._m8 = _220;
            }
        }
    }
    return _182;
}

_10 _26(vec3 _22, vec3 _23, accelerationStructureEXT _24)
{
    uvec2 _44 = uvec2(0u);
    uint _40 = 0u;
    float _42 = 0.0;
    rayQueryEXT _37;
    _50(_37, _24, _12(4u, 255u, 0.100000001490116119384765625, 100.0, _22, _23), _40, _42);
    _44 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_44, uvec2(0u))))
        {
            break;
        }
        _44 -= uvec2(uint(_44.y == 0u), 1u);
        bool _146 = _149(_37, _40);
        if (!_146)
        {
            break;
        }
        continue;
    }
    return _177(_37, _40);
}

vec3 _230(vec3 _228, _10 _229)
{
    return normalize(_228 - (_229._m9 * vec4(normalize((_229._m10 * vec4(_228, 1.0)).xy) * 2.400000095367431640625, 0.0, 1.0)));
}

void main()
{
    _10 _255 = _26(vec3(0.0), vec3(0.0, 1.0, 0.0), _16);
    _18._m0._m0 = uint(_255._m0 == 0u);
    _18._m0._m1 = _230(vec3(0.0, 1.0, 0.0) * _255._m1, _255);
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

void _50(rayQueryEXT _51, accelerationStructureEXT _52, _12 _53, inout uint _54, out float _55)
{
    bool _80 = (_53._m0 & 256u) != 0u;
    bool _88 = (_53._m0 & 16u) != 0u;
    bool _91 = (_53._m0 & 32u) != 0u;
    bool _99 = (_53._m0 & 1u) != 0u;
    bool _102 = (_53._m0 & 2u) != 0u;
    bool _105 = (_53._m0 & 64u) != 0u;
    bool _108 = (_53._m0 & 128u) != 0u;
    _55 = _53._m3;
    if (((((((!((((((_102 && _99) || (_108 && _99)) || (_108 && _102)) || (_108 && _105)) || (_105 && _99)) || (_105 && _102))) && (_53._m2 <= _53._m3)) && (_53._m2 >= 0.0)) && (!(any(isnan(_53._m4)) || any(isinf(_53._m4))))) && (!(any(isnan(_53._m5)) || any(isinf(_53._m5))))) && (!(((_53._m0 & 512u) != 0u) && _80))) && (!(((_88 && _80) || (_91 && _80)) || (_91 && _88))))
    {
        rayQueryInitializeEXT(_51, _52, _53._m0, _53._m1, _53._m4, _53._m2, _53._m5, _53._m3);
        _54 = 1u;
    }
    else
    {
    }
}

_10 _276(rayQueryEXT _277, uint _278)
{
    _10 _280 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if ((!((_278 & 4u) != 0u)) && ((_278 & 2u) != 0u))
    {
        uint _290 = rayQueryGetIntersectionTypeEXT(_277, bool(0u));
        uint _292 = (_290 == 0u) ? 1u : 3u;
        _280._m0 = _292;
        if (_292 != 0u)
        {
            uint _297 = rayQueryGetIntersectionInstanceCustomIndexEXT(_277, bool(0u));
            uint _298 = rayQueryGetIntersectionInstanceIdEXT(_277, bool(0u));
            uint _299 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_277, bool(0u));
            uint _300 = rayQueryGetIntersectionGeometryIndexEXT(_277, bool(0u));
            uint _301 = rayQueryGetIntersectionPrimitiveIndexEXT(_277, bool(0u));
            mat4x3 _302 = rayQueryGetIntersectionObjectToWorldEXT(_277, bool(0u));
            mat4x3 _303 = rayQueryGetIntersectionWorldToObjectEXT(_277, bool(0u));
            _280._m2 = _297;
            _280._m3 = _298;
            _280._m4 = _299;
            _280._m5 = _300;
            _280._m6 = _301;
            _280._m9 = _302;
            _280._m10 = _303;
            if (_292 == 1u)
            {
                float _314 = rayQueryGetIntersectionTEXT(_277, bool(0u));
                _280._m1 = _314;
                vec2 _316 = rayQueryGetIntersectionBarycentricsEXT(_277, bool(0u));
                bool _317 = rayQueryGetIntersectionFrontFaceEXT(_277, bool(0u));
                _280._m7 = _316;
                _280._m8 = _317;
            }
        }
    }
    return _280;
}

void _328(rayQueryEXT _329, uint _330, float _331, float _332)
{
    if ((!((_330 & 4u) != 0u)) && ((_330 & 2u) != 0u))
    {
        uint _345 = rayQueryGetIntersectionTypeEXT(_329, bool(0u));
        float _347 = rayQueryGetRayTMinEXT(_329);
        uint _348 = rayQueryGetIntersectionTypeEXT(_329, bool(1u));
        float _335;
        if (_348 == 0u)
        {
            _335 = _332;
        }
        else
        {
            float _354 = rayQueryGetIntersectionTEXT(_329, bool(1u));
            _335 = _354;
        }
        if (((_331 >= _347) && (_331 <= _335)) && (_345 == 1u))
        {
            rayQueryGenerateIntersectionEXT(_329, _331);
        }
    }
}

void _369(rayQueryEXT _370, uint _371)
{
    if ((!((_371 & 4u) != 0u)) && ((_371 & 2u) != 0u))
    {
        uint _382 = rayQueryGetIntersectionTypeEXT(_370, bool(0u));
        if (_382 == 0u)
        {
            rayQueryConfirmIntersectionEXT(_370);
        }
    }
}

void _388(rayQueryEXT _389, uint _390)
{
    if ((!((_390 & 4u) != 0u)) && ((_390 & 2u) != 0u))
    {
        rayQueryTerminateEXT(_389);
    }
}

void main()
{
    uint _272 = 0u;
    float _273 = 0.0;
    rayQueryEXT _271;
    _50(_271, _16, _12(4u, 255u, 0.100000001490116119384765625, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), _272, _273);
    _10 _321 = _276(_271, _272);
    if (_321._m0 == 3u)
    {
        _328(_271, _272, 10.0, _273);
        return;
    }
    else
    {
        if (_321._m0 == 1u)
        {
            _369(_271, _272);
            return;
        }
        else
        {
            _388(_271, _272);
            return;
        }
    }
}

