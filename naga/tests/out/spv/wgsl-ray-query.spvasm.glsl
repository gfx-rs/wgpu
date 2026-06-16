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
    if ((_150 & 1u) != 0u)
    {
        bool _159 = rayQueryProceedEXT(_149);
        _152 = _159;
        _150 |= (_159 ? 2u : 6u);
    }
    return _152;
}

_10 _172(rayQueryEXT _173, uint _174)
{
    _10 _177 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if (((_174 & 4u) != 0u) && ((_174 & 2u) != 0u))
    {
        uint _186 = rayQueryGetIntersectionTypeEXT(_173, bool(1u));
        _177._m0 = _186;
        bool _188 = _186 != 0u;
        if (_188)
        {
            uint _191 = rayQueryGetIntersectionInstanceCustomIndexEXT(_173, bool(1u));
            uint _192 = rayQueryGetIntersectionInstanceIdEXT(_173, bool(1u));
            uint _193 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_173, bool(1u));
            uint _194 = rayQueryGetIntersectionGeometryIndexEXT(_173, bool(1u));
            uint _195 = rayQueryGetIntersectionPrimitiveIndexEXT(_173, bool(1u));
            mat4x3 _196 = rayQueryGetIntersectionObjectToWorldEXT(_173, bool(1u));
            mat4x3 _197 = rayQueryGetIntersectionWorldToObjectEXT(_173, bool(1u));
            _177._m2 = _191;
            _177._m3 = _192;
            _177._m4 = _193;
            _177._m5 = _194;
            _177._m6 = _195;
            _177._m9 = _196;
            _177._m10 = _197;
            float _212 = rayQueryGetIntersectionTEXT(_173, bool(1u));
            _177._m1 = _212;
            if (_188)
            {
                vec2 _214 = rayQueryGetIntersectionBarycentricsEXT(_173, bool(1u));
                bool _215 = rayQueryGetIntersectionFrontFaceEXT(_173, bool(1u));
                _177._m7 = _214;
                _177._m8 = _215;
            }
        }
    }
    return _177;
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
    return _172(_31, _34);
}

vec3 _225(vec3 _223, _10 _224)
{
    return normalize(_223 - (_224._m9 * vec4(normalize((_224._m10 * vec4(_223, 1.0)).xy) * 2.400000095367431640625, 0.0, 1.0)));
}

void main()
{
    _10 _250 = _25(vec3(0.0), vec3(0.0, 1.0, 0.0), _15);
    _17._m0._m0 = uint(_250._m0 == 0u);
    _17._m0._m1 = _225(vec3(0.0, 1.0, 0.0) * _250._m1, _250);
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

_10 _271(rayQueryEXT _272, uint _273)
{
    _10 _275 = _10(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if ((!((_273 & 4u) != 0u)) && ((_273 & 2u) != 0u))
    {
        uint _285 = rayQueryGetIntersectionTypeEXT(_272, bool(0u));
        uint _287 = (_285 == 0u) ? 1u : 3u;
        _275._m0 = _287;
        bool _289 = _287 != 0u;
        if (_289)
        {
            uint _292 = rayQueryGetIntersectionInstanceCustomIndexEXT(_272, bool(0u));
            uint _293 = rayQueryGetIntersectionInstanceIdEXT(_272, bool(0u));
            uint _294 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_272, bool(0u));
            uint _295 = rayQueryGetIntersectionGeometryIndexEXT(_272, bool(0u));
            uint _296 = rayQueryGetIntersectionPrimitiveIndexEXT(_272, bool(0u));
            mat4x3 _297 = rayQueryGetIntersectionObjectToWorldEXT(_272, bool(0u));
            mat4x3 _298 = rayQueryGetIntersectionWorldToObjectEXT(_272, bool(0u));
            _275._m2 = _292;
            _275._m3 = _293;
            _275._m4 = _294;
            _275._m5 = _295;
            _275._m6 = _296;
            _275._m9 = _297;
            _275._m10 = _298;
            if (_289)
            {
                float _309 = rayQueryGetIntersectionTEXT(_272, bool(0u));
                _275._m1 = _309;
                vec2 _311 = rayQueryGetIntersectionBarycentricsEXT(_272, bool(0u));
                bool _312 = rayQueryGetIntersectionFrontFaceEXT(_272, bool(0u));
                _275._m7 = _311;
                _275._m8 = _312;
            }
        }
    }
    return _275;
}

void _323(rayQueryEXT _324, uint _325, float _326, float _327)
{
    if ((!((_325 & 4u) != 0u)) && ((_325 & 2u) != 0u))
    {
        uint _340 = rayQueryGetIntersectionTypeEXT(_324, bool(0u));
        float _342 = rayQueryGetRayTMinEXT(_324);
        uint _343 = rayQueryGetIntersectionTypeEXT(_324, bool(1u));
        float _330;
        if (_343 == 0u)
        {
            _330 = _327;
        }
        else
        {
            float _349 = rayQueryGetIntersectionTEXT(_324, bool(0u));
            _330 = _349;
        }
        if (((_326 >= _342) && (_326 <= _330)) && (_340 == 1u))
        {
            rayQueryGenerateIntersectionEXT(_324, _326);
        }
    }
}

void _364(rayQueryEXT _365, uint _366)
{
    if ((!((_366 & 4u) != 0u)) && ((_366 & 2u) != 0u))
    {
        uint _377 = rayQueryGetIntersectionTypeEXT(_365, bool(0u));
        if (_377 == 0u)
        {
            rayQueryConfirmIntersectionEXT(_365);
        }
    }
}

void _383(rayQueryEXT _384, uint _385)
{
    if ((!((_385 & 4u) != 0u)) && ((_385 & 2u) != 0u))
    {
        rayQueryTerminateEXT(_384);
    }
}

void main()
{
    uint _267 = 0u;
    float _268 = 0.0;
    rayQueryEXT _266;
    _42(_266, _15, _12(4u, 255u, 0.100000001490116119384765625, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), _267, _268);
    _10 _316 = _271(_266, _267);
    if (_316._m0 == 3u)
    {
        _323(_266, _267, 10.0, _268);
        return;
    }
    else
    {
        if (_316._m0 == 1u)
        {
            _364(_266, _267);
            return;
        }
        else
        {
            _383(_266, _267);
            return;
        }
    }
}

