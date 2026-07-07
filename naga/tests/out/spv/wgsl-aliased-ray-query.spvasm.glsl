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

struct _12
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

layout(set = 0, binding = 0) uniform accelerationStructureEXT _13;

void _40(rayQueryEXT _41, accelerationStructureEXT _42, _8 _43, inout uint _44, out float _45)
{
    bool _70 = (_43._m0 & 256u) != 0u;
    bool _78 = (_43._m0 & 16u) != 0u;
    bool _81 = (_43._m0 & 32u) != 0u;
    bool _89 = (_43._m0 & 1u) != 0u;
    bool _92 = (_43._m0 & 2u) != 0u;
    bool _95 = (_43._m0 & 64u) != 0u;
    bool _98 = (_43._m0 & 128u) != 0u;
    _45 = _43._m3;
    if (((((((!((((((_92 && _89) || (_98 && _89)) || (_98 && _92)) || (_98 && _95)) || (_95 && _89)) || (_95 && _92))) && (_43._m2 <= _43._m3)) && (_43._m2 >= 0.0)) && (!(any(isnan(_43._m4)) || any(isinf(_43._m4))))) && (!(any(isnan(_43._m5)) || any(isinf(_43._m5))))) && (!(((_43._m0 & 512u) != 0u) && _70))) && (!(((_78 && _70) || (_81 && _70)) || (_81 && _78))))
    {
        rayQueryInitializeEXT(_41, _42, _43._m0, _43._m1, _43._m4, _43._m2, _43._m5, _43._m3);
        _44 = 1u;
    }
    else
    {
    }
}

_12 _126(rayQueryEXT _127, uint _128)
{
    _12 _131 = _12(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if ((!((_128 & 4u) != 0u)) && ((_128 & 2u) != 0u))
    {
        uint _141 = rayQueryGetIntersectionTypeEXT(_127, bool(0u));
        uint _143 = (_141 == 0u) ? 1u : 3u;
        _131._m0 = _143;
        if (_143 != 0u)
        {
            uint _148 = rayQueryGetIntersectionInstanceCustomIndexEXT(_127, bool(0u));
            uint _149 = rayQueryGetIntersectionInstanceIdEXT(_127, bool(0u));
            uint _150 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_127, bool(0u));
            uint _151 = rayQueryGetIntersectionGeometryIndexEXT(_127, bool(0u));
            uint _152 = rayQueryGetIntersectionPrimitiveIndexEXT(_127, bool(0u));
            mat4x3 _153 = rayQueryGetIntersectionObjectToWorldEXT(_127, bool(0u));
            mat4x3 _154 = rayQueryGetIntersectionWorldToObjectEXT(_127, bool(0u));
            _131._m2 = _148;
            _131._m3 = _149;
            _131._m4 = _150;
            _131._m5 = _151;
            _131._m6 = _152;
            _131._m9 = _153;
            _131._m10 = _154;
            if (_143 == 1u)
            {
                float _169 = rayQueryGetIntersectionTEXT(_127, bool(0u));
                _131._m1 = _169;
                vec2 _171 = rayQueryGetIntersectionBarycentricsEXT(_127, bool(0u));
                bool _172 = rayQueryGetIntersectionFrontFaceEXT(_127, bool(0u));
                _131._m7 = _171;
                _131._m8 = _172;
            }
        }
    }
    return _131;
}

void _185(rayQueryEXT _186, uint _187, float _188, float _189)
{
    if ((!((_187 & 4u) != 0u)) && ((_187 & 2u) != 0u))
    {
        uint _202 = rayQueryGetIntersectionTypeEXT(_186, bool(0u));
        float _204 = rayQueryGetRayTMinEXT(_186);
        uint _205 = rayQueryGetIntersectionTypeEXT(_186, bool(1u));
        float _192;
        if (_205 == 0u)
        {
            _192 = _189;
        }
        else
        {
            float _211 = rayQueryGetIntersectionTEXT(_186, bool(0u));
            _192 = _211;
        }
        if (((_188 >= _204) && (_188 <= _192)) && (_202 == 1u))
        {
            rayQueryGenerateIntersectionEXT(_186, _188);
        }
    }
}

void _226(rayQueryEXT _227, uint _228)
{
    if ((!((_228 & 4u) != 0u)) && ((_228 & 2u) != 0u))
    {
        uint _239 = rayQueryGetIntersectionTypeEXT(_227, bool(0u));
        if (_239 == 0u)
        {
            rayQueryConfirmIntersectionEXT(_227);
        }
    }
}

void _245(rayQueryEXT _246, uint _247)
{
    if ((!((_247 & 4u) != 0u)) && ((_247 & 2u) != 0u))
    {
        rayQueryTerminateEXT(_246);
    }
}

void main()
{
    uint _34 = 0u;
    float _37 = 0.0;
    rayQueryEXT _31;
    _40(_31, _13, _8(4u, 255u, 0.100000001490116119384765625, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), _34, _37);
    _12 _178 = _126(_31, _34);
    if (_178._m0 == 3u)
    {
        _185(_31, _34, 10.0, _37);
        return;
    }
    else
    {
        if (_178._m0 == 1u)
        {
            _226(_31, _34);
            return;
        }
        else
        {
            _245(_31, _34);
            return;
        }
    }
}

