#version 460
#extension GL_EXT_ray_query : require
#extension GL_EXT_nonuniform_qualifier : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct _4
{
    uint _m0;
};

struct _14
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

struct _17
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

layout(set = 0, binding = 2, std140) uniform _23_22
{
    _4 _m0;
} _22;

layout(set = 0, binding = 3, std430) buffer _26_25
{
    vec2 _m0[];
} _25;

layout(set = 0, binding = 0) uniform accelerationStructureEXT _18[4];
layout(set = 0, binding = 1) uniform accelerationStructureEXT _21[4];

void _62(rayQueryEXT _63, accelerationStructureEXT _64, _14 _65, inout uint _66, out float _67)
{
    bool _92 = (_65._m0 & 256u) != 0u;
    bool _100 = (_65._m0 & 16u) != 0u;
    bool _103 = (_65._m0 & 32u) != 0u;
    bool _112 = (_65._m0 & 1u) != 0u;
    bool _115 = (_65._m0 & 2u) != 0u;
    bool _118 = (_65._m0 & 64u) != 0u;
    bool _121 = (_65._m0 & 128u) != 0u;
    _67 = _65._m3;
    if (((((((!((((((_115 && _112) || (_121 && _112)) || (_121 && _115)) || (_121 && _118)) || (_118 && _112)) || (_118 && _115))) && (_65._m2 <= _65._m3)) && (_65._m2 >= 0.0)) && (!(any(isnan(_65._m4)) || any(isinf(_65._m4))))) && (!(any(isnan(_65._m5)) || any(isinf(_65._m5))))) && (!(((_65._m0 & 512u) != 0u) && _92))) && (!(((_100 && _92) || (_103 && _92)) || (_103 && _100))))
    {
        rayQueryInitializeEXT(_63, _64, _65._m0, _65._m1, _65._m4, _65._m2, _65._m5, _65._m3);
        _66 = 1u;
    }
    else
    {
    }
}

bool _168(rayQueryEXT _169, inout uint _170)
{
    bool _172 = false;
    if ((!((_170 & 4u) != 0u)) && ((_170 & 1u) != 0u))
    {
        bool _183 = rayQueryProceedEXT(_169);
        _172 = _183;
        _170 |= (_183 ? 2u : 6u);
    }
    return _172;
}

_17 _196(rayQueryEXT _197, uint _198)
{
    _17 _201 = _17(0u, 0.0, 0u, 0u, 0u, 0u, 0u, vec2(0.0), false, mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)), mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)));
    if (((_198 & 4u) != 0u) && ((_198 & 2u) != 0u))
    {
        uint _210 = rayQueryGetIntersectionTypeEXT(_197, bool(1u));
        _201._m0 = _210;
        if (_210 != 0u)
        {
            uint _215 = rayQueryGetIntersectionInstanceCustomIndexEXT(_197, bool(1u));
            uint _216 = rayQueryGetIntersectionInstanceIdEXT(_197, bool(1u));
            uint _217 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(_197, bool(1u));
            uint _218 = rayQueryGetIntersectionGeometryIndexEXT(_197, bool(1u));
            uint _219 = rayQueryGetIntersectionPrimitiveIndexEXT(_197, bool(1u));
            mat4x3 _220 = rayQueryGetIntersectionObjectToWorldEXT(_197, bool(1u));
            mat4x3 _221 = rayQueryGetIntersectionWorldToObjectEXT(_197, bool(1u));
            _201._m2 = _215;
            _201._m3 = _216;
            _201._m4 = _217;
            _201._m5 = _218;
            _201._m6 = _219;
            _201._m9 = _220;
            _201._m10 = _221;
            float _236 = rayQueryGetIntersectionTEXT(_197, bool(1u));
            _201._m1 = _236;
            if (_210 == 1u)
            {
                vec2 _238 = rayQueryGetIntersectionBarycentricsEXT(_197, bool(1u));
                bool _239 = rayQueryGetIntersectionFrontFaceEXT(_197, bool(1u));
                _201._m7 = _238;
                _201._m8 = _239;
            }
        }
    }
    return _201;
}

void main()
{
    uint _51 = 0u;
    float _53 = 0.0;
    uvec2 _154 = uvec2(4294967295u);
    uvec2 _256 = uvec2(4294967295u);
    rayQueryEXT _48;
    _62(_48, _18[_22._m0._m0], _14(4u, 255u, 0.100000001490116119384765625, 100.0, vec3(0.0, 0.0, -2.0), vec3(0.0, 0.0, 1.0)), _51, _53);
    for (;;)
    {
        if (all(equal(uvec2(0u), _154)))
        {
            break;
        }
        _154 -= uvec2(uint(_154.y == 0u), 1u);
        bool _165 = _168(_48, _51);
        if (!_165)
        {
            break;
        }
        continue;
    }
    _17 _245 = _196(_48, _51);
    _62(_48, _21[nonuniformEXT(gl_LocalInvocationIndex)], _14(4u, 255u, 0.100000001490116119384765625, 100.0, vec3(_245._m7, 0.0), vec3(0.0, 0.0, 1.0)), _51, _53);
    for (;;)
    {
        if (all(equal(uvec2(0u), _256)))
        {
            break;
        }
        _256 -= uvec2(uint(_256.y == 0u), 1u);
        bool _267 = _168(_48, _51);
        if (!_267)
        {
            break;
        }
        continue;
    }
    _25._m0[gl_LocalInvocationIndex] = vec2(_245._m1, _196(_48, _51)._m1);
}

