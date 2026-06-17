////////////////////////////////////////
// Entry point: "ray_gen_main" (rgen) //
////////////////////////////////////////
#version 460
#extension GL_EXT_ray_tracing : require

struct _4
{
    uint _m0;
    uint _m1;
};

struct _9
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

layout(location = 0) rayPayloadEXT _4 _11;
layout(set = 0, binding = 0) uniform accelerationStructureEXT _13;
layout(location = 1) rayPayloadInEXT _4 _15;

void _49(accelerationStructureEXT _50, _9 _51)
{
    bool _77 = (_51._m0 & 256u) != 0u;
    bool _85 = (_51._m0 & 16u) != 0u;
    bool _88 = (_51._m0 & 32u) != 0u;
    bool _97 = (_51._m0 & 1u) != 0u;
    bool _100 = (_51._m0 & 2u) != 0u;
    bool _103 = (_51._m0 & 64u) != 0u;
    bool _106 = (_51._m0 & 128u) != 0u;
    if (((((((!((((((_100 && _97) || (_106 && _97)) || (_106 && _100)) || (_106 && _103)) || (_103 && _97)) || (_103 && _100))) && (_51._m2 <= _51._m3)) && (_51._m2 >= 0.0)) && (!(any(isnan(_51._m4)) || any(isinf(_51._m4))))) && (!(any(isnan(_51._m5)) || any(isinf(_51._m5))))) && (!(((_51._m0 & 512u) != 0u) && _77))) && (!(((_85 && _77) || (_88 && _77)) || (_88 && _85))))
    {
        traceRayEXT(_50, _51._m0, _51._m1, 0u, 0u, 0u, _51._m4, _51._m2, _51._m5, _51._m3, 0);
    }
    else
    {
    }
}

void main()
{
    _11 = _4(0u, 0u);
    vec3 _40 = vec3(gl_LaunchIDEXT) / vec3(gl_LaunchSizeEXT);
    _49(_13, _9(0u, 255u, 0.00999999977648258209228515625, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0) + ((vec3(_40.x, 0.0, _40.y) * 2.0) - vec3(1.0))));
}


/////////////////////////////////
// Entry point: "miss" (rmiss) //
/////////////////////////////////
#version 460
#extension GL_EXT_ray_tracing : require

struct _4
{
    uint _m0;
    uint _m1;
};

struct _9
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

layout(location = 0) rayPayloadEXT _4 _11;
layout(set = 0, binding = 0) uniform accelerationStructureEXT _13;
layout(location = 1) rayPayloadInEXT _4 _15;

void main()
{
}


/////////////////////////////////////////
// Entry point: "any_hit_main" (rahit) //
/////////////////////////////////////////
#version 460
#extension GL_EXT_ray_tracing : require

struct _4
{
    uint _m0;
    uint _m1;
};

struct _9
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

layout(location = 0) rayPayloadEXT _4 _11;
layout(set = 0, binding = 0) uniform accelerationStructureEXT _13;
layout(location = 1) rayPayloadInEXT _4 _15;

void main()
{
    _15._m0++;
    _15._m1 = uint(gl_InstanceCustomIndexEXT);
}


/////////////////////////////////////////////
// Entry point: "closest_hit_main" (rchit) //
/////////////////////////////////////////////
#version 460
#extension GL_EXT_ray_tracing : require

struct _4
{
    uint _m0;
    uint _m1;
};

struct _9
{
    uint _m0;
    uint _m1;
    float _m2;
    float _m3;
    vec3 _m4;
    vec3 _m5;
};

layout(location = 0) rayPayloadEXT _4 _11;
layout(set = 0, binding = 0) uniform accelerationStructureEXT _13;
layout(location = 1) rayPayloadInEXT _4 _15;

void main()
{
}

