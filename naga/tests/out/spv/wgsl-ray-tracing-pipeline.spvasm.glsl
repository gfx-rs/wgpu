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

layout(location = 0) rayPayloadEXT _4 _12;
layout(set = 0, binding = 0) uniform accelerationStructureEXT _14;

void _50(accelerationStructureEXT _51, _9 _52)
{
    bool _78 = (_52._m0 & 256u) != 0u;
    bool _86 = (_52._m0 & 16u) != 0u;
    bool _89 = (_52._m0 & 32u) != 0u;
    bool _98 = (_52._m0 & 1u) != 0u;
    bool _101 = (_52._m0 & 2u) != 0u;
    bool _104 = (_52._m0 & 64u) != 0u;
    bool _107 = (_52._m0 & 128u) != 0u;
    if (((((((!((((((_101 && _98) || (_107 && _98)) || (_107 && _101)) || (_107 && _104)) || (_104 && _98)) || (_104 && _101))) && (_52._m2 <= _52._m3)) && (_52._m2 >= 0.0)) && (!(any(isnan(_52._m4)) || any(isinf(_52._m4))))) && (!(any(isnan(_52._m5)) || any(isinf(_52._m5))))) && (!(((_52._m0 & 512u) != 0u) && _78))) && (!(((_86 && _78) || (_89 && _78)) || (_89 && _86))))
    {
        traceRayEXT(_51, _52._m0, _52._m1, 0u, 0u, 0u, _52._m4, _52._m2, _52._m5, _52._m3, 0);
    }
    else
    {
    }
}

void main()
{
    _12 = _4(0u, 0u);
    vec3 _41 = vec3(gl_LaunchIDEXT) / vec3(gl_LaunchSizeEXT);
    _50(_14, _9(0u, 255u, 0.00999999977648258209228515625, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0) + ((vec3(_41.x, 0.0, _41.y) * 2.0) - vec3(1.0))));
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

layout(location = 0) rayPayloadInEXT _4 _16;
hitAttributeEXT vec2 _151;

void main()
{
    _16._m0++;
    _16._m1 = uint(gl_InstanceCustomIndexEXT) + uint(clamp(_151.x + _151.y, 0.0, 4294967040.0));
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

layout(location = 0) rayPayloadInEXT _4 _16;
hitAttributeEXT vec2 _179;

void main()
{
    _16._m1 = uint(clamp((1.0 - _179.x) - _179.y, 0.0, 4294967040.0));
}


/////////////////////////////////////////////////
// Entry point: "closest_hit_triangle" (rchit) //
/////////////////////////////////////////////////
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

layout(location = 0) rayPayloadInEXT _4 _16;

void main()
{
    _16._m0 = uint(gl_PrimitiveID);
}

