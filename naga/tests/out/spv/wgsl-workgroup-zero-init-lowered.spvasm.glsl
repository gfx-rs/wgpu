////////////////////////////////
// Entry point: "main" (comp) //
////////////////////////////////
#version 460
layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;

struct _5
{
    vec4 _m0;
    vec4 _m1;
};

struct _9
{
    uint _m0;
    uint _m1[100];
};

struct _16
{
    mat4 _m0;
    uint _m1;
    vec4 _m2[18][18];
    _9 _m3[3];
};

struct _27
{
    uvec3 _m0;
    uint _m1;
};

layout(set = 0, binding = 0, std430) buffer _41_40
{
    float _m0[];
} _40;

shared uint _28;
shared float _30[16];
shared _5 _32[256];
shared uvec2 _34[1];
shared _16 _36;
shared int _38;

uint _43(uint _45, uint _46)
{
    return _45 % ((_46 == 0u) ? 1u : _46);
}

uint _53(uint _54, uint _55)
{
    return _54 / ((_55 == 0u) ? 1u : _55);
}

void main()
{
    uint _84 = 0u;
    uvec2 _102 = uvec2(4294967295u);
    uvec2 _123 = uvec2(4294967295u);
    uvec2 _155 = uvec2(4294967295u);
    if (gl_LocalInvocationIndex < 16u)
    {
        _30[gl_LocalInvocationIndex] = 0.0;
    }
    _84 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _102)))
        {
            break;
        }
        _102 -= uvec2(uint(_102.y == 0u), 1u);
        _32[_84] = _5(vec4(0.0), vec4(0.0));
        uint _116 = _84;
        uint _117 = _116 + 32u;
        _84 = _117;
        if (_117 >= 256u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    _84 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _123)))
        {
            break;
        }
        _123 -= uvec2(uint(_123.y == 0u), 1u);
        _36._m2[_53(_84, 18u)][_43(_84, 18u)] = vec4(0.0);
        uint _141 = _84;
        uint _142 = _141 + 32u;
        _84 = _142;
        if (_142 >= 324u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    if (gl_LocalInvocationIndex < 3u)
    {
        atomicExchange(_36._m3[gl_LocalInvocationIndex]._m0, 0u);
    }
    _84 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _155)))
        {
            break;
        }
        _155 -= uvec2(uint(_155.y == 0u), 1u);
        atomicExchange(_36._m3[_53(_84, 100u)]._m1[_43(_84, 100u)], 0u);
        uint _171 = _84;
        uint _172 = _171 + 32u;
        _84 = _172;
        if (_172 >= 300u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _28 = 0u;
        _34[0u] = uvec2(0u);
        _36._m0 = mat4(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
        _36._m1 = 0u;
        atomicExchange(_38, 0);
    }
    barrier();
    uint _187 = (gl_LocalInvocationID.x + (gl_LocalInvocationID.y * 8u)) * 4u;
    _40._m0[_187] = ((float(_28) + _30[3u]) + _32[7u]._m1.y) + float(_34[0u].x);
    _40._m0[_187 + 1u] = (_36._m0[1u].x + float(_36._m1)) + _36._m2[2u][5u].z;
    uint _218 = atomicAdd(_36._m3[1u]._m1[42u], 0u);
    uint _220 = atomicAdd(_36._m3[2u]._m0, 0u);
    _40._m0[_187 + 2u] = float(_218 + _220);
    int _225 = atomicAdd(_38, 0);
    _40._m0[_187 + 3u] = float(_225);
}


//////////////////////////////////////
// Entry point: "with_index" (comp) //
//////////////////////////////////////
#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    vec4 _m0;
    vec4 _m1;
};

struct _9
{
    uint _m0;
    uint _m1[100];
};

struct _16
{
    mat4 _m0;
    uint _m1;
    vec4 _m2[18][18];
    _9 _m3[3];
};

struct _27
{
    uvec3 _m0;
    uint _m1;
};

layout(set = 0, binding = 0, std430) buffer _41_40
{
    float _m0[];
} _40;

shared uint _28;
shared float _30[16];
shared _5 _32[256];
shared uvec2 _34[1];
shared _16 _36;
shared int _38;

uint _43(uint _45, uint _46)
{
    return _45 % ((_46 == 0u) ? 1u : _46);
}

void main()
{
    if (gl_LocalInvocationIndex < 16u)
    {
        _30[gl_LocalInvocationIndex] = 0.0;
    }
    barrier();
    _40._m0[gl_LocalInvocationIndex] = _30[_43(gl_LocalInvocationIndex, 16u)];
}


///////////////////////////////////////
// Entry point: "with_struct" (comp) //
///////////////////////////////////////
#version 460
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    vec4 _m0;
    vec4 _m1;
};

struct _9
{
    uint _m0;
    uint _m1[100];
};

struct _16
{
    mat4 _m0;
    uint _m1;
    vec4 _m2[18][18];
    _9 _m3[3];
};

struct _27
{
    uvec3 _m0;
    uint _m1;
};

layout(set = 0, binding = 0, std430) buffer _41_40
{
    float _m0[];
} _40;

shared uint _28;
shared float _30[16];
shared _5 _32[256];
shared uvec2 _34[1];
shared _16 _36;
shared int _38;

void main()
{
    uvec2 _258 = uvec2(4294967295u);
    _27 _243 = _27(gl_WorkGroupID, gl_LocalInvocationIndex);
    uint _253 = _243._m1;
    uint _250 = _253;
    for (;;)
    {
        if (all(equal(uvec2(0u), _258)))
        {
            break;
        }
        _258 -= uvec2(uint(_258.y == 0u), 1u);
        _32[_250] = _5(vec4(0.0), vec4(0.0));
        uint _271 = _250;
        uint _272 = _271 + 16u;
        _250 = _272;
        if (_272 >= 256u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    if (_253 == 0u)
    {
        atomicExchange(_38, 0);
    }
    barrier();
    float _280 = _32[_243._m1]._m0.x;
    int _281 = atomicAdd(_38, 0);
    _40._m0[_243._m1] = _280 + float(_281);
}

