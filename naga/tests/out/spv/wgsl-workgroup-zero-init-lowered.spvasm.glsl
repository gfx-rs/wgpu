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
    uint _78 = 0u;
    uvec2 _96 = uvec2(4294967295u);
    uvec2 _117 = uvec2(4294967295u);
    uvec2 _150 = uvec2(4294967295u);
    if (gl_LocalInvocationIndex < 16u)
    {
        _30[gl_LocalInvocationIndex] = 0.0;
    }
    _78 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _96)))
        {
            break;
        }
        _96 -= uvec2(uint(_96.y == 0u), 1u);
        _32[_78] = _5(vec4(0.0), vec4(0.0));
        uint _110 = _78;
        uint _111 = _110 + 32u;
        _78 = _111;
        if (_111 >= 256u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    _78 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _117)))
        {
            break;
        }
        _117 -= uvec2(uint(_117.y == 0u), 1u);
        _36._m2[_53(_78, 18u)][_43(_78, 18u)] = vec4(0.0);
        uint _136 = _78;
        uint _137 = _136 + 32u;
        _78 = _137;
        if (_137 >= 324u)
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
    _78 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _150)))
        {
            break;
        }
        _150 -= uvec2(uint(_150.y == 0u), 1u);
        atomicExchange(_36._m3[_53(_78, 100u)]._m1[_43(_78, 100u)], 0u);
        uint _166 = _78;
        uint _167 = _166 + 32u;
        _78 = _167;
        if (_167 >= 300u)
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
    _40._m0[0u] = ((float(_28) + _30[3u]) + _32[7u]._m1.y) + float(_34[0u].x);
    _40._m0[1u] = (_36._m0[1u].x + float(_36._m1)) + _36._m2[2u][5u].z;
    uint _206 = atomicAdd(_36._m3[1u]._m1[42u], 0u);
    uint _208 = atomicAdd(_36._m3[2u]._m0, 0u);
    _40._m0[2u] = float(_206 + _208);
    int _212 = atomicAdd(_38, 0);
    _40._m0[3u] = float(_212);
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
    uvec2 _246 = uvec2(4294967295u);
    _27 _230 = _27(gl_WorkGroupID, gl_LocalInvocationIndex);
    uint _241 = _230._m1;
    uint _238 = _241;
    for (;;)
    {
        if (all(equal(uvec2(0u), _246)))
        {
            break;
        }
        _246 -= uvec2(uint(_246.y == 0u), 1u);
        _32[_238] = _5(vec4(0.0), vec4(0.0));
        uint _259 = _238;
        uint _260 = _259 + 16u;
        _238 = _260;
        if (_260 >= 256u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    if (_241 == 0u)
    {
        atomicExchange(_38, 0);
    }
    barrier();
    float _268 = _32[_230._m1]._m0.x;
    int _269 = atomicAdd(_38, 0);
    _40._m0[_230._m1] = _268 + float(_269);
}

