////////////////////////////////
// Entry point: "main" (comp) //
////////////////////////////////
#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    uvec3 _m0;
    uint _m1;
};

layout(set = 0, binding = 0, std430) buffer _25_24
{
    uint _m0[];
} _24;

shared uint _13;
shared uvec2 _15;
shared uint _17[513];
shared uint _19;
shared uvec3 _20[3];
shared _5 _22;

uint _28()
{
    uint _32 = 0u;
    uint _35 = 0u;
    uvec2 _68 = uvec2(4294967295u);
    _32 = (((((_13 + _15.x) + _15.y) + _22._m0.x) + _22._m0.y) + _22._m0.z) + _22._m1;
    for (;;)
    {
        if (all(equal(uvec2(0u), _68)))
        {
            break;
        }
        _68 -= uvec2(uint(_68.y == 0u), 1u);
        if (!(_35 < 3u))
        {
            break;
        }
        _32 += ((_20[_35].x + _20[_35].y) + _20[_35].z);
        _35++;
        continue;
    }
    return _32;
}

void main()
{
    uint _138 = 0u;
    uint _134 = 0u;
    uint _137 = 0u;
    uint _132 = 0u;
    uvec2 _164 = uvec2(4294967295u);
    uvec2 _195 = uvec2(4294967295u);
    uvec2 _231 = uvec2(4294967295u);
    if (gl_LocalInvocationIndex == 0u)
    {
        _13 = 0u;
        _15 = uvec2(0u);
        _17 = uint[](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        _19 = 0u;
        _20 = uvec3[](uvec3(0u), uvec3(0u), uvec3(0u));
        _22 = _5(uvec3(0u), 0u);
    }
    barrier();
    uint _151 = gl_WorkGroupID.x * 4u;
    if (gl_LocalInvocationIndex == 0u)
    {
        uint _155 = _28();
        uint _156 = atomicAdd(_19, 0u);
        _132 = _155 + _156;
        for (;;)
        {
            if (all(equal(uvec2(0u), _164)))
            {
                break;
            }
            _164 -= uvec2(uint(_164.y == 0u), 1u);
            if (!(_134 < 513u))
            {
                break;
            }
            _132 += _17[_134];
            _134++;
            continue;
        }
        _24._m0[_151] = _132;
    }
    barrier();
    uint _135 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _195)))
        {
            break;
        }
        _195 -= uvec2(uint(_195.y == 0u), 1u);
        if (!(_135 < 513u))
        {
            break;
        }
        _17[_135] = _135 + gl_WorkGroupID.x;
        _135 += 64u;
        continue;
    }
    uint _219 = atomicAdd(_19, 1u);
    if (gl_LocalInvocationIndex == 0u)
    {
        _13 = 1u;
        _15 = uvec2(2u, 3u);
        _20 = uvec3[](uvec3(4u, 5u, 6u), uvec3(7u, 8u, 9u), uvec3(10u, 11u, 12u));
        _22 = _5(uvec3(13u, 14u, 15u), 16u);
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u)
    {
        for (;;)
        {
            if (all(equal(uvec2(0u), _231)))
            {
                break;
            }
            _231 -= uvec2(uint(_231.y == 0u), 1u);
            if (!(_138 < 513u))
            {
                break;
            }
            _137 += _17[_138];
            _138++;
            continue;
        }
        _24._m0[_151 + 1u] = _137;
        uint _259 = atomicAdd(_19, 0u);
        _24._m0[_151 + 2u] = _259;
        _24._m0[_151 + 3u] = _28();
        return;
    }
    else
    {
        return;
    }
}


///////////////////////////////////////
// Entry point: "padded_only" (comp) //
///////////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    uvec3 _m0;
    uint _m1;
};

layout(set = 0, binding = 0, std430) buffer _25_24
{
    uint _m0[];
} _24;

shared uint _13;
shared uvec2 _15;
shared uint _17[513];
shared uint _19;
shared uvec3 _20[3];
shared _5 _22;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _22 = _5(uvec3(0u), 0u);
    }
    barrier();
    uint _277 = gl_WorkGroupID.x * 4u;
    _24._m0[_277] = ((_22._m0.x + _22._m0.y) + _22._m0.z) + _22._m1;
    _22 = _5(uvec3(13u, 14u, 15u), 16u);
    _24._m0[_277 + 1u] = ((_22._m0.x + _22._m0.y) + _22._m0.z) + _22._m1;
    _24._m0[_277 + 2u] = 0u;
    _24._m0[_277 + 3u] = 0u;
}


////////////////////////////////////////
// Entry point: "no_workgroup" (comp) //
////////////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    uvec3 _m0;
    uint _m1;
};

layout(set = 0, binding = 0, std430) buffer _25_24
{
    uint _m0[];
} _24;

shared uint _13;
shared uvec2 _15;
shared uint _17[513];
shared uint _19;
shared uvec3 _20[3];
shared _5 _22;

void main()
{
    uint _314 = gl_WorkGroupID.x * 4u;
    _24._m0[_314] = gl_WorkGroupID.x;
    _24._m0[_314 + 1u] = 0u;
    _24._m0[_314 + 2u] = 0u;
    _24._m0[_314 + 3u] = 0u;
}

