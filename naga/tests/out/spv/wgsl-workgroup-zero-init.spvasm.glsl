////////////////////////////////
// Entry point: "main" (comp) //
////////////////////////////////
#version 460
layout(local_size_x = 4, local_size_y = 2, local_size_z = 1) in;

struct _8
{
    vec3 _m0;
    uint _m1;
    float _m2[4];
};

shared uint _18;
shared uint _20[3];
shared vec4 _22[8];
shared float _24[5][4];
shared _8 _26[2];

uint _39(uint _40, uint _41)
{
    return _40 / ((_41 == 0u) ? 1u : _41);
}

uint _28(uint _30, uint _31)
{
    return _30 % ((_31 == 0u) ? 1u : _31);
}

void main()
{
    uint _56 = 0u;
    uvec2 _76 = uvec2(4294967295u);
    if (gl_LocalInvocationIndex < 3u)
    {
        _20[gl_LocalInvocationIndex] = 0u;
    }
    _22[gl_LocalInvocationIndex] = vec4(0.0);
    _56 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _76)))
        {
            break;
        }
        _76 -= uvec2(uint(_76.y == 0u), 1u);
        if (_56 >= 20u)
        {
            break;
        }
        _24[_39(_56, 4u)][_28(_56, 4u)] = 0.0;
        _56 += 8u;
        continue;
    }
    if (gl_LocalInvocationIndex < 2u)
    {
        _26[gl_LocalInvocationIndex]._m0 = vec3(0.0);
        _26[gl_LocalInvocationIndex]._m1 = 0u;
    }
    _26[_39(gl_LocalInvocationIndex, 4u)]._m2[_28(gl_LocalInvocationIndex, 4u)] = 0.0;
    if (gl_LocalInvocationIndex == 0u)
    {
        _18 = 0u;
    }
    barrier();
    uint _135 = atomicAdd(_26[0u]._m1, uint(clamp((((float(_20[_28(gl_LocalInvocationIndex, 3u)]) + _22[gl_LocalInvocationIndex].x) + _24[_28(gl_LocalInvocationIndex, 5u)][_28(gl_LocalInvocationIndex, 4u)]) + _26[_28(gl_LocalInvocationIndex, 2u)]._m0.x) + _26[_28(gl_LocalInvocationIndex, 2u)]._m2[_28(gl_LocalInvocationIndex, 4u)], 0.0, 4294967040.0)));
    if (gl_LocalInvocationIndex == 0u)
    {
        uint _144 = atomicAdd(_26[1u]._m1, 0u);
        _18 = _144;
        return;
    }
    else
    {
        return;
    }
}


////////////////////////////////////
// Entry point: "no_index" (comp) //
////////////////////////////////////
#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct _8
{
    vec3 _m0;
    uint _m1;
    float _m2[4];
};

shared uint _18;
shared uint _20[3];
shared vec4 _22[8];
shared float _24[5][4];
shared _8 _26[2];

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _18 = 0u;
    }
    barrier();
}

