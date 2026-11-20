////////////////////////////////
// Entry point: "main" (comp) //
////////////////////////////////
#version 460
layout(local_size_x = 8, local_size_y = 2, local_size_z = 1) in;

struct _16
{
    vec4 _m0[5][3];
    uint _m1[19];
    mat3x2 _m2;
    mat2 _m3[3];
};

struct _17
{
    mat2 _m0;
};

layout(set = 0, binding = 0, std430) buffer _44_43
{
    uint _m0[];
} _43;

shared _16 _29[3];
shared uint _31[37];
shared float _33[37];
shared _17 _35;
shared uint _37;
shared uvec3 _39;
shared mat2x3 _41;

void main()
{
    uint _92 = 0u;
    uint _81 = 0u;
    bool _75 = false;
    bool _100 = false;
    uint _91 = 0u;
    bool _79 = false;
    bool _72 = false;
    uint _99 = 0u;
    bool _77 = false;
    uvec2 _231 = uvec2(4294967295u);
    uvec2 _252 = uvec2(4294967295u);
    uvec2 _273 = uvec2(4294967295u);
    uvec2 _311 = uvec2(4294967295u);
    uvec2 _344 = uvec2(4294967295u);
    uvec2 _394 = uvec2(4294967295u);
    uvec2 _432 = uvec2(4294967295u);
    uvec2 _453 = uvec2(4294967295u);
    uvec2 _474 = uvec2(4294967295u);
    uvec2 _503 = uvec2(4294967295u);
    uvec2 _531 = uvec2(4294967295u);
    for (uint _109 = gl_LocalInvocationIndex; _109 < 45u; _109 += 16u)
    {
        _29[_109 / 15u]._m0[(_109 / 3u) % 5u][_109 % 3u] = vec4(0.0);
    }
    for (uint _125 = gl_LocalInvocationIndex; _125 < 57u; _125 += 16u)
    {
        _29[_125 / 19u]._m1[_125 % 19u] = 0u;
    }
    for (uint _138 = gl_LocalInvocationIndex; _138 < 3u; _138 += 16u)
    {
        _29[_138]._m2 = mat3x2(vec2(0.0), vec2(0.0), vec2(0.0));
    }
    for (uint _148 = gl_LocalInvocationIndex; _148 < 9u; _148 += 16u)
    {
        _29[_148 / 3u]._m3[_148 % 3u] = mat2(vec2(0.0), vec2(0.0));
    }
    for (uint _159 = gl_LocalInvocationIndex; _159 < 37u; _159 += 16u)
    {
        _31[_159] = 0u;
    }
    for (uint _169 = gl_LocalInvocationIndex; _169 < 37u; _169 += 16u)
    {
        _33[_169] = 0.0;
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _35 = _17(mat2(vec2(0.0), vec2(0.0)));
        _37 = 0u;
        _39 = uvec3(0u);
        _41 = mat2x3(vec3(0.0), vec3(0.0));
    }
    barrier();
    if (_37 == 0u)
    {
        _72 = all(equal(_39, uvec3(0u)));
    }
    else
    {
        _72 = false;
    }
    _75 = _72;
    if (all(equal(_41[0u], vec3(0.0))))
    {
        _77 = all(equal(_41[1u], vec3(0.0)));
    }
    else
    {
        _77 = false;
    }
    _75 = _75 && _77;
    if (all(equal(_35._m0[0u], vec2(0.0))))
    {
        _79 = all(equal(_35._m0[1u], vec2(0.0)));
    }
    else
    {
        _79 = false;
    }
    _75 = _75 && _79;
    for (;;)
    {
        if (all(equal(uvec2(0u), _231)))
        {
            break;
        }
        _231 -= uvec2(uint(_231.y == 0u), 1u);
        if (!(_81 < 3u))
        {
            break;
        }
        uint _83 = 0u;
        for (;;)
        {
            if (all(equal(uvec2(0u), _252)))
            {
                break;
            }
            _252 -= uvec2(uint(_252.y == 0u), 1u);
            if (!(_83 < 5u))
            {
                break;
            }
            uint _85 = 0u;
            for (;;)
            {
                if (all(equal(uvec2(0u), _273)))
                {
                    break;
                }
                _273 -= uvec2(uint(_273.y == 0u), 1u);
                if (!(_85 < 3u))
                {
                    break;
                }
                _75 = _75 && all(equal(_29[_81]._m0[_83][_85], vec4(0.0)));
                _85++;
                continue;
            }
            _83++;
            continue;
        }
        uint _87 = 0u;
        for (;;)
        {
            if (all(equal(uvec2(0u), _311)))
            {
                break;
            }
            _311 -= uvec2(uint(_311.y == 0u), 1u);
            if (!(_87 < 19u))
            {
                break;
            }
            uint _333 = atomicAdd(_29[_81]._m1[_87], 0u);
            _75 = _75 && (_333 == 0u);
            _87++;
            continue;
        }
        uint _89 = 0u;
        for (;;)
        {
            if (all(equal(uvec2(0u), _344)))
            {
                break;
            }
            _344 -= uvec2(uint(_344.y == 0u), 1u);
            if (!(_89 < 3u))
            {
                break;
            }
            _75 = _75 && all(equal(_29[_81]._m2[_89], vec2(0.0)));
            _75 = _75 && all(equal(_29[_81]._m3[_89][0u], vec2(0.0)));
            _75 = _75 && all(equal(_29[_81]._m3[_89][1u], vec2(0.0)));
            _89++;
            continue;
        }
        _81++;
        continue;
    }
    for (;;)
    {
        if (all(equal(uvec2(0u), _394)))
        {
            break;
        }
        _394 -= uvec2(uint(_394.y == 0u), 1u);
        if (!(_91 < 37u))
        {
            break;
        }
        _75 = _75 && (_31[_91] == 0u);
        _75 = _75 && (_33[_91] == 0.0);
        _91++;
        continue;
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u)
    {
        _37 = 42u;
        _39 = uvec3(42u);
        _41 = mat2x3(vec3(42.0), vec3(42.0));
        for (;;)
        {
            if (all(equal(uvec2(0u), _432)))
            {
                break;
            }
            _432 -= uvec2(uint(_432.y == 0u), 1u);
            if (!(_92 < 3u))
            {
                break;
            }
            uint _93 = 0u;
            for (;;)
            {
                if (all(equal(uvec2(0u), _453)))
                {
                    break;
                }
                _453 -= uvec2(uint(_453.y == 0u), 1u);
                if (!(_93 < 5u))
                {
                    break;
                }
                uint _95 = 0u;
                for (;;)
                {
                    if (all(equal(uvec2(0u), _474)))
                    {
                        break;
                    }
                    _474 -= uvec2(uint(_474.y == 0u), 1u);
                    if (!(_95 < 3u))
                    {
                        break;
                    }
                    _29[_92]._m0[_93][_95] = vec4(42.0);
                    _95++;
                    continue;
                }
                _93++;
                continue;
            }
            uint _97 = 0u;
            for (;;)
            {
                if (all(equal(uvec2(0u), _503)))
                {
                    break;
                }
                _503 -= uvec2(uint(_503.y == 0u), 1u);
                if (!(_97 < 19u))
                {
                    break;
                }
                atomicExchange(_29[_92]._m1[_97], 42u);
                _97++;
                continue;
            }
            _92++;
            continue;
        }
        for (;;)
        {
            if (all(equal(uvec2(0u), _531)))
            {
                break;
            }
            _531 -= uvec2(uint(_531.y == 0u), 1u);
            if (!(_99 < 37u))
            {
                break;
            }
            _31[_99] = 42u;
            _33[_99] = 42.0;
            _99++;
            continue;
        }
    }
    barrier();
    if (_37 == 42u)
    {
        _100 = all(equal(_39, uvec3(42u)));
    }
    else
    {
        _100 = false;
    }
    _75 = _75 && _100;
    _75 = _75 && all(equal(_41[0u], vec3(42.0)));
    _75 = _75 && all(equal(_29[2u]._m0[4u][2u], vec4(42.0)));
    uint _581 = atomicAdd(_29[2u]._m1[18u], 0u);
    _75 = _75 && (_581 == 42u);
    _75 = _75 && (_31[36u] == 42u);
    _75 = _75 && (_33[36u] == 42.0);
    uint _597 = ((gl_WorkGroupID.x * 8u) * 2u) + gl_LocalInvocationIndex;
    _43._m0[_597] |= (_75 ? 2u : 1u);
}


//////////////////////////////////
// Entry point: "single" (comp) //
//////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _16
{
    vec4 _m0[5][3];
    uint _m1[19];
    mat3x2 _m2;
    mat2 _m3[3];
};

struct _17
{
    mat2 _m0;
};

layout(set = 0, binding = 0, std430) buffer _44_43
{
    uint _m0[];
} _43;

shared _16 _29[3];
shared uint _31[37];
shared float _33[37];
shared _17 _35;
shared uint _37;
shared uvec3 _39;
shared mat2x3 _41;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _37 = 0u;
    }
    barrier();
    _43._m0[0u] = _37;
}

