///////////////////////////////////
// Entry point: "ts_main" (task) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    vec4 _m0;
    bool _m1;
};

struct _7
{
    vec4 _m0;
    vec4 _m1;
};

struct _10
{
    uvec3 _m0;
    bool _m1;
    vec4 _m2;
};

struct _11
{
    vec4 _m0;
};

struct _16
{
    _7 _m0[3];
    _10 _m1[1];
    uint _m2;
    uint _m3;
};

struct _66
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _6 _17;
shared float _19;

void _32(bool _31)
{
    _17._m1 = _31;
}

bool _24()
{
    return _17._m1;
}

uvec3 _40()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _17 = _6(vec4(0.0), false);
        _19 = 0.0;
    }
    barrier();
    _19 = 1.0;
    _17._m0 = vec4(1.0, 1.0, 0.0, 1.0);
    _32(true);
    _17._m1 = _24();
    return uvec3(1u);
}

void main()
{
    uvec3 _63 = _40();
    barrier();
    _66 _73;
    umulExtended(_63.x, _63.y, _73._m1, _73._m0);
    _66 _75;
    umulExtended(_73._m0, _63.z, _75._m1, _75._m0);
    uvec3 _88 = ((((((_75._m0 > 1024u) || (_63.x > 256u)) || (_63.y > 256u)) || (_63.z > 256u)) || (_73._m1 != 0u)) || (_75._m1 != 0u)) ? uvec3(0u) : _63;
    EmitMeshTasksEXT(_88.x, _88.y, _88.z);
}


////////////////////////////////////////
// Entry point: "ts_divergent" (task) //
////////////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    vec4 _m0;
    bool _m1;
};

struct _7
{
    vec4 _m0;
    vec4 _m1;
};

struct _10
{
    uvec3 _m0;
    bool _m1;
    vec4 _m2;
};

struct _11
{
    vec4 _m0;
};

struct _16
{
    _7 _m0[3];
    _10 _m1[1];
    uint _m2;
    uint _m3;
};

struct _66
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _6 _17;

uvec3 _98()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _17 = _6(vec4(0.0), false);
    }
    barrier();
    if (gl_LocalInvocationID.x == 0u)
    {
        _17._m0 = vec4(1.0, 1.0, 0.0, 1.0);
        _17._m1 = true;
        return uvec3(1u);
    }
    return uvec3(2u);
}

void main()
{
    uvec3 _112 = _98();
    barrier();
    _66 _119;
    umulExtended(_112.x, _112.y, _119._m1, _119._m0);
    _66 _121;
    umulExtended(_119._m0, _112.z, _121._m1, _121._m0);
    uvec3 _134 = ((((((_121._m0 > 1024u) || (_112.x > 256u)) || (_112.y > 256u)) || (_112.z > 256u)) || (_119._m1 != 0u)) || (_121._m1 != 0u)) ? uvec3(0u) : _112;
    EmitMeshTasksEXT(_134.x, _134.y, _134.z);
}


///////////////////////////////////
// Entry point: "ms_main" (mesh) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3, max_primitives = 1, triangles) out;

struct _6
{
    vec4 _m0;
    bool _m1;
};

struct _7
{
    vec4 _m0;
    vec4 _m1;
};

struct _10
{
    uvec3 _m0;
    bool _m1;
    vec4 _m2;
};

struct _11
{
    vec4 _m0;
};

struct _16
{
    _7 _m0[3];
    _10 _m1[1];
    uint _m2;
    uint _m3;
};

struct _66
{
    uint _m0;
    uint _m1;
};

layout(location = 0) out vec4 _151[3];
layout(location = 1) perprimitiveEXT out vec4 _157[1];
taskPayloadSharedEXT _6 _17;
shared float _19;
shared _16 _21;

bool _24()
{
    return _17._m1;
}

void _158()
{
    uint _170 = 0u;
    _170 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (_170 >= 3u)
        {
            break;
        }
        _21._m0[_170] = _7(vec4(0.0), vec4(0.0));
        _170++;
        continue;
    }
    _21._m1[gl_LocalInvocationIndex] = _10(uvec3(0u), false, vec4(0.0));
    if (gl_LocalInvocationIndex == 0u)
    {
        _19 = 0.0;
        _21._m2 = 0u;
        _21._m3 = 0u;
    }
    barrier();
    _21._m2 = 3u;
    _21._m3 = 1u;
    _19 = 2.0;
    _21._m0[0u]._m0 = vec4(0.0, 1.0, 0.0, 1.0);
    _21._m0[0u]._m1 = vec4(0.0, 1.0, 0.0, 1.0) * _17._m0;
    _21._m0[1u]._m0 = vec4(-1.0, -1.0, 0.0, 1.0);
    _21._m0[1u]._m1 = vec4(0.0, 0.0, 1.0, 1.0) * _17._m0;
    _21._m0[2u]._m0 = vec4(1.0, -1.0, 0.0, 1.0);
    _21._m0[2u]._m1 = vec4(1.0, 0.0, 0.0, 1.0) * _17._m0;
    _21._m1[0u]._m0 = uvec3(0u, 1u, 2u);
    _21._m1[0u]._m1 = !_24();
    _21._m1[0u]._m2 = vec4(1.0, 0.0, 1.0, 1.0);
}

void main()
{
    _158();
    barrier();
    uint _228 = min(_21._m2, 3u);
    uint _231 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_228, _231);
    for (uint _222 = gl_LocalInvocationIndex; _222 < _228; _222++)
    {
        gl_MeshVerticesEXT[_222].gl_Position = _21._m0[_222]._m0;
        _151[_222] = _21._m0[_222]._m1;
    }
    for (uint _223 = gl_LocalInvocationIndex; _223 < _231; _223++)
    {
        gl_PrimitiveTriangleIndicesEXT[_223] = _21._m1[_223]._m0;
        gl_MeshPrimitivesEXT[_223].gl_CullPrimitiveEXT = _21._m1[_223]._m1;
        _157[_223] = _21._m1[_223]._m2;
    }
}


////////////////////////////////////
// Entry point: "ms_no_ts" (mesh) //
////////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3, max_primitives = 1, triangles) out;

struct _6
{
    vec4 _m0;
    bool _m1;
};

struct _7
{
    vec4 _m0;
    vec4 _m1;
};

struct _10
{
    uvec3 _m0;
    bool _m1;
    vec4 _m2;
};

struct _11
{
    vec4 _m0;
};

struct _16
{
    _7 _m0[3];
    _10 _m1[1];
    uint _m2;
    uint _m3;
};

struct _66
{
    uint _m0;
    uint _m1;
};

layout(location = 0) out vec4 _285[3];
layout(location = 1) perprimitiveEXT out vec4 _291[1];
shared float _19;
shared _16 _21;

void _292()
{
    uint _294 = 0u;
    _294 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (_294 >= 3u)
        {
            break;
        }
        _21._m0[_294] = _7(vec4(0.0), vec4(0.0));
        _294++;
        continue;
    }
    _21._m1[gl_LocalInvocationIndex] = _10(uvec3(0u), false, vec4(0.0));
    if (gl_LocalInvocationIndex == 0u)
    {
        _19 = 0.0;
        _21._m2 = 0u;
        _21._m3 = 0u;
    }
    barrier();
    _21._m2 = 3u;
    _21._m3 = 1u;
    _19 = 2.0;
    _21._m0[0u]._m0 = vec4(0.0, 1.0, 0.0, 1.0);
    _21._m0[0u]._m1 = vec4(0.0, 1.0, 0.0, 1.0);
    _21._m0[1u]._m0 = vec4(-1.0, -1.0, 0.0, 1.0);
    _21._m0[1u]._m1 = vec4(0.0, 0.0, 1.0, 1.0);
    _21._m0[2u]._m0 = vec4(1.0, -1.0, 0.0, 1.0);
    _21._m0[2u]._m1 = vec4(1.0, 0.0, 0.0, 1.0);
    _21._m1[0u]._m0 = uvec3(0u, 1u, 2u);
    _21._m1[0u]._m1 = false;
    _21._m1[0u]._m2 = vec4(1.0, 0.0, 1.0, 1.0);
}

void main()
{
    _292();
    barrier();
    uint _332 = min(_21._m2, 3u);
    uint _335 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_332, _335);
    for (uint _326 = gl_LocalInvocationIndex; _326 < _332; _326++)
    {
        gl_MeshVerticesEXT[_326].gl_Position = _21._m0[_326]._m0;
        _285[_326] = _21._m0[_326]._m1;
    }
    for (uint _327 = gl_LocalInvocationIndex; _327 < _335; _327++)
    {
        gl_PrimitiveTriangleIndicesEXT[_327] = _21._m1[_327]._m0;
        gl_MeshPrimitivesEXT[_327].gl_CullPrimitiveEXT = _21._m1[_327]._m1;
        _291[_327] = _21._m1[_327]._m2;
    }
}


////////////////////////////////////////
// Entry point: "ms_divergent" (mesh) //
////////////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3, max_primitives = 1, triangles) out;

struct _6
{
    vec4 _m0;
    bool _m1;
};

struct _7
{
    vec4 _m0;
    vec4 _m1;
};

struct _10
{
    uvec3 _m0;
    bool _m1;
    vec4 _m2;
};

struct _11
{
    vec4 _m0;
};

struct _16
{
    _7 _m0[3];
    _10 _m1[1];
    uint _m2;
    uint _m3;
};

struct _66
{
    uint _m0;
    uint _m1;
};

layout(location = 0) out vec4 _388[3];
layout(location = 1) perprimitiveEXT out vec4 _394[1];
shared float _19;
shared _16 _21;

void _395()
{
    uint _396 = 0u;
    _396 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (_396 >= 3u)
        {
            break;
        }
        _21._m0[_396] = _7(vec4(0.0), vec4(0.0));
        _396 += 2u;
        continue;
    }
    if (gl_LocalInvocationIndex < 1u)
    {
        _21._m1[gl_LocalInvocationIndex] = _10(uvec3(0u), false, vec4(0.0));
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _19 = 0.0;
        _21._m2 = 0u;
        _21._m3 = 0u;
    }
    barrier();
    if (gl_LocalInvocationID.x == 0u)
    {
        _21._m2 = 3u;
        _21._m3 = 1u;
        _19 = 2.0;
        _21._m0[0u]._m0 = vec4(0.0, 1.0, 0.0, 1.0);
        _21._m0[0u]._m1 = vec4(0.0, 1.0, 0.0, 1.0);
        _21._m0[1u]._m0 = vec4(-1.0, -1.0, 0.0, 1.0);
        _21._m0[1u]._m1 = vec4(0.0, 0.0, 1.0, 1.0);
        _21._m0[2u]._m0 = vec4(1.0, -1.0, 0.0, 1.0);
        _21._m0[2u]._m1 = vec4(1.0, 0.0, 0.0, 1.0);
        _21._m1[0u]._m0 = uvec3(0u, 1u, 2u);
        _21._m1[0u]._m1 = false;
        _21._m1[0u]._m2 = vec4(1.0, 0.0, 1.0, 1.0);
        return;
    }
    else
    {
        return;
    }
}

void main()
{
    _395();
    barrier();
    uint _442 = min(_21._m2, 3u);
    uint _445 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_442, _445);
    for (uint _436 = gl_LocalInvocationIndex; _436 < _442; _436 += 2u)
    {
        gl_MeshVerticesEXT[_436].gl_Position = _21._m0[_436]._m0;
        _388[_436] = _21._m0[_436]._m1;
    }
    for (uint _437 = gl_LocalInvocationIndex; _437 < _445; _437 += 2u)
    {
        gl_PrimitiveTriangleIndicesEXT[_437] = _21._m1[_437]._m0;
        gl_MeshPrimitivesEXT[_437].gl_CullPrimitiveEXT = _21._m1[_437]._m1;
        _394[_437] = _21._m1[_437]._m2;
    }
}


///////////////////////////////////
// Entry point: "fs_main" (frag) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require

struct _6
{
    vec4 _m0;
    bool _m1;
};

struct _7
{
    vec4 _m0;
    vec4 _m1;
};

struct _10
{
    uvec3 _m0;
    bool _m1;
    vec4 _m2;
};

struct _11
{
    vec4 _m0;
};

struct _16
{
    _7 _m0[3];
    _10 _m1[1];
    uint _m2;
    uint _m3;
};

struct _66
{
    uint _m0;
    uint _m1;
};

layout(location = 0) in vec4 _488;
layout(location = 1) perprimitiveEXT in vec4 _491;
layout(location = 0) out vec4 _493;

void main()
{
    _493 = _7(gl_FragCoord, _488)._m1 * _11(_491)._m0;
}

