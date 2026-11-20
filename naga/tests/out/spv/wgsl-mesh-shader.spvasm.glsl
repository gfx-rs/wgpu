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

struct _68
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

uvec3 _37()
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
    uvec3 _65 = _37();
    barrier();
    _68 _75;
    umulExtended(_65.x, _65.y, _75._m1, _75._m0);
    _68 _77;
    umulExtended(_75._m0, _65.z, _77._m1, _77._m0);
    uvec3 _90 = ((((((_77._m0 > 1024u) || (_65.x > 256u)) || (_65.y > 256u)) || (_65.z > 256u)) || (_75._m1 != 0u)) || (_77._m1 != 0u)) ? uvec3(0u) : _65;
    EmitMeshTasksEXT(_90.x, _90.y, _90.z);
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

struct _68
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
    uvec3 _115 = _98();
    barrier();
    _68 _122;
    umulExtended(_115.x, _115.y, _122._m1, _122._m0);
    _68 _124;
    umulExtended(_122._m0, _115.z, _124._m1, _124._m0);
    uvec3 _137 = ((((((_124._m0 > 1024u) || (_115.x > 256u)) || (_115.y > 256u)) || (_115.z > 256u)) || (_122._m1 != 0u)) || (_124._m1 != 0u)) ? uvec3(0u) : _115;
    EmitMeshTasksEXT(_137.x, _137.y, _137.z);
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

struct _68
{
    uint _m0;
    uint _m1;
};

layout(location = 0) out vec4 _154[3];
layout(location = 1) perprimitiveEXT out vec4 _160[1];
taskPayloadSharedEXT _6 _17;
shared float _19;
shared _16 _21;

bool _24()
{
    return _17._m1;
}

void _161()
{
    for (uint _178 = gl_LocalInvocationIndex; _178 < 3u; _178++)
    {
        _21._m0[_178] = _7(vec4(0.0), vec4(0.0));
    }
    for (uint _188 = gl_LocalInvocationIndex; _188 < 1u; _188++)
    {
        _21._m1[_188] = _10(uvec3(0u), false, vec4(0.0));
    }
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
    _161();
    barrier();
    uint _236 = min(_21._m2, 3u);
    uint _239 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_236, _239);
    for (uint _229 = gl_LocalInvocationIndex; _229 < _236; _229++)
    {
        gl_MeshVerticesEXT[_229].gl_Position = _21._m0[_229]._m0;
        _154[_229] = _21._m0[_229]._m1;
    }
    for (uint _230 = gl_LocalInvocationIndex; _230 < _239; _230++)
    {
        gl_PrimitiveTriangleIndicesEXT[_230] = _21._m1[_230]._m0;
        gl_MeshPrimitivesEXT[_230].gl_CullPrimitiveEXT = _21._m1[_230]._m1;
        _160[_230] = _21._m1[_230]._m2;
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

struct _68
{
    uint _m0;
    uint _m1;
};

layout(location = 0) out vec4 _293[3];
layout(location = 1) perprimitiveEXT out vec4 _299[1];
shared float _19;
shared _16 _21;

void _300()
{
    for (uint _307 = gl_LocalInvocationIndex; _307 < 3u; _307++)
    {
        _21._m0[_307] = _7(vec4(0.0), vec4(0.0));
    }
    for (uint _315 = gl_LocalInvocationIndex; _315 < 1u; _315++)
    {
        _21._m1[_315] = _10(uvec3(0u), false, vec4(0.0));
    }
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
    _300();
    barrier();
    uint _344 = min(_21._m2, 3u);
    uint _347 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_344, _347);
    for (uint _338 = gl_LocalInvocationIndex; _338 < _344; _338++)
    {
        gl_MeshVerticesEXT[_338].gl_Position = _21._m0[_338]._m0;
        _293[_338] = _21._m0[_338]._m1;
    }
    for (uint _339 = gl_LocalInvocationIndex; _339 < _347; _339++)
    {
        gl_PrimitiveTriangleIndicesEXT[_339] = _21._m1[_339]._m0;
        gl_MeshPrimitivesEXT[_339].gl_CullPrimitiveEXT = _21._m1[_339]._m1;
        _299[_339] = _21._m1[_339]._m2;
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

struct _68
{
    uint _m0;
    uint _m1;
};

layout(location = 0) out vec4 _400[3];
layout(location = 1) perprimitiveEXT out vec4 _406[1];
shared float _19;
shared _16 _21;

void _407()
{
    for (uint _413 = gl_LocalInvocationIndex; _413 < 3u; _413 += 2u)
    {
        _21._m0[_413] = _7(vec4(0.0), vec4(0.0));
    }
    for (uint _421 = gl_LocalInvocationIndex; _421 < 1u; _421 += 2u)
    {
        _21._m1[_421] = _10(uvec3(0u), false, vec4(0.0));
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
    _407();
    barrier();
    uint _455 = min(_21._m2, 3u);
    uint _458 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_455, _458);
    for (uint _449 = gl_LocalInvocationIndex; _449 < _455; _449 += 2u)
    {
        gl_MeshVerticesEXT[_449].gl_Position = _21._m0[_449]._m0;
        _400[_449] = _21._m0[_449]._m1;
    }
    for (uint _450 = gl_LocalInvocationIndex; _450 < _458; _450 += 2u)
    {
        gl_PrimitiveTriangleIndicesEXT[_450] = _21._m1[_450]._m0;
        gl_MeshPrimitivesEXT[_450].gl_CullPrimitiveEXT = _21._m1[_450]._m1;
        _406[_450] = _21._m1[_450]._m2;
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

struct _68
{
    uint _m0;
    uint _m1;
};

layout(location = 0) in vec4 _501;
layout(location = 1) perprimitiveEXT in vec4 _504;
layout(location = 0) out vec4 _506;

void main()
{
    _506 = _7(gl_FragCoord, _501)._m1 * _11(_504)._m0;
}

