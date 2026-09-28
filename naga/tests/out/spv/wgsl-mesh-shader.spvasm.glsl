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
    if (gl_LocalInvocationIndex == 0u)
    {
        _19 = 0.0;
        _21 = _16(_7[](_7(vec4(0.0), vec4(0.0)), _7(vec4(0.0), vec4(0.0)), _7(vec4(0.0), vec4(0.0))), _10[](_10(uvec3(0u), false, vec4(0.0))), 0u, 0u);
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
    uint _216 = min(_21._m2, 3u);
    uint _219 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_216, _219);
    for (uint _209 = gl_LocalInvocationIndex; _209 < _216; _209++)
    {
        gl_MeshVerticesEXT[_209].gl_Position = _21._m0[_209]._m0;
        _154[_209] = _21._m0[_209]._m1;
    }
    for (uint _210 = gl_LocalInvocationIndex; _210 < _219; _210++)
    {
        gl_PrimitiveTriangleIndicesEXT[_210] = _21._m1[_210]._m0;
        gl_MeshPrimitivesEXT[_210].gl_CullPrimitiveEXT = _21._m1[_210]._m1;
        _160[_210] = _21._m1[_210]._m2;
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

layout(location = 0) out vec4 _273[3];
layout(location = 1) perprimitiveEXT out vec4 _279[1];
shared float _19;
shared _16 _21;

void _280()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _19 = 0.0;
        _21 = _16(_7[](_7(vec4(0.0), vec4(0.0)), _7(vec4(0.0), vec4(0.0)), _7(vec4(0.0), vec4(0.0))), _10[](_10(uvec3(0u), false, vec4(0.0))), 0u, 0u);
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
    _280();
    barrier();
    uint _306 = min(_21._m2, 3u);
    uint _309 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_306, _309);
    for (uint _300 = gl_LocalInvocationIndex; _300 < _306; _300++)
    {
        gl_MeshVerticesEXT[_300].gl_Position = _21._m0[_300]._m0;
        _273[_300] = _21._m0[_300]._m1;
    }
    for (uint _301 = gl_LocalInvocationIndex; _301 < _309; _301++)
    {
        gl_PrimitiveTriangleIndicesEXT[_301] = _21._m1[_301]._m0;
        gl_MeshPrimitivesEXT[_301].gl_CullPrimitiveEXT = _21._m1[_301]._m1;
        _279[_301] = _21._m1[_301]._m2;
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

layout(location = 0) out vec4 _362[3];
layout(location = 1) perprimitiveEXT out vec4 _368[1];
shared float _19;
shared _16 _21;

void _369()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _19 = 0.0;
        _21 = _16(_7[](_7(vec4(0.0), vec4(0.0)), _7(vec4(0.0), vec4(0.0)), _7(vec4(0.0), vec4(0.0))), _10[](_10(uvec3(0u), false, vec4(0.0))), 0u, 0u);
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
    _369();
    barrier();
    uint _399 = min(_21._m2, 3u);
    uint _402 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_399, _402);
    for (uint _393 = gl_LocalInvocationIndex; _393 < _399; _393 += 2u)
    {
        gl_MeshVerticesEXT[_393].gl_Position = _21._m0[_393]._m0;
        _362[_393] = _21._m0[_393]._m1;
    }
    for (uint _394 = gl_LocalInvocationIndex; _394 < _402; _394 += 2u)
    {
        gl_PrimitiveTriangleIndicesEXT[_394] = _21._m1[_394]._m0;
        gl_MeshPrimitivesEXT[_394].gl_CullPrimitiveEXT = _21._m1[_394]._m1;
        _368[_394] = _21._m1[_394]._m2;
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

layout(location = 0) in vec4 _445;
layout(location = 1) perprimitiveEXT in vec4 _448;
layout(location = 0) out vec4 _450;

void main()
{
    _450 = _7(gl_FragCoord, _445)._m1 * _11(_448)._m0;
}

