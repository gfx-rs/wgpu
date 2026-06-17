///////////////////////////////////
// Entry point: "ts_main" (task) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct _4
{
    uint _m0;
};

struct _7
{
    vec4 _m0;
};

struct _9
{
    uvec3 _m0;
};

struct _14
{
    _7 _m0[3];
    _9 _m1[1];
    uint _m2;
    uint _m3;
};

struct _42
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _15;

uvec3 _20()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _15 = _4(0u);
    }
    barrier();
    return uvec3(1u);
}

void main()
{
    uvec3 _39 = _20();
    barrier();
    _42 _49;
    umulExtended(_39.x, _39.y, _49._m1, _49._m0);
    _42 _51;
    umulExtended(_49._m0, _39.z, _51._m1, _51._m0);
    uvec3 _64 = ((((((_51._m0 > 1024u) || (_39.x > 256u)) || (_39.y > 256u)) || (_39.z > 256u)) || (_49._m1 != 0u)) || (_51._m1 != 0u)) ? uvec3(0u) : _39;
    EmitMeshTasksEXT(_64.x, _64.y, _64.z);
}


///////////////////////////////////
// Entry point: "ms_main" (mesh) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3, max_primitives = 1, triangles) out;

struct _4
{
    uint _m0;
};

struct _7
{
    vec4 _m0;
};

struct _9
{
    uvec3 _m0;
};

struct _14
{
    _7 _m0[3];
    _9 _m1[1];
    uint _m2;
    uint _m3;
};

struct _42
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _15;
shared _14 _17;

void _78()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _17 = _14(_7[](_7(vec4(0.0)), _7(vec4(0.0)), _7(vec4(0.0))), _9[](_9(uvec3(0u))), 0u, 0u);
    }
    barrier();
}

void main()
{
    _78();
    barrier();
    uint _97 = min(_17._m2, 3u);
    uint _100 = min(_17._m3, 1u);
    SetMeshOutputsEXT(_97, _100);
    for (uint _89 = gl_LocalInvocationIndex; _89 < _97; _89 += 64u)
    {
        gl_MeshVerticesEXT[_89].gl_Position = _17._m0[_89]._m0;
    }
    for (uint _90 = gl_LocalInvocationIndex; _90 < _100; _90 += 64u)
    {
        gl_PrimitiveTriangleIndicesEXT[_90] = _17._m1[_90]._m0;
    }
}

