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
    uvec2 _m0;
};

struct _15
{
    _7 _m0[2];
    _9 _m1[1];
    uint _m2;
    uint _m3;
};

struct _42
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _16;

uvec3 _21()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _16 = _4(0u);
    }
    barrier();
    return uvec3(1u);
}

void main()
{
    uvec3 _39 = _21();
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
layout(max_vertices = 2, max_primitives = 1, lines) out;

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
    uvec2 _m0;
};

struct _15
{
    _7 _m0[2];
    _9 _m1[1];
    uint _m2;
    uint _m3;
};

struct _42
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _16;
shared _15 _18;

void _78()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _18 = _15(_7[](_7(vec4(0.0)), _7(vec4(0.0))), _9[](_9(uvec2(0u))), 0u, 0u);
    }
    barrier();
}

void main()
{
    _78();
    barrier();
    uint _97 = min(_18._m2, 2u);
    uint _101 = min(_18._m3, 1u);
    SetMeshOutputsEXT(_97, _101);
    for (uint _89 = gl_LocalInvocationIndex; _89 < _97; _89 += 64u)
    {
        gl_MeshVerticesEXT[_89].gl_Position = _18._m0[_89]._m0;
    }
    for (uint _90 = gl_LocalInvocationIndex; _90 < _101; _90 += 64u)
    {
        gl_PrimitiveLineIndicesEXT[_90] = _18._m1[_90]._m0;
    }
}

