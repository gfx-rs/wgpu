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

void _79()
{
    for (uint _87 = gl_LocalInvocationIndex; _87 < 2u; _87 += 64u)
    {
        _18._m0[_87] = _7(vec4(0.0));
    }
    for (uint _97 = gl_LocalInvocationIndex; _97 < 1u; _97 += 64u)
    {
        _18._m1[_97] = _9(uvec2(0u));
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _18._m2 = 0u;
        _18._m3 = 0u;
    }
    barrier();
}

void main()
{
    _79();
    barrier();
    uint _119 = min(_18._m2, 2u);
    uint _122 = min(_18._m3, 1u);
    SetMeshOutputsEXT(_119, _122);
    for (uint _112 = gl_LocalInvocationIndex; _112 < _119; _112 += 64u)
    {
        gl_MeshVerticesEXT[_112].gl_Position = _18._m0[_112]._m0;
    }
    for (uint _113 = gl_LocalInvocationIndex; _113 < _122; _113 += 64u)
    {
        gl_PrimitiveLineIndicesEXT[_113] = _18._m1[_113]._m0;
    }
}

