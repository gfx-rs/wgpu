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

struct _41
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _16;

uvec3 _24()
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
    uvec3 _38 = _24();
    barrier();
    _41 _48;
    umulExtended(_38.x, _38.y, _48._m1, _48._m0);
    _41 _50;
    umulExtended(_48._m0, _38.z, _50._m1, _50._m0);
    uvec3 _63 = ((((((_50._m0 > 1024u) || (_38.x > 256u)) || (_38.y > 256u)) || (_38.z > 256u)) || (_48._m1 != 0u)) || (_50._m1 != 0u)) ? uvec3(0u) : _38;
    EmitMeshTasksEXT(_63.x, _63.y, _63.z);
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

struct _41
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _16;
shared _15 _18;

void _78()
{
    if (gl_LocalInvocationIndex < 2u)
    {
        _18._m0[gl_LocalInvocationIndex] = _7(vec4(0.0));
    }
    if (gl_LocalInvocationIndex < 1u)
    {
        _18._m1[gl_LocalInvocationIndex] = _9(uvec2(0u));
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
    _78();
    barrier();
    uint _110 = min(_18._m2, 2u);
    uint _113 = min(_18._m3, 1u);
    SetMeshOutputsEXT(_110, _113);
    for (uint _103 = gl_LocalInvocationIndex; _103 < _110; _103 += 64u)
    {
        gl_MeshVerticesEXT[_103].gl_Position = _18._m0[_103]._m0;
    }
    for (uint _104 = gl_LocalInvocationIndex; _104 < _113; _104 += 64u)
    {
        gl_PrimitiveLineIndicesEXT[_104] = _18._m1[_104]._m0;
    }
}

