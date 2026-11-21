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

struct _41
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _15;

uvec3 _23()
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
    uvec3 _38 = _23();
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

struct _41
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _15;
shared _14 _17;

void _78()
{
    if (gl_LocalInvocationIndex < 3u)
    {
        _17._m0[gl_LocalInvocationIndex] = _7(vec4(0.0));
    }
    if (gl_LocalInvocationIndex < 1u)
    {
        _17._m1[gl_LocalInvocationIndex] = _9(uvec3(0u));
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _17._m2 = 0u;
        _17._m3 = 0u;
    }
    barrier();
}

void main()
{
    _78();
    barrier();
    uint _109 = min(_17._m2, 3u);
    uint _112 = min(_17._m3, 1u);
    SetMeshOutputsEXT(_109, _112);
    for (uint _102 = gl_LocalInvocationIndex; _102 < _109; _102 += 64u)
    {
        gl_MeshVerticesEXT[_102].gl_Position = _17._m0[_102]._m0;
    }
    for (uint _103 = gl_LocalInvocationIndex; _103 < _112; _103 += 64u)
    {
        gl_PrimitiveTriangleIndicesEXT[_103] = _17._m1[_103]._m0;
    }
}

