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

struct _8
{
    uint _m0;
};

struct _13
{
    _7 _m0[1];
    _8 _m1[1];
    uint _m2;
    uint _m3;
};

struct _41
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _14;

uvec3 _19()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _14 = _4(0u);
    }
    barrier();
    return uvec3(1u);
}

void main()
{
    uvec3 _38 = _19();
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
layout(max_vertices = 1, max_primitives = 1, points) out;

struct _4
{
    uint _m0;
};

struct _7
{
    vec4 _m0;
};

struct _8
{
    uint _m0;
};

struct _13
{
    _7 _m0[1];
    _8 _m1[1];
    uint _m2;
    uint _m3;
};

struct _41
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _14;
shared _13 _16;

void _78()
{
    for (uint _86 = gl_LocalInvocationIndex; _86 < 1u; _86 += 64u)
    {
        _16._m0[_86] = _7(vec4(0.0));
    }
    for (uint _96 = gl_LocalInvocationIndex; _96 < 1u; _96 += 64u)
    {
        _16._m1[_96] = _8(0u);
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _16._m2 = 0u;
        _16._m3 = 0u;
    }
    barrier();
}

void main()
{
    _78();
    barrier();
    uint _118 = min(_16._m2, 1u);
    uint _121 = min(_16._m3, 1u);
    SetMeshOutputsEXT(_118, _121);
    for (uint _111 = gl_LocalInvocationIndex; _111 < _118; _111 += 64u)
    {
        gl_MeshVerticesEXT[_111].gl_Position = _16._m0[_111]._m0;
    }
    for (uint _112 = gl_LocalInvocationIndex; _112 < _121; _112 += 64u)
    {
        gl_PrimitivePointIndicesEXT[_112] = _16._m1[_112]._m0;
    }
}

