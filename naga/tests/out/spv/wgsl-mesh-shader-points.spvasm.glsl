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

struct _40
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _14;

uvec3 _22()
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
    uvec3 _37 = _22();
    barrier();
    _40 _47;
    umulExtended(_37.x, _37.y, _47._m1, _47._m0);
    _40 _49;
    umulExtended(_47._m0, _37.z, _49._m1, _49._m0);
    uvec3 _62 = ((((((_49._m0 > 1024u) || (_37.x > 256u)) || (_37.y > 256u)) || (_37.z > 256u)) || (_47._m1 != 0u)) || (_49._m1 != 0u)) ? uvec3(0u) : _37;
    EmitMeshTasksEXT(_62.x, _62.y, _62.z);
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

struct _40
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _14;
shared _13 _16;

void _77()
{
    if (gl_LocalInvocationIndex < 1u)
    {
        _16._m0[gl_LocalInvocationIndex] = _7(vec4(0.0));
        _16._m1[gl_LocalInvocationIndex] = _8(0u);
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
    _77();
    barrier();
    uint _106 = min(_16._m2, 1u);
    uint _109 = min(_16._m3, 1u);
    SetMeshOutputsEXT(_106, _109);
    for (uint _99 = gl_LocalInvocationIndex; _99 < _106; _99 += 64u)
    {
        gl_MeshVerticesEXT[_99].gl_Position = _16._m0[_99]._m0;
    }
    for (uint _100 = gl_LocalInvocationIndex; _100 < _109; _100 += 64u)
    {
        gl_PrimitivePointIndicesEXT[_100] = _16._m1[_100]._m0;
    }
}

