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

void _79()
{
    for (uint _87 = gl_LocalInvocationIndex; _87 < 3u; _87 += 64u)
    {
        _17._m0[_87] = _7(vec4(0.0));
    }
    for (uint _97 = gl_LocalInvocationIndex; _97 < 1u; _97 += 64u)
    {
        _17._m1[_97] = _9(uvec3(0u));
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
    _79();
    barrier();
    uint _118 = min(_17._m2, 3u);
    uint _121 = min(_17._m3, 1u);
    SetMeshOutputsEXT(_118, _121);
    for (uint _111 = gl_LocalInvocationIndex; _111 < _118; _111 += 64u)
    {
        gl_MeshVerticesEXT[_111].gl_Position = _17._m0[_111]._m0;
    }
    for (uint _112 = gl_LocalInvocationIndex; _112 < _121; _112 += 64u)
    {
        gl_PrimitiveTriangleIndicesEXT[_112] = _17._m1[_112]._m0;
    }
}

