#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3, max_primitives = 1, triangles) out;

struct _5
{
    vec4 _m0;
};

struct _8
{
    uvec3 _m0;
    uint _m1;
};

struct _13
{
    _5 _m0[3];
    _8 _m1[1];
    uint _m2;
    uint _m3;
};

shared _13 _14;

void _31()
{
    for (uint _51 = gl_LocalInvocationIndex; _51 < 3u; _51++)
    {
        _14._m0[_51] = _5(vec4(0.0));
    }
    for (uint _61 = gl_LocalInvocationIndex; _61 < 1u; _61++)
    {
        _14._m1[_61] = _8(uvec3(0u), 0u);
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _14._m2 = 0u;
        _14._m3 = 0u;
    }
    barrier();
    _14._m2 = 3u;
    _14._m3 = 1u;
    _14._m0[0u]._m0 = vec4(0.0, 1.0, 0.0, 1.0);
    _14._m0[1u]._m0 = vec4(-1.0, -1.0, 0.0, 1.0);
    _14._m0[2u]._m0 = vec4(1.0, -1.0, 0.0, 1.0);
    _14._m1[0u]._m0 = uvec3(0u, 1u, 2u);
    _14._m1[0u]._m1 = 7u;
}

void main()
{
    _31();
    barrier();
    uint _94 = min(_14._m2, 3u);
    uint _97 = min(_14._m3, 1u);
    SetMeshOutputsEXT(_94, _97);
    for (uint _87 = gl_LocalInvocationIndex; _87 < _94; _87++)
    {
        gl_MeshVerticesEXT[_87].gl_Position = _14._m0[_87]._m0;
    }
    for (uint _88 = gl_LocalInvocationIndex; _88 < _97; _88++)
    {
        gl_PrimitiveTriangleIndicesEXT[_88] = _14._m1[_88]._m0;
        gl_MeshPrimitivesEXT[_88].gl_PrimitiveID = _14._m1[_88]._m1;
    }
}

