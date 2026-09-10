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

void _30()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _14 = _13(_5[](_5(vec4(0.0)), _5(vec4(0.0)), _5(vec4(0.0))), _8[](_8(uvec3(0u), 0u)), 0u, 0u);
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
    _30();
    barrier();
    uint _75 = min(_14._m2, 3u);
    uint _78 = min(_14._m3, 1u);
    SetMeshOutputsEXT(_75, _78);
    for (uint _68 = gl_LocalInvocationIndex; _68 < _75; _68++)
    {
        gl_MeshVerticesEXT[_68].gl_Position = _14._m0[_68]._m0;
    }
    for (uint _69 = gl_LocalInvocationIndex; _69 < _78; _69++)
    {
        gl_PrimitiveTriangleIndicesEXT[_69] = _14._m1[_69]._m0;
        gl_MeshPrimitivesEXT[_69].gl_PrimitiveID = _14._m1[_69]._m1;
    }
}

