#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3, max_primitives = 1, triangles) out;

out gl_MeshPerVertexEXT
{
    vec4 gl_Position;
    float gl_ClipDistance[2];
} gl_MeshVerticesEXT[];

struct _8
{
    vec4 _m0;
    float _m1[2];
};

struct _10
{
    uvec3 _m0;
};

struct _15
{
    _8 _m0[3];
    _10 _m1[1];
    uint _m2;
    uint _m3;
};

shared _15 _16;

void _29()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _16 = _15(_8[](_8(vec4(0.0), float[](0.0, 0.0)), _8(vec4(0.0), float[](0.0, 0.0)), _8(vec4(0.0), float[](0.0, 0.0))), _10[](_10(uvec3(0u))), 0u, 0u);
    }
    barrier();
    _16._m2 = 3u;
    _16._m3 = 1u;
    _16._m0[0u]._m0 = vec4(0.0, 1.0, 0.0, 1.0);
    _16._m0[0u]._m1[0u] = 0.5;
    _16._m0[0u]._m1[1u] = -0.5;
    _16._m0[1u]._m0 = vec4(-1.0, -1.0, 0.0, 1.0);
    _16._m0[1u]._m1[0u] = 0.5;
    _16._m0[1u]._m1[1u] = -0.5;
    _16._m0[2u]._m0 = vec4(1.0, -1.0, 0.0, 1.0);
    _16._m0[2u]._m1[0u] = 0.5;
    _16._m0[2u]._m1[1u] = -0.5;
    _16._m1[0u]._m0 = uvec3(0u, 1u, 2u);
}

void main()
{
    _29();
    barrier();
    uint _79 = min(_16._m2, 3u);
    uint _82 = min(_16._m3, 1u);
    SetMeshOutputsEXT(_79, _82);
    for (uint _72 = gl_LocalInvocationIndex; _72 < _79; _72++)
    {
        gl_MeshVerticesEXT[_72].gl_Position = _16._m0[_72]._m0;
        gl_MeshVerticesEXT[_72].gl_ClipDistance = _16._m0[_72]._m1;
    }
    for (uint _73 = gl_LocalInvocationIndex; _73 < _82; _73++)
    {
        gl_PrimitiveTriangleIndicesEXT[_73] = _16._m1[_73]._m0;
    }
}

