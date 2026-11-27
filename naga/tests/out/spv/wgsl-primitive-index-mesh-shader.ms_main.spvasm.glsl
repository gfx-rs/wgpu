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
    uint _45 = 0u;
    uvec2 _60 = uvec2(4294967295u);
    _45 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _60)))
        {
            break;
        }
        _60 -= uvec2(uint(_60.y == 0u), 1u);
        if (_45 >= 3u)
        {
            break;
        }
        _14._m0[_45] = _5(vec4(0.0));
        _45++;
        continue;
    }
    _14._m1[gl_LocalInvocationIndex] = _8(uvec3(0u), 0u);
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
    uint _106 = min(_14._m2, 3u);
    uint _109 = min(_14._m3, 1u);
    SetMeshOutputsEXT(_106, _109);
    for (uint _100 = gl_LocalInvocationIndex; _100 < _106; _100++)
    {
        gl_MeshVerticesEXT[_100].gl_Position = _14._m0[_100]._m0;
    }
    for (uint _101 = gl_LocalInvocationIndex; _101 < _109; _101++)
    {
        gl_PrimitiveTriangleIndicesEXT[_101] = _14._m1[_101]._m0;
        gl_MeshPrimitivesEXT[_101].gl_PrimitiveID = _14._m1[_101]._m1;
    }
}

