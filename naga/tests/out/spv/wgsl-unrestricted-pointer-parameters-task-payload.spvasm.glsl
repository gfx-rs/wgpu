///////////////////////////////////
// Entry point: "ts_main" (task) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _9
{
    vec4 _m0[2];
    bool _m1;
};

struct _10
{
    vec4 _m0;
    vec4 _m1;
};

struct _12
{
    uvec3 _m0;
};

struct _19
{
    _10 _m0[3];
    _12 _m1[1];
    uint _m2;
    uint _m3;
};

struct _54
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _9 _20;

uvec3 _24()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _20 = _9(vec4[](vec4(0.0), vec4(0.0)), false);
    }
    barrier();
    _20._m0[0u] = vec4(1.0, 0.0, 0.0, 1.0);
    _20._m0[1u] = vec4(0.0, 1.0, 0.0, 1.0);
    _20._m1 = true;
    return uvec3(1u);
}

void main()
{
    uvec3 _51 = _24();
    barrier();
    _54 _61;
    umulExtended(_51.x, _51.y, _61._m1, _61._m0);
    _54 _63;
    umulExtended(_61._m0, _51.z, _63._m1, _63._m0);
    uvec3 _76 = ((((((_63._m0 > 1024u) || (_51.x > 256u)) || (_51.y > 256u)) || (_51.z > 256u)) || (_61._m1 != 0u)) || (_63._m1 != 0u)) ? uvec3(0u) : _51;
    EmitMeshTasksEXT(_76.x, _76.y, _76.z);
}


///////////////////////////////////
// Entry point: "ms_main" (mesh) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 3, max_primitives = 1, triangles) out;

struct _9
{
    vec4 _m0[2];
    bool _m1;
};

struct _10
{
    vec4 _m0;
    vec4 _m1;
};

struct _12
{
    uvec3 _m0;
};

struct _19
{
    _10 _m0[3];
    _12 _m1[1];
    uint _m2;
    uint _m3;
};

struct _54
{
    uint _m0;
    uint _m1;
};

layout(location = 0) out vec4 _89[3];
taskPayloadSharedEXT _9 _20;
shared _19 _21;

void _93()
{
    vec4 _98 = vec4(0.0);
    vec4 _101 = vec4(0.0);
    vec4 _103 = vec4(0.0);
    if (gl_LocalInvocationIndex == 0u)
    {
        _21 = _19(_10[](_10(vec4(0.0), vec4(0.0)), _10(vec4(0.0), vec4(0.0)), _10(vec4(0.0), vec4(0.0))), _12[](_12(uvec3(0u))), 0u, 0u);
    }
    barrier();
    _21._m2 = 3u;
    _21._m3 = 1u;
    _21._m0[0u]._m0 = vec4(0.0, 1.0, 0.0, 1.0);
    _98 = _20._m0[0u];
    _21._m0[0u]._m1 = _98;
    _21._m0[1u]._m0 = vec4(-1.0, -1.0, 0.0, 1.0);
    _101 = _20._m0[1u];
    _21._m0[1u]._m1 = _101;
    _21._m0[2u]._m0 = vec4(1.0, -1.0, 0.0, 1.0);
    _103 = _20._m0[0u];
    _21._m0[2u]._m1 = _103;
    _21._m1[0u]._m0 = uvec3(0u, 1u, 2u);
}

void main()
{
    _93();
    barrier();
    uint _145 = min(_21._m2, 3u);
    uint _148 = min(_21._m3, 1u);
    SetMeshOutputsEXT(_145, _148);
    for (uint _138 = gl_LocalInvocationIndex; _138 < _145; _138++)
    {
        gl_MeshVerticesEXT[_138].gl_Position = _21._m0[_138]._m0;
        _89[_138] = _21._m0[_138]._m1;
    }
    for (uint _139 = gl_LocalInvocationIndex; _139 < _148; _139++)
    {
        gl_PrimitiveTriangleIndicesEXT[_139] = _21._m1[_139]._m0;
    }
}

