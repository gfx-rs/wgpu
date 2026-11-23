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

struct _17
{
    _10 _m0[3];
    _12 _m1[1];
    uint _m2;
    uint _m3;
};

struct _68
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _9 _18;

uvec3 _26()
{
    uint _37 = 0u;
    _37 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (_37 >= 2u)
        {
            break;
        }
        _18._m0[_37] = vec4(0.0);
        _37++;
        continue;
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _18._m1 = false;
    }
    barrier();
    _18._m0[0u] = vec4(1.0, 0.0, 0.0, 1.0);
    _18._m0[1u] = vec4(0.0, 1.0, 0.0, 1.0);
    _18._m1 = true;
    return uvec3(1u);
}

void main()
{
    uvec3 _65 = _26();
    barrier();
    _68 _75;
    umulExtended(_65.x, _65.y, _75._m1, _75._m0);
    _68 _77;
    umulExtended(_75._m0, _65.z, _77._m1, _77._m0);
    uvec3 _90 = ((((((_77._m0 > 1024u) || (_65.x > 256u)) || (_65.y > 256u)) || (_65.z > 256u)) || (_75._m1 != 0u)) || (_77._m1 != 0u)) ? uvec3(0u) : _65;
    EmitMeshTasksEXT(_90.x, _90.y, _90.z);
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

struct _17
{
    _10 _m0[3];
    _12 _m1[1];
    uint _m2;
    uint _m3;
};

struct _68
{
    uint _m0;
    uint _m1;
};

layout(location = 0) out vec4 _103[3];
taskPayloadSharedEXT _9 _18;
shared _17 _20;

void _107()
{
    vec4 _117 = vec4(0.0);
    uint _121 = 0u;
    vec4 _114 = vec4(0.0);
    vec4 _119 = vec4(0.0);
    _121 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (_121 >= 3u)
        {
            break;
        }
        _20._m0[_121] = _10(vec4(0.0), vec4(0.0));
        _121++;
        continue;
    }
    _20._m1[gl_LocalInvocationIndex] = _12(uvec3(0u));
    if (gl_LocalInvocationIndex == 0u)
    {
        _20._m2 = 0u;
        _20._m3 = 0u;
    }
    barrier();
    _20._m2 = 3u;
    _20._m3 = 1u;
    _20._m0[0u]._m0 = vec4(0.0, 1.0, 0.0, 1.0);
    _114 = _18._m0[0u];
    _20._m0[0u]._m1 = _114;
    _20._m0[1u]._m0 = vec4(-1.0, -1.0, 0.0, 1.0);
    _117 = _18._m0[1u];
    _20._m0[1u]._m1 = _117;
    _20._m0[2u]._m0 = vec4(1.0, -1.0, 0.0, 1.0);
    _119 = _18._m0[0u];
    _20._m0[2u]._m1 = _119;
    _20._m1[0u]._m0 = uvec3(0u, 1u, 2u);
}

void main()
{
    _107();
    barrier();
    uint _173 = min(_20._m2, 3u);
    uint _176 = min(_20._m3, 1u);
    SetMeshOutputsEXT(_173, _176);
    for (uint _167 = gl_LocalInvocationIndex; _167 < _173; _167++)
    {
        gl_MeshVerticesEXT[_167].gl_Position = _20._m0[_167]._m0;
        _103[_167] = _20._m0[_167]._m1;
    }
    for (uint _168 = gl_LocalInvocationIndex; _168 < _176; _168++)
    {
        gl_PrimitiveTriangleIndicesEXT[_168] = _20._m1[_168]._m0;
    }
}

