///////////////////////////////////
// Entry point: "ts_main" (task) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    uint _m0[64];
    uint _m1;
};

struct _9
{
    vec4 _m0;
};

struct _11
{
    uvec3 _m0;
};

struct _15
{
    _9 _m0[64];
    _11 _m1[126];
    uint _m2;
    uint _m3;
};

struct _113
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _6 _18;
shared vec4 _20[256];

uvec3 _28()
{
    uint _36 = 0u;
    uvec2 _51 = uvec2(4294967295u);
    uvec2 _73 = uvec2(4294967295u);
    _36 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _51)))
        {
            break;
        }
        _51 -= uvec2(uint(_51.y == 0u), 1u);
        _18._m0[_36] = 0u;
        uint _66 = _36;
        uint _67 = _66 + 32u;
        _36 = _67;
        if (_67 >= 64u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    _36 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _73)))
        {
            break;
        }
        _73 -= uvec2(uint(_73.y == 0u), 1u);
        _20[_36] = vec4(0.0);
        uint _87 = _36;
        uint _88 = _87 + 32u;
        _36 = _88;
        if (_88 >= 256u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _18._m1 = 0u;
    }
    barrier();
    _20[gl_LocalInvocationIndex] = vec4(float(gl_LocalInvocationIndex));
    _18._m0[gl_LocalInvocationIndex] = uint(clamp(_20[gl_LocalInvocationIndex].x, 0.0, 4294967040.0));
    _18._m1 = 32u;
    return uvec3(1u);
}

void main()
{
    uvec3 _111 = _28();
    barrier();
    _113 _120;
    umulExtended(_111.x, _111.y, _120._m1, _120._m0);
    _113 _122;
    umulExtended(_120._m0, _111.z, _122._m1, _122._m0);
    uvec3 _135 = ((((((_122._m0 > 1024u) || (_111.x > 256u)) || (_111.y > 256u)) || (_111.z > 256u)) || (_120._m1 != 0u)) || (_122._m1 != 0u)) ? uvec3(0u) : _111;
    EmitMeshTasksEXT(_135.x, _135.y, _135.z);
}


///////////////////////////////////
// Entry point: "ms_main" (mesh) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = 64, max_primitives = 126, triangles) out;

struct _6
{
    uint _m0[64];
    uint _m1;
};

struct _9
{
    vec4 _m0;
};

struct _11
{
    uvec3 _m0;
};

struct _15
{
    _9 _m0[64];
    _11 _m1[126];
    uint _m2;
    uint _m3;
};

struct _113
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _6 _18;
shared _15 _22;

void _152()
{
    uint _157 = 0u;
    uvec2 _164 = uvec2(4294967295u);
    uvec2 _186 = uvec2(4294967295u);
    _157 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _164)))
        {
            break;
        }
        _164 -= uvec2(uint(_164.y == 0u), 1u);
        _22._m0[_157] = _9(vec4(0.0));
        uint _179 = _157;
        uint _180 = _179 + 32u;
        _157 = _180;
        if (_180 >= 64u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    _157 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _186)))
        {
            break;
        }
        _186 -= uvec2(uint(_186.y == 0u), 1u);
        _22._m1[_157] = _11(uvec3(0u));
        uint _201 = _157;
        uint _202 = _201 + 32u;
        _157 = _202;
        if (_202 >= 126u)
        {
            break;
        }
        else
        {
            continue;
        }
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _22._m2 = 0u;
        _22._m3 = 0u;
    }
    barrier();
    _22._m2 = 3u;
    _22._m3 = 1u;
    _22._m0[gl_LocalInvocationID.x]._m0 = vec4(float(_18._m0[gl_LocalInvocationID.x]));
    _22._m1[0u]._m0 = uvec3(0u, 1u, 2u);
}

void main()
{
    _152();
    barrier();
    uint _229 = min(_22._m2, 64u);
    uint _232 = min(_22._m3, 126u);
    SetMeshOutputsEXT(_229, _232);
    for (uint _223 = gl_LocalInvocationIndex; _223 < _229; _223 += 32u)
    {
        gl_MeshVerticesEXT[_223].gl_Position = _22._m0[_223]._m0;
    }
    for (uint _224 = gl_LocalInvocationIndex; _224 < _232; _224 += 32u)
    {
        gl_PrimitiveTriangleIndicesEXT[_224] = _22._m1[_224]._m0;
    }
}

