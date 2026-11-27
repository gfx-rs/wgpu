///////////////////////////////////
// Entry point: "ts_main" (task) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
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

struct _10
{
    uint _m0;
    uvec3 _m1;
};

struct _15
{
    _7 _m0[3];
    _9 _m1[1];
    uint _m2;
    uint _m3;
};

struct _16
{
    uint _m0;
    uvec3 _m1;
};

struct _59
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _17;

uvec3 _31()
{
    _10 _22 = _10(gl_LocalInvocationIndex, gl_WorkGroupID);
    if (gl_LocalInvocationIndex == 0u)
    {
        _17 = _4(0u);
    }
    barrier();
    if (_22._m0 == 0u)
    {
        _17._m0 = _22._m1.x + gl_SubgroupSize;
    }
    return uvec3(1u);
}

void main()
{
    uvec3 _56 = _31();
    barrier();
    _59 _66;
    umulExtended(_56.x, _56.y, _66._m1, _66._m0);
    _59 _68;
    umulExtended(_66._m0, _56.z, _68._m1, _68._m0);
    uvec3 _81 = ((((((_68._m0 > 1024u) || (_56.x > 256u)) || (_56.y > 256u)) || (_56.z > 256u)) || (_66._m1 != 0u)) || (_68._m1 != 0u)) ? uvec3(0u) : _56;
    EmitMeshTasksEXT(_81.x, _81.y, _81.z);
}


////////////////////////////////////
// Entry point: "ts_index" (task) //
////////////////////////////////////
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

struct _10
{
    uint _m0;
    uvec3 _m1;
};

struct _15
{
    _7 _m0[3];
    _9 _m1[1];
    uint _m2;
    uint _m3;
};

struct _16
{
    uint _m0;
    uvec3 _m1;
};

struct _59
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _17;

uvec3 _88()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _17 = _4(0u);
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u)
    {
        _17._m0 = 1u;
    }
    return uvec3(1u);
}

void main()
{
    uvec3 _100 = _88();
    barrier();
    _59 _107;
    umulExtended(_100.x, _100.y, _107._m1, _107._m0);
    _59 _109;
    umulExtended(_107._m0, _100.z, _109._m1, _109._m0);
    uvec3 _122 = ((((((_109._m0 > 1024u) || (_100.x > 256u)) || (_100.y > 256u)) || (_100.z > 256u)) || (_107._m1 != 0u)) || (_109._m1 != 0u)) ? uvec3(0u) : _100;
    EmitMeshTasksEXT(_122.x, _122.y, _122.z);
}


///////////////////////////////////
// Entry point: "ms_main" (mesh) //
///////////////////////////////////
#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_basic : require
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

struct _10
{
    uint _m0;
    uvec3 _m1;
};

struct _15
{
    _7 _m0[3];
    _9 _m1[1];
    uint _m2;
    uint _m3;
};

struct _16
{
    uint _m0;
    uvec3 _m1;
};

struct _59
{
    uint _m0;
    uint _m1;
};

taskPayloadSharedEXT _4 _17;
shared _15 _19;

void _142()
{
    _16 _127 = _16(gl_LocalInvocationIndex, gl_WorkGroupID);
    if (gl_LocalInvocationIndex == 0u)
    {
        _19 = _15(_7[](_7(vec4(0.0)), _7(vec4(0.0)), _7(vec4(0.0))), _9[](_9(uvec3(0u))), 0u, 0u);
    }
    barrier();
    if (_127._m0 == 0u)
    {
        _19._m2 = 3u;
        _19._m3 = 1u;
        _19._m1[0u]._m0 = uvec3(0u, 1u, 2u);
    }
    if (_127._m0 < 3u)
    {
        _19._m0[_127._m0]._m0 = vec4(float((_17._m0 + _127._m1.x) + gl_SubgroupInvocationID), 0.0, 0.0, 1.0);
        return;
    }
    else
    {
        return;
    }
}

void main()
{
    _142();
    barrier();
    uint _190 = min(_19._m2, 3u);
    uint _193 = min(_19._m3, 1u);
    SetMeshOutputsEXT(_190, _193);
    for (uint _183 = gl_LocalInvocationIndex; _183 < _190; _183 += 64u)
    {
        gl_MeshVerticesEXT[_183].gl_Position = _19._m0[_183]._m0;
    }
    for (uint _184 = gl_LocalInvocationIndex; _184 < _193; _184 += 64u)
    {
        gl_PrimitiveTriangleIndicesEXT[_184] = _19._m1[_184]._m0;
    }
}

