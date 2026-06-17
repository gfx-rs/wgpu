///////////////////////////////////
// Entry point: "vs_main" (vert) //
///////////////////////////////////
#version 460

struct _6
{
    vec4 _m0;
    vec3 _m1;
};

struct _8
{
    mat4 _m0;
    mat4 _m1;
};

layout(set = 0, binding = 0, std140) uniform _15_14
{
    _8 _m0;
} _14;

layout(set = 0, binding = 1) uniform textureCube _17;
layout(set = 0, binding = 2) uniform sampler _19;

layout(location = 0) out vec3 _44;

int _21(int _23, int _24)
{
    return _23 / (((_24 == 0) || ((_23 == int(0x80000000)) && (_24 == (-1)))) ? 1 : _24);
}

void main()
{
    int _55 = 0;
    int _58 = 0;
    _55 = _21(int(uint(gl_VertexIndex)), 2);
    _58 = int(uint(gl_VertexIndex)) & 1;
    vec4 _73 = vec4((float(_55) * 4.0) - 1.0, (float(_58) * 4.0) - 1.0, 0.0, 1.0);
    _6 _94 = _6(_73, transpose(mat3(_14._m0._m1[0u].xyz, _14._m0._m1[1u].xyz, _14._m0._m1[2u].xyz)) * (_14._m0._m0 * _73).xyz);
    gl_Position = _94._m0;
    _44 = _94._m1;
}


///////////////////////////////////
// Entry point: "fs_main" (frag) //
///////////////////////////////////
#version 460

struct _6
{
    vec4 _m0;
    vec3 _m1;
};

struct _8
{
    mat4 _m0;
    mat4 _m1;
};

layout(set = 0, binding = 0, std140) uniform _15_14
{
    _8 _m0;
} _14;

layout(set = 0, binding = 1) uniform textureCube _17;
layout(set = 0, binding = 2) uniform sampler _19;

layout(location = 0) in vec3 _102;
layout(location = 0) out vec4 _105;

void main()
{
    _105 = texture(samplerCube(_17, _19), _6(gl_FragCoord, _102)._m1);
}

