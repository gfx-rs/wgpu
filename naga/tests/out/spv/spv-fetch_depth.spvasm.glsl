#version 460
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    float _m0;
};

struct _8
{
    uvec2 _m0;
};

layout(set = 0, binding = 0, std430) buffer _12_11
{
    _5 _m0;
} _11;

layout(set = 0, binding = 1, std430) readonly buffer _15_14
{
    _8 _m0;
} _14;

layout(set = 0, binding = 2) uniform texture2D _17;

void _20()
{
    _11._m0._m0 = vec4(texelFetch(_17, ivec2(_14._m0._m0), 0).x).x;
}

void main()
{
    _20();
}

