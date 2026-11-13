#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Inner
{
    mat2 m1;
    mat4x2 m2;
};

struct Outer
{
    vec4 a;
    Inner i;
};

struct std140_Inner
{
    vec2 m1_col0;
    vec2 m1_col1;
    vec2 m2_col0;
    vec2 m2_col1;
    vec2 m2_col2;
    vec2 m2_col3;
};

struct std140_Outer
{
    vec4 a;
    std140_Inner i;
};

layout(set = 0, binding = 0, std140) uniform u1
{
    std140_Inner _m0;
} u1_1;

layout(set = 0, binding = 1, std140) uniform u2
{
    std140_Outer _m0;
} u2_1;

layout(set = 0, binding = 2, std430) buffer o
{
    float _m0[16];
} o_1;

void main()
{
    mat4x2 _60 = mat4x2(u2_1._m0.i.m2_col0, u2_1._m0.i.m2_col1, u2_1._m0.i.m2_col2, u2_1._m0.i.m2_col3);
    o_1._m0[0u] = (((_60[0].x + _60[1].x) + _60[2].x) + _60[3].x) + mat2(u1_1._m0.m1_col0, u1_1._m0.m1_col1)[0].x;
}

