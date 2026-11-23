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

struct Inner_std140
{
    vec2 m1_0;
    vec2 m1_1;
    vec2 m2_0;
    vec2 m2_1;
    vec2 m2_2;
    vec2 m2_3;
};

struct Outer_std140
{
    vec4 a;
    Inner_std140 i;
};

layout(set = 0, binding = 0, std140) uniform u1
{
    Inner_std140 _m0;
} u1_1;

layout(set = 0, binding = 1, std140) uniform u2
{
    Outer_std140 _m0;
} u2_1;

layout(set = 0, binding = 2, std430) buffer o
{
    float _m0[16];
} o_1;

void main()
{
    mat4x2 _55 = mat4x2(u2_1._m0.i.m2_0, u2_1._m0.i.m2_1, u2_1._m0.i.m2_2, u2_1._m0.i.m2_3);
    o_1._m0[0u] = (((_55[0].x + _55[1].x) + _55[2].x) + _55[3].x) + mat2(u1_1._m0.m1_0, u1_1._m0.m1_1)[0].x;
}

