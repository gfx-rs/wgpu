#version 460
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

struct _7
{
    float _m0;
    float _m1[2];
};

layout(set = 0, binding = 0, std430) buffer _10_9
{
    float _m0;
} _9;

layout(set = 0, binding = 1, std430) buffer _13_12
{
    float _m0[2];
} _12;

layout(set = 0, binding = 2, std430) buffer _16_15
{
    _7 _m0;
} _15;

void main()
{
    atomicExchange(_9._m0, 1.5);
    atomicExchange(_12._m0[1u], 1.5);
    atomicExchange(_15._m0._m0, 1.5);
    atomicExchange(_15._m0._m1[1u], 1.5);
    barrier();
    float _40 = atomicAdd(_9._m0, 0);
    float _42 = atomicAdd(_12._m0[1u], 0);
    float _44 = atomicAdd(_15._m0._m0, 0);
    float _46 = atomicAdd(_15._m0._m1[1u], 0);
    barrier();
    float _47 = atomicAdd(_9._m0, 1.5);
    float _48 = atomicAdd(_12._m0[1u], 1.5);
    float _50 = atomicAdd(_15._m0._m0, 1.5);
    float _52 = atomicAdd(_15._m0._m1[1u], 1.5);
    barrier();
    float _54 = atomicExchange(_9._m0, 1.5);
    float _55 = atomicExchange(_12._m0[1u], 1.5);
    float _57 = atomicExchange(_15._m0._m0, 1.5);
    float _59 = atomicExchange(_15._m0._m1[1u], 1.5);
}

