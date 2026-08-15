#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _5_6
{
    uint _m0[];
} _6;

layout(set = 0, binding = 1, std430) buffer _5_8
{
    uint _m0[];
} _8;

layout(set = 0, binding = 2, std430) buffer _5_9
{
    uint _m0[];
} _9;

layout(set = 0, binding = 3, std430) buffer _11_10
{
    uint _m0;
} _10;

shared uint _13;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _13 = 0u;
    }
    barrier();
    _13 = _9._m0[0u];
    barrier();
    _6._m0[0u] = _13;
    memoryBarrierBuffer();
    barrier();
    _8._m0[1u] = _8._m0[0u];
    atomicExchange(_10._m0, 1u);
    uint _47 = atomicAdd(_10._m0, 0u);
    uint _48 = atomicAdd(_10._m0, 1u);
    memoryBarrierBuffer();
}

