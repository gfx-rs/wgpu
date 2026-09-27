#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _5_7
{
    uint _m0[];
} _7;

layout(set = 0, binding = 1, std430) buffer _5_9
{
    uint _m0[];
} _9;

layout(set = 0, binding = 2, std430) buffer _5_10
{
    uint _m0[];
} _10;

layout(set = 0, binding = 3, std430) buffer _12_11
{
    uint _m0;
} _11;

layout(set = 0, binding = 4, rgba8ui) uniform readonly uimage2D _14;
layout(set = 0, binding = 5, rgba8ui) uniform writeonly uimage2D _16;

shared uint _17;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _17 = 0u;
    }
    barrier();
    _17 = _10._m0[0u];
    barrier();
    _7._m0[0u] = _17;
    groupMemoryBarrier();
    barrier();
    uint _53 = _9._m0[0u];
    _9._m0[1u] = _53;
    atomicExchange(_11._m0, 1u);
    uint _55 = atomicAdd(_11._m0, 0u);
    uint _56 = atomicAdd(_11._m0, 1u);
    imageStore(_16, ivec2(0), imageLoad(_14, ivec2(0)));
    memoryBarrierBuffer();
}

