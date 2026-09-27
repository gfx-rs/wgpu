#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _5_6
{
    uint _m0[];
} _6;

layout(set = 0, binding = 1, std430) buffer _9_8
{
    uint _m0;
} _8;

shared uint _11;

void main()
{
    uint _24 = 0u;
    uvec2 _53 = uvec2(4294967295u);
    if (gl_LocalInvocationIndex == 0u)
    {
        _11 = 0u;
    }
    barrier();
    _6._m0[gl_LocalInvocationIndex] = gl_LocalInvocationIndex;
    memoryBarrierBuffer();
    if (gl_LocalInvocationIndex == 0u)
    {
        atomicExchange(_8._m0, 1u);
    }
    for (;;)
    {
        if (all(equal(uvec2(0u), _53)))
        {
            break;
        }
        _53 -= uvec2(uint(_53.y == 0u), 1u);
        uint _64 = atomicAdd(_8._m0, 0u);
        if (_64 != 0u)
        {
            memoryBarrierBuffer();
            _11 = _6._m0[0u];
            memoryBarrierShared();
            break;
        }
        _24++;
        if (_24 > 65536u)
        {
            break;
        }
        continue;
    }
}

