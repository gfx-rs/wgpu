#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct _7
{
    uint _m0;
    int _m1[2];
};

shared uint _10;
shared int _12;
shared _7 _14;

void main()
{
    bool _34 = false;
    bool _37 = false;
    bool _39 = false;
    if (gl_LocalInvocationIndex < 2u)
    {
        _14._m1[gl_LocalInvocationIndex] = 0;
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        _10 = 0u;
        _12 = 0;
        _14._m0 = 0u;
    }
    barrier();
    uint _58 = atomicOr(_10, uint((gl_WorkGroupID.x + (gl_WorkGroupID.y * 32768u)) >= 64u));
    int _60 = atomicAdd(_12, 1);
    atomicExchange(_14._m0, 1u);
    int _62 = atomicAdd(_14._m1[0u], 1);
    barrier();
    barrier();
    uint _64 = atomicAdd(_10, 0u);
    barrier();
    barrier();
    int _65 = atomicAdd(_12, 0);
    barrier();
    barrier();
    uint _67 = atomicAdd(_14._m0, 0u);
    barrier();
    barrier();
    int _69 = atomicAdd(_14._m1[0u], 0);
    barrier();
    if (_64 == 0u)
    {
        _34 = _65 > 0;
    }
    else
    {
        _34 = false;
    }
    if (_34)
    {
        _37 = _67 > 0u;
    }
    else
    {
        _37 = false;
    }
    if (_37)
    {
        _39 = _69 > 0;
    }
    else
    {
        _39 = false;
    }
    if (_39)
    {
        return;
    }
    else
    {
        return;
    }
}

