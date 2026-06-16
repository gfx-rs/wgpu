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
    bool _31 = false;
    bool _34 = false;
    bool _36 = false;
    if (gl_LocalInvocationIndex == 0u)
    {
        _10 = 0u;
        _12 = 0;
        _14 = _7(0u, int[](0, 0));
    }
    barrier();
    uint _56 = atomicOr(_10, uint((gl_WorkGroupID.x + (gl_WorkGroupID.y * 32768u)) >= 64u));
    int _58 = atomicAdd(_12, 1);
    atomicExchange(_14._m0, 1u);
    int _61 = atomicAdd(_14._m1[0u], 1);
    barrier();
    barrier();
    uint _63 = atomicAdd(_10, 0u);
    barrier();
    barrier();
    int _64 = atomicAdd(_12, 0);
    barrier();
    barrier();
    uint _66 = atomicAdd(_14._m0, 0u);
    barrier();
    barrier();
    int _68 = atomicAdd(_14._m1[0u], 0);
    barrier();
    if (_63 == 0u)
    {
        _31 = _64 > 0;
    }
    else
    {
        _31 = false;
    }
    if (_31)
    {
        _34 = _66 > 0u;
    }
    else
    {
        _34 = false;
    }
    if (_34)
    {
        _36 = _68 > 0;
    }
    else
    {
        _36 = false;
    }
    if (_36)
    {
        return;
    }
    else
    {
        return;
    }
}

