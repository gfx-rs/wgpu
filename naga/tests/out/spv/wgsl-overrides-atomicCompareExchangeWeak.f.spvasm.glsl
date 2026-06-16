#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    uint _m0;
    bool _m1;
};

shared uint _8;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _8 = 0u;
    }
    barrier();
    uint _27 = atomicCompSwap(_8, 2u, 1u);
}

