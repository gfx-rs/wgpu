#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

shared int _4;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _4 = 0;
    }
    barrier();
    int _24 = atomicAdd(_4, -(-1));
}

