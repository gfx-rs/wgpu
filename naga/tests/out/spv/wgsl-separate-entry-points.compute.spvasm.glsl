#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _6()
{
    memoryBarrierBuffer();
    barrier();
    barrier();
    groupMemoryBarrier();
    barrier();
}

void main()
{
    _6();
}

