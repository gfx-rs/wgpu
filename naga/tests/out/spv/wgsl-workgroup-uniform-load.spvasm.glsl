#version 460
layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

shared int _8[128];

void main()
{
    for (uint _28 = gl_LocalInvocationIndex; _28 < 128u; _28 += 4u)
    {
        _8[_28] = 0;
    }
    barrier();
    barrier();
    int _38 = _8[gl_WorkGroupID.x];
    barrier();
    if (_38 > 10)
    {
        barrier();
        return;
    }
    else
    {
        return;
    }
}

