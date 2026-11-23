#version 460
layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

shared int _8[128];

void main()
{
    uint _22 = 0u;
    _22 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (_22 >= 128u)
        {
            break;
        }
        _8[_22] = 0;
        _22 += 4u;
        continue;
    }
    barrier();
    barrier();
    int _42 = _8[gl_WorkGroupID.x];
    barrier();
    if (_42 > 10)
    {
        barrier();
        return;
    }
    else
    {
        return;
    }
}

