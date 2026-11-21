#version 460
layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

shared int _8[128];

void main()
{
    uint _22 = 0u;
    uvec2 _39 = uvec2(4294967295u);
    _22 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), _39)))
        {
            break;
        }
        _39 -= uvec2(uint(_39.y == 0u), 1u);
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
    int _61 = _8[gl_WorkGroupID.x];
    barrier();
    if (_61 > 10)
    {
        barrier();
        return;
    }
    else
    {
        return;
    }
}

