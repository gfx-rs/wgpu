#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    uvec3 _m0;
    uint _m1;
};

shared uint _6;

void main()
{
    _5 _9 = _5(gl_LocalInvocationID, gl_LocalInvocationIndex);
    if (gl_LocalInvocationIndex == 0u)
    {
        _6 = 0u;
    }
    barrier();
    _6 = _9._m1 * 2u;
    _6 += _9._m0.x;
}

